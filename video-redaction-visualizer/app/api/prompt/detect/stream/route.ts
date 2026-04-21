import { annotateFrame } from "@/lib/server/frame-annotate";
import { applyShrinkInPlace, shrinkBoxesOnFrame } from "@/lib/server/box-shrink";
import { createEntry, putFrame, type FrameState } from "@/lib/server/gemini-cache";
import { fetchDeduplicatedFramesServer } from "@/lib/server/frames";
import { fetchOcrRaw } from "@/lib/server/ocr-client";
import { curateFramePrompt } from "@/lib/server/prompt/curator";
import { precomputeFrame } from "@/lib/server/prompt/resolver";
import { hasUnresolvedPredicate } from "@/lib/server/prompt/types";
import {
  ndjsonStreamResponse,
  readPromptFormInputs,
} from "@/lib/server/route-helpers";
import { openRunLog } from "@/lib/server/run-log";

export const runtime = "nodejs";
export const maxDuration = 600;

function parseIntEnv(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : fallback;
}

const PROMPT_DETECT_CONCURRENCY = parseIntEnv("PROMPT_DETECT_CONCURRENCY", 8);

function json(detail: string, status: number): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function POST(req: Request): Promise<Response> {
  const parsed = await readPromptFormInputs(req);
  if ("error" in parsed) return json(parsed.error, 400);

  const predicate = parsed.predicate;
  if (!predicate) {
    return json("Missing predicate_json. Run /api/prompt/plan first.", 400);
  }
  if (hasUnresolvedPredicate(predicate)) {
    return json(
      "Predicate still has unresolved leaves. Complete clarifications in /api/prompt/plan first.",
      400,
    );
  }

  const prompt = parsed.prompt;
  const fps = parsed.fps;
  const dedupThreshold = parsed.dedupThreshold;
  const maxGap = parsed.maxGap;

  let framesRes;
  try {
    framesRes = await fetchDeduplicatedFramesServer({
      file: parsed.file,
      fps: fps ?? undefined,
      dedupThreshold,
      maxGap,
    });
  } catch (e) {
    return json(e instanceof Error ? e.message : "frame extraction failed", 502);
  }

  const n = framesRes.deduplicatedCount;
  if (n === 0) return json("No frames to process.", 422);

  const frameFrom = Math.max(1, Math.min(n, parsed.frameFrom));
  const frameTo =
    parsed.frameTo == null ? n : Math.max(1, Math.min(n, parsed.frameTo));
  if (frameFrom > frameTo) {
    return json(`Invalid frame range: from=${frameFrom} to=${frameTo}`, 400);
  }
  const lo = frameFrom;
  const hi = frameTo;

  let ocr;
  try {
    ocr = await fetchOcrRaw(parsed.file, {
      frameFrom: lo,
      frameTo: hi,
      fps: fps ?? null,
      dedupThreshold,
      maxGap,
    });
  } catch (e) {
    return json(e instanceof Error ? e.message : "Python OCR raw stream failed", 502);
  }

  const ocrByIndex = new Map<number, (typeof ocr.frames)[number]>();
  for (const frame of ocr.frames) ocrByIndex.set(frame.index, frame);

  const predicateHash = parsed.predicateHash;
  if (!predicateHash) {
    return json("Missing or invalid predicate hash.", 400);
  }

  const entry = createEntry({
    engine: "prompt",
    videoHash: framesRes.videoHash,
    predicateHash,
    fps: fps ?? null,
    dedupThreshold,
    maxGap,
    frameFrom: lo,
    frameTo: hi,
  });

  const indices: number[] = [];
  for (let i = lo; i <= hi; i++) indices.push(i);

  const runLog = openRunLog("prompt-detect", {
    query: prompt,
    prompt,
    predicate,
    predicate_hash: predicateHash,
    video_hash: framesRes.videoHash,
    frame_from: lo,
    frame_to: hi,
    total_frames: indices.length,
    fps,
    dedup_threshold: dedupThreshold,
    max_gap: maxGap,
    ocr_frames_returned: ocr.frames.length,
  });

  return ndjsonStreamResponse(async ({ emit, error }) => {
    emit({
      type: "start",
      video_hash: framesRes.videoHash,
      query: prompt,
      deduplicated_count: n,
      frame_from: lo,
      frame_to: hi,
      frame_indices: indices,
      total: indices.length,
      run_log_path: runLog.path,
    });

    let matchedFrames = 0;
    let totalBoxes = 0;
    const t0 = Date.now();

    const concurrency = Math.max(1, PROMPT_DETECT_CONCURRENCY);
    let cursor = 0;

    const worker = async () => {
      while (true) {
        const myIdx = cursor++;
        if (myIdx >= indices.length) return;
        const idx1 = indices[myIdx];
        const src = framesRes.frames[idx1 - 1];
        if (!src) continue;
        const rawFrame = ocrByIndex.get(idx1);
        const raw = rawFrame?.raw ?? null;

        try {
          const pre = await precomputeFrame({
            videoHash: framesRes.videoHash,
            frameIndex: idx1,
            predicate,
            rawOcr: raw,
            frameImage: src.blob,
            width: src.width,
            height: src.height,
            runLog,
          });

          const curated = await curateFramePrompt({
            videoHash: framesRes.videoHash,
            frameIndex: idx1,
            jpeg: src.blob,
            width: src.width,
            height: src.height,
            predicate,
            regions: pre.regions,
            textCandidates: pre.textCandidates,
            rawOcr: raw,
            runLog,
          });

          const finalBoxes = [...curated.kept, ...curated.added].map((b) => ({
            ...b,
            track_id: b.instance_id ?? b.track_id,
          }));

          const state: FrameState = {
            width: src.width,
            height: src.height,
            matched: finalBoxes.map((b) => ({ ...b })),
            raw,
            blob: src.blob,
            ocrRaw: raw,
            ocrMatched: [],
            flagged: false,
            instances: curated.instances,
            regions: pre.regions,
          };
          putFrame(entry, idx1, state);

          if (finalBoxes.length > 0) matchedFrames += 1;
          totalBoxes += finalBoxes.length;

          runLog.write({
            kind: "frame_result",
            frame_index: idx1,
            kept_count: curated.kept.length,
            added_count: curated.added.length,
            dropped_count: curated.dropped.length,
            final_box_count: finalBoxes.length,
            region_count: pre.regions.length,
            text_candidate_count: pre.textCandidates.length,
            box_texts: finalBoxes.map((b) => b.text),
          });

          emit({
            type: "frame",
            index: idx1,
            width: src.width,
            height: src.height,
            boxes: finalBoxes.map((b) => ({
              x: b.x,
              y: b.y,
              w: b.w,
              h: b.h,
              text: b.text,
              score: Math.round((b.score ?? 1) * 1000) / 1000,
              origin: b.origin,
              label: b.label,
              track_id: b.instance_id ?? b.track_id,
              instance_id: b.instance_id,
              branch: b.branch,
              category: b.category ?? undefined,
            })),
            raw: {
              ocr: raw,
              curator: curated.raw,
              regions: pre.regions,
            },
          });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          runLog.write({
            kind: "frame_error",
            frame_index: idx1,
            error: msg,
          });
          error(`Prompt detect failed on frame ${idx1}: ${msg}`);
          putFrame(entry, idx1, {
            width: src.width,
            height: src.height,
            matched: [],
            raw,
            blob: src.blob,
            ocrRaw: raw,
            ocrMatched: [],
            flagged: false,
            instances: [],
            regions: [],
          });
        }
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(concurrency, indices.length) }, () => worker()),
    );

    // ---- Box-shrink post-processing ---------------------------------
    // Prompt-mode keep_box + add_box tools stamp text_color_hex and
    // background_color_hex onto every emitted box. This pass samples
    // corner pixels (3x3 window) and iteratively trims sides whose 2
    // corners both read as background. Boxes that came from the
    // deterministic / text-fastpath fallbacks (no colors) are passed
    // through unchanged. Runs before the annotated-frame dump so the
    // debug JPEGs show the tightened coords.
    const shrinkT0 = Date.now();
    let shrinkFramesProcessed = 0;
    let shrinkBoxesChanged = 0;
    let shrinkBoxesInspected = 0;
    for (const idx1 of indices) {
      const state = entry.perFrame[idx1];
      if (!state || state.matched.length === 0) continue;
      try {
        const results = await shrinkBoxesOnFrame(state.blob, state.matched);
        const changed = applyShrinkInPlace(results);
        shrinkBoxesInspected += results.length;
        shrinkBoxesChanged += changed;
        shrinkFramesProcessed += 1;
        if (changed > 0) {
          runLog.write({
            kind: "box_shrink_frame",
            frame_index: idx1,
            inspected: results.length,
            changed,
            deltas: results
              .filter((r) => r.changed)
              .map((r) => ({
                text: r.box.text,
                trimmed: r.trimmed,
                reason: r.reason,
                text_color_hex: r.box.text_color_hex,
                background_color_hex: r.box.background_color_hex,
              })),
          });
          emit({
            type: "post_process",
            phase: "box_shrink",
            index: idx1,
            width: state.width,
            height: state.height,
            boxes: state.matched.map((b) => ({
              x: b.x,
              y: b.y,
              w: b.w,
              h: b.h,
              text: b.text,
              score: Math.round((b.score ?? 1) * 1000) / 1000,
              origin: b.origin,
              track_id: b.instance_id ?? b.track_id,
              instance_id: b.instance_id,
              branch: b.branch,
              category: b.category ?? undefined,
              text_color_hex: b.text_color_hex,
              background_color_hex: b.background_color_hex,
            })),
          });
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        runLog.write({
          kind: "box_shrink_error",
          frame_index: idx1,
          error: msg,
        });
      }
    }
    runLog.write({
      kind: "box_shrink_summary",
      frames_processed: shrinkFramesProcessed,
      boxes_inspected: shrinkBoxesInspected,
      boxes_changed: shrinkBoxesChanged,
      elapsed_ms: Date.now() - shrinkT0,
    });

    // ---- Annotated-frame dump ----------------------------------------
    // Mirror the teamwork detect route: once every per-frame curator
    // call has settled, render each frame's final box set onto the
    // source JPEG and drop it in `<runId>-frames/` for visual review.
    // Failures are best-effort (annotation is a debug artifact).
    if (runLog.enabled && runLog.framesDir) {
      const annotateT0 = Date.now();
      const padWidth = Math.max(3, String(hi).length);
      const pad = (idx: number): string => String(idx).padStart(padWidth, "0");
      let annotatedFrames = 0;
      for (const idx1 of indices) {
        const state = entry.perFrame[idx1];
        if (!state) continue;
        try {
          const annotated = await annotateFrame(
            state.blob,
            state.matched.map((b) => ({
              x: b.x,
              y: b.y,
              w: b.w,
              h: b.h,
              label: b.text,
              origin: b.origin ?? "ocr",
            })),
          );
          runLog.writeFrame(`frame-${pad(idx1)}.jpg`, annotated);
          annotatedFrames += 1;
        } catch (e) {
          console.warn(
            `[prompt-detect] annotated frame dump failed for idx=${idx1}:`,
            e instanceof Error ? e.message : String(e),
          );
        }
      }
      runLog.write({
        kind: "annotated_frames_written",
        frames_written: annotatedFrames,
        frames_dir: runLog.framesDir,
        elapsed_ms: Date.now() - annotateT0,
      });
    }

    const elapsed = Date.now() - t0;
    runLog.write({
      kind: "run_end",
      elapsed_ms: elapsed,
      matched_frames: matchedFrames,
      total_boxes: totalBoxes,
    });
    await runLog.close();

    emit({
      type: "done",
      matched_frames: matchedFrames,
      total_boxes: totalBoxes,
      run_log_path: runLog.path,
    });
  });
}

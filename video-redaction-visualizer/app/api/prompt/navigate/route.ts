import {
  type NavFrameState,
  type NavHit,
  type NavigatorEvent,
} from "@/lib/server/agentic-navigator";
import { aerr, alog } from "@/lib/server/agentic-log";
import { applyShrinkInPlace, shrinkBoxesOnFrame } from "@/lib/server/box-shrink";
import { annotateFrame } from "@/lib/server/frame-annotate";
import { fetchDeduplicatedFramesServer } from "@/lib/server/frames";
import { getEntry } from "@/lib/server/gemini-cache";
import { runPromptCascade } from "@/lib/server/prompt/cascade";
import {
  ndjsonStreamResponse,
  readPromptFormInputs,
} from "@/lib/server/route-helpers";
import { openRunLog } from "@/lib/server/run-log";

export const runtime = "nodejs";
export const maxDuration = 600;

function json(detail: string, status: number): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function POST(req: Request): Promise<Response> {
  const parsed = await readPromptFormInputs(req);
  if ("error" in parsed) return json(parsed.error, 400);
  if (!parsed.predicate || !parsed.predicateHash) {
    return json("Missing predicate_json. Run /api/prompt/detect/stream first.", 400);
  }
  const predicate = parsed.predicate;

  let framesRes;
  try {
    framesRes = await fetchDeduplicatedFramesServer({
      file: parsed.file,
      fps: parsed.fps ?? undefined,
      dedupThreshold: parsed.dedupThreshold,
      maxGap: parsed.maxGap,
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

  const entry = getEntry({
    engine: "prompt",
    videoHash: framesRes.videoHash,
    predicateHash: parsed.predicateHash,
    fps: parsed.fps ?? null,
    dedupThreshold: parsed.dedupThreshold,
    maxGap: parsed.maxGap,
    frameFrom: lo,
    frameTo: hi,
  });
  if (!entry) {
    return json(
      "No prompt cache entry for this predicate/range. Run /api/prompt/detect/stream first.",
      404,
    );
  }

  const runLog = openRunLog("prompt-navigate", {
    query: parsed.prompt,
    prompt: parsed.prompt,
    predicate,
    predicate_hash: parsed.predicateHash,
    video_hash: framesRes.videoHash,
    frame_from: lo,
    frame_to: hi,
    total_frames: hi - lo + 1,
    fps: parsed.fps,
    dedup_threshold: parsed.dedupThreshold,
    max_gap: parsed.maxGap,
  });

  return ndjsonStreamResponse(async ({ emit, error }) => {
    alog("prompt navigate route start", {
      video_hash: framesRes.videoHash,
      prompt: parsed.prompt,
      predicate_hash: parsed.predicateHash,
      frame_from: lo,
      frame_to: hi,
      total_frames: hi - lo + 1,
      run_log_path: runLog.path,
    });

    emit({
      type: "start",
      video_hash: framesRes.videoHash,
      query: parsed.prompt,
      frame_from: lo,
      frame_to: hi,
      total_frames: hi - lo + 1,
      run_log_path: runLog.path,
    });

    const navFrames: Array<NavFrameState & { regions?: unknown }> = [];
    for (let idx1 = lo; idx1 <= hi; idx1++) {
      const f = entry.perFrame[idx1];
      if (!f) continue;
      const hits: NavHit[] = f.matched.map((b, i) => ({
        ...b,
        hit_id: `P${idx1}-${i}`,
      }));
      navFrames.push({
        index: idx1,
        width: f.width,
        height: f.height,
        blob: f.blob,
        hits,
        ocrBoxes: f.ocrMatched ? f.ocrMatched.map((b) => ({ ...b })) : [],
        ocrRaw: f.ocrRaw ?? null,
        regions: f.regions,
      });
    }

    if (navFrames.length === 0) {
      runLog.write({ kind: "run_end", status: "no_frames" });
      await runLog.close();
      emit({
        type: "done",
        added_boxes: 0,
        removed_boxes: 0,
        total_steps: 0,
        run_log_path: runLog.path,
      });
      return;
    }

    const forward = (ev: NavigatorEvent): void => {
      runLog.write({ kind: "navigator_event", event: ev });
      switch (ev.type) {
        case "agent_start":
          emit({
            type: "agent_start",
            agent_id: ev.agent_id,
            focus_frame: ev.focus_frame,
            source: ev.source,
            parent_agent_id: ev.parent_agent_id,
            reason: ev.reason,
          });
          return;
        case "agent_end":
          emit({
            type: "agent_end",
            agent_id: ev.agent_id,
            focus_frame: ev.focus_frame,
            added: ev.added,
            removed: ev.removed,
            total_steps: ev.total_steps,
            finish_summary: ev.finish_summary,
            cost_usd: ev.cost_usd ?? 0,
          });
          return;
        case "frame_update":
          emit({
            type: "frame_update",
            index: ev.index,
            action: ev.action,
            box: {
              x: ev.hit.x,
              y: ev.hit.y,
              w: ev.hit.w,
              h: ev.hit.h,
              text: ev.hit.text,
              score: Math.round((ev.hit.score ?? 1) * 1000) / 1000,
              label: ev.hit.label,
              hit_id: ev.hit.hit_id,
              origin: ev.hit.origin,
              track_id: ev.hit.instance_id ?? ev.hit.track_id,
              instance_id: ev.hit.instance_id,
              branch: ev.hit.branch,
              category: ev.hit.category,
            },
            reason: ev.reason ?? null,
            agent_id: ev.agent_id ?? null,
          });
          return;
        case "tool_call":
          emit({
            type: "tool_call",
            step: ev.step,
            name: ev.name,
            input: ev.input,
            agent_id: ev.agent_id ?? null,
          });
          return;
        case "tool_result":
          emit({
            type: "tool_result",
            step: ev.step,
            name: ev.name,
            summary: ev.summary,
            agent_id: ev.agent_id ?? null,
          });
          return;
        case "model_text":
          emit({
            type: "model_text",
            step: ev.step,
            text: ev.text,
            agent_id: ev.agent_id ?? null,
          });
          return;
        case "finish":
          emit({
            type: "finish",
            summary: ev.summary,
            total_steps: ev.total_steps,
          });
          return;
      }
    };

    const t0 = Date.now();
    try {
      const result = await runPromptCascade({
        predicate,
        frames: navFrames,
        onEvent: forward,
        runLog,
      });

      for (const nf of navFrames) {
        const f = entry.perFrame[nf.index];
        if (!f) continue;
        f.matched = nf.hits.map((h) => {
          const rest = { ...h };
          delete (rest as { hit_id?: string }).hit_id;
          return rest;
        });
      }

      // ---- Box-shrink post-processing -------------------------------
      // Corner-constraint tightening using the text_color_hex /
      // background_color_hex labels the prompt cascade recovery + tool
      // calls stamp on every emitted box. Runs once after the cascade
      // settles so the annotated-frame dump and cache reflect
      // tightened coords.
      const shrinkT0 = Date.now();
      let shrinkFramesProcessed = 0;
      let shrinkBoxesChanged = 0;
      let shrinkBoxesInspected = 0;
      for (let idx1 = lo; idx1 <= hi; idx1++) {
        const state = entry.perFrame[idx1];
        if (!state || state.matched.length === 0) continue;
        try {
          const results = await shrinkBoxesOnFrame(
            state.blob,
            state.matched,
          );
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
          aerr(`prompt navigate box shrink failed for idx=${idx1}`, e);
        }
      }
      const shrinkElapsed = Date.now() - shrinkT0;
      runLog.write({
        kind: "box_shrink_summary",
        frames_processed: shrinkFramesProcessed,
        boxes_inspected: shrinkBoxesInspected,
        boxes_changed: shrinkBoxesChanged,
        elapsed_ms: shrinkElapsed,
      });
      alog("prompt navigate route box_shrink done", {
        frames_processed: shrinkFramesProcessed,
        boxes_inspected: shrinkBoxesInspected,
        boxes_changed: shrinkBoxesChanged,
        elapsed_ms: shrinkElapsed,
      });

      // ---- Annotated-frame dump -------------------------------------
      // Mirror the teamwork navigate route: dump every frame in range
      // with its post-navigate box set into `<runId>-frames/`. Best
      // effort — a single frame failure must not abort the run.
      if (runLog.enabled && runLog.framesDir) {
        const annotateT0 = Date.now();
        const padWidth = Math.max(3, String(hi).length);
        const pad = (idx: number): string => String(idx).padStart(padWidth, "0");
        let annotatedFrames = 0;
        for (let idx1 = lo; idx1 <= hi; idx1++) {
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
            aerr(`prompt navigate annotated frame dump failed for idx=${idx1}`, e);
          }
        }
        const annotateElapsed = Date.now() - annotateT0;
        runLog.write({
          kind: "annotated_frames_written",
          frames_written: annotatedFrames,
          frames_dir: runLog.framesDir,
          elapsed_ms: annotateElapsed,
        });
        alog("prompt navigate route annotated frames written", {
          frames_written: annotatedFrames,
          frames_dir: runLog.framesDir,
          elapsed_ms: annotateElapsed,
        });
      }

      const totalElapsed = Date.now() - t0;
      runLog.write({
        kind: "run_end",
        status: "ok",
        elapsed_ms: totalElapsed,
        added_boxes: result.added,
        removed_boxes: result.removed,
        total_steps: result.totalSteps,
        finish_summary: result.finishSummary,
      });
      await runLog.close();

      alog("prompt navigate route done", {
        elapsed_ms: totalElapsed,
        added_boxes: result.added,
        removed_boxes: result.removed,
        total_steps: result.totalSteps,
        finish_summary: result.finishSummary,
      });

      emit({
        type: "done",
        added_boxes: result.added,
        removed_boxes: result.removed,
        total_steps: result.totalSteps,
        run_log_path: runLog.path,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      runLog.write({
        kind: "run_end",
        status: "error",
        elapsed_ms: Date.now() - t0,
        error: msg,
      });
      await runLog.close();
      aerr("prompt navigate route threw", e);
      error(`Prompt navigator failed: ${msg}`);
    }
  });
}

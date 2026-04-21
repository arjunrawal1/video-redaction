// Agentic teamwork phase-1 (per-frame curator) + phase-1.5 (linker).
//
// Pipeline for each requested frame:
//   1. Python OCR (Textract) has already produced candidate boxes for this
//      frame. We fan those out per-frame (parallel).
//   2. For each frame, ask Gemini 3 Pro: keep/drop every OCR box, and
//      propose any box OCR missed entirely.
//   3. Stream the curated result to the client as soon as the per-frame
//      Gemini call resolves (out-of-order completion is fine; events carry
//      their own frame `index`).
//   4. After every curator call resolves, run the phase-1.5 LINKER
//      sequentially over adjacent scanned pairs. Gemini judges "is
//      box #i in frame N the same redaction as box #j in frame N+1",
//      and we stamp a `track_id` onto each box. The client uses
//      track_id to interpolate (tween) box coordinates smoothly
//      between sparse scanned keyframes at render time.
//
// Phase-2 (the navigator) runs afterward and can still add/remove boxes;
// the navigator does NOT consume track_ids today, but any boxes left
// untouched keep theirs, and `fix` boxes it adds simply arrive without
// an id (client falls back to the deterministic assignLabels pass).

import { curateFrame } from "@/lib/server/agentic-curator";
import { aerr, alog } from "@/lib/server/agentic-log";
import {
  linkFramePair,
  type LinkDecision,
} from "@/lib/server/agentic-linker";
import {
  addUsage,
  emptyUsage,
  formatTokens,
  formatUSD,
  geminiCost,
  textractCost,
} from "@/lib/server/cost";
import { fetchDeduplicatedFramesServer } from "@/lib/server/frames";
import {
  createEntry,
  putFrame,
  type FrameState,
} from "@/lib/server/gemini-cache";
import { linkFramePairFallback } from "@/lib/server/linker-fallback";
import { fetchOcrDetect } from "@/lib/server/ocr-client";
import {
  agenticCuratorConcurrency,
  agenticLinkerModelId,
  agenticModelId,
  type ServerBox,
} from "@/lib/server/openrouter";
import {
  ndjsonStreamResponse,
  normalizeQuery,
  readFormInputs,
} from "@/lib/server/route-helpers";
import { openRunLog } from "@/lib/server/run-log";

export const runtime = "nodejs";
export const maxDuration = 600;

export async function POST(req: Request): Promise<Response> {
  const parsed = await readFormInputs(req);
  if ("error" in parsed) return json(parsed.error, 400);

  const { file, query, fps, dedupThreshold } = parsed;
  let { frameFrom, frameTo } = parsed;
  const qNorm = normalizeQuery(query);

  let framesRes;
  try {
    framesRes = await fetchDeduplicatedFramesServer({
      file,
      fps: fps ?? undefined,
      dedupThreshold,
    });
  } catch (e) {
    return json(e instanceof Error ? e.message : "frame extraction failed", 502);
  }
  const n = framesRes.deduplicatedCount;
  if (n === 0) return json("No frames to process.", 422);

  frameFrom = Math.max(1, Math.min(n, frameFrom));
  frameTo = frameTo == null ? n : Math.max(1, Math.min(n, frameTo));
  if (frameFrom > frameTo) {
    return json(`Invalid frame range: from=${frameFrom} to=${frameTo}`, 400);
  }
  const lo = frameFrom;
  const hi = frameTo;

  let ocr;
  try {
    ocr = await fetchOcrDetect(file, query, {
      frameFrom: lo,
      frameTo: hi,
      fps: fps ?? null,
      dedupThreshold,
    });
  } catch (e) {
    return json(
      e instanceof Error ? e.message : "Python OCR detect failed",
      502,
    );
  }

  const ocrByIndex = new Map<number, (typeof ocr.frames)[number]>();
  for (const f of ocr.frames) ocrByIndex.set(f.index, f);

  const entry = createEntry({
    engine: "teamwork",
    videoHash: framesRes.videoHash,
    queryNorm: qNorm,
    fps: fps ?? null,
    dedupThreshold,
    frameFrom: lo,
    frameTo: hi,
  });

  const indices: number[] = [];
  for (let i = lo; i <= hi; i++) indices.push(i);

  const runLog = openRunLog("detect", {
    video_hash: framesRes.videoHash,
    query,
    query_norm: qNorm,
    frame_from: lo,
    frame_to: hi,
    total_frames: indices.length,
    model_curator: agenticModelId(),
    model_linker: agenticLinkerModelId(),
    ocr_frames_returned: ocr.frames.length,
  });

  return ndjsonStreamResponse(async ({ emit, error }) => {
    alog("detect route start", {
      video_hash: framesRes.videoHash,
      query,
      query_norm: qNorm,
      frame_from: lo,
      frame_to: hi,
      total_frames: indices.length,
      ocr_frames_returned: ocr.frames.length,
      concurrency: agenticCuratorConcurrency(),
      run_log_path: runLog.path,
    });

    emit({
      type: "start",
      video_hash: framesRes.videoHash,
      query,
      deduplicated_count: n,
      frame_from: lo,
      frame_to: hi,
      frame_indices: indices,
      total: indices.length,
      run_log_path: runLog.path,
    });

    if (!qNorm) {
      await runLog.close();
      emit({ type: "done", matched_frames: 0, total_boxes: 0 });
      return;
    }

    // Accumulators are updated from the async worker pool; we rely on the
    // single-threaded event loop to make these writes safe.
    let matchedFrames = 0;
    let totalBoxes = 0;
    const curatorUsage = emptyUsage();
    // Running cost total — streamed to the client after every priced
    // event so the UI can display a live "Running: $X.XX" counter.
    const ocrRunningBill = textractCost(ocr.frames.length);
    let runningUSD = ocrRunningBill.totalUSD;
    // Fire an initial cost event so the UI shows the OCR floor the
    // moment the route starts processing frames (before any Gemini
    // calls land). Textract cost is fully known up-front — it's just
    // pages × $0.0015.
    emit({
      type: "cost_update",
      phase: "ocr",
      call_usd: ocrRunningBill.totalUSD,
      running_usd: runningUSD,
      breakdown: {
        ocr_pages: ocr.frames.length,
        ocr_usd: ocrRunningBill.totalUSD,
        curator_calls: 0,
        curator_usd: 0,
        linker_calls: 0,
        linker_usd: 0,
      },
    });
    const t0 = Date.now();

    const concurrency = Math.max(1, agenticCuratorConcurrency());
    let cursor = 0;

    const worker = async () => {
      while (true) {
        const myIdx = cursor++;
        if (myIdx >= indices.length) return;
        const idx1 = indices[myIdx];
        const src = framesRes.frames[idx1 - 1];
        if (!src) continue;
        const ocrFrame = ocrByIndex.get(idx1);
        const ocrBoxes: ServerBox[] = ocrFrame?.boxes ?? [];

        let kept: ServerBox[] = [];
        let added: ServerBox[] = [];
        let dropped: number[] = [];
        let curatorRaw: unknown = null;
        try {
          const r = await curateFrame({
            jpeg: src.blob,
            query: qNorm,
            frameIndex: idx1,
            frameWidth: src.width,
            frameHeight: src.height,
            ocrBoxes,
            ocrRaw: ocrFrame?.raw ?? null,
            runLog,
          });
          kept = r.kept;
          added = r.added;
          dropped = r.dropped;
          curatorRaw = r.raw;
          addUsage(curatorUsage, r.usage);
          const callBill = geminiCost(agenticModelId(), {
            inputTokens: r.usage?.inputTokens ?? 0,
            outputTokens: r.usage?.outputTokens ?? 0,
            reasoningTokens: r.usage?.outputTokenDetails?.reasoningTokens ?? 0,
            cachedInputTokens: r.usage?.inputTokenDetails?.cacheReadTokens ?? 0,
            callCount: 1,
          });
          runningUSD += callBill.totalUSD;
          const bill = geminiCost(agenticModelId(), curatorUsage);
          emit({
            type: "cost_update",
            phase: "curator",
            frame_index: idx1,
            call_usd: callBill.totalUSD,
            running_usd: runningUSD,
            breakdown: {
              ocr_pages: ocr.frames.length,
              ocr_usd: ocrRunningBill.totalUSD,
              curator_calls: bill.callCount,
              curator_usd: bill.totalUSD,
              curator_input_tokens: bill.inputTokens,
              curator_output_tokens: bill.outputTokens,
              curator_cached_input_tokens: bill.cachedInputTokens,
              linker_calls: 0,
              linker_usd: 0,
            },
          });
        } catch (e) {
          // Curator failed — fall back to OCR boxes verbatim so we don't
          // lose redaction coverage on a Gemini hiccup.
          error(
            `Curator failed on frame ${idx1}: ${e instanceof Error ? e.message : String(e)}`,
          );
          kept = ocrBoxes.map((b) => ({ ...b }));
        }

        const finalBoxes: ServerBox[] = [
          ...kept.map((b) => ({ ...b, origin: undefined })),
          ...added,
        ];

        const state: FrameState = {
          width: src.width,
          height: src.height,
          matched: finalBoxes.map((b) => ({ ...b })),
          raw: ocrFrame?.raw ?? null,
          blob: src.blob,
          ocrRaw: ocrFrame?.raw ?? null,
          ocrMatched: ocrBoxes.map((b) => ({ ...b })),
          flagged: false,
        };
        putFrame(entry, idx1, state);

        if (finalBoxes.length > 0) matchedFrames += 1;
        totalBoxes += finalBoxes.length;

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
            score: Math.round(b.score * 1000) / 1000,
            origin: b.origin,
          })),
          raw: {
            ocr: ocrFrame?.raw ?? null,
            curator: curatorRaw,
            dropped_ocr_indices: dropped,
          },
        });
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(concurrency, indices.length) }, () =>
        worker(),
      ),
    );

    const curatorElapsed = Date.now() - t0;

    // ---- Phase 1.5: linker ------------------------------------------
    // Walk adjacent scanned pairs in order and ask Gemini to match
    // each frame N+1 box back to a frame N box (or mark it as a new
    // redaction). Sequential: track ids in N+1 depend on N. The
    // linker fallback kicks in per-pair on any Gemini failure so one
    // bad pair can't corrupt the whole chain.
    const linkerT0 = Date.now();
    const linkerUsage = emptyUsage();
    let linkerCalls = 0;
    let linkerFallbacks = 0;
    let nextTrackId = 0;
    const mintTrack = (): string => `t${nextTrackId++}`;

    // Chain state: the most-recently-processed scanned frame that had
    // boxes. When the range has unscanned gaps (missing source frame)
    // or a zero-box frame we reset this to null so re-appearances get
    // fresh ids — same strict-adjacency semantics as the old client
    // assignLabels pass.
    type PrevLink = {
      index: number;
      boxes: ServerBox[];
      trackIds: string[];
      jpeg: Uint8Array;
      width: number;
      height: number;
    };
    let prev: PrevLink | null = null;

    for (const idx1 of indices) {
      const state = entry.perFrame[idx1];
      if (!state) {
        // Unscanned frame: break the chain.
        prev = null;
        continue;
      }
      const boxes = state.matched;

      if (boxes.length === 0) {
        emit({
          type: "link",
          index: idx1,
          track_ids: [],
          links: [],
        });
        // Break the chain: a gap of zero-box frames is indistinguishable
        // from content disappearing, and forcing fresh ids on the next
        // populated frame avoids teleporting a track across dead space.
        prev = null;
        continue;
      }

      let decisions: LinkDecision[];
      let fallback = false;

      if (!prev) {
        // First populated frame in a chain — every box is a new track.
        decisions = boxes.map((_, bi) => ({
          b_index: bi,
          a_index: null,
          reason: "chain start",
        }));
      } else {
        try {
          const r = await linkFramePair({
            jpegA: prev.jpeg,
            jpegB: state.blob,
            frameIndexA: prev.index,
            frameIndexB: idx1,
            frameWidthA: prev.width,
            frameHeightA: prev.height,
            frameWidthB: state.width,
            frameHeightB: state.height,
            boxesA: prev.boxes,
            boxesB: boxes,
            runLog,
          });
          decisions = r.links;
          addUsage(linkerUsage, r.usage);
          if (r.usage != null) {
            linkerCalls += 1;
            const callBill = geminiCost(agenticLinkerModelId(), {
              inputTokens: r.usage.inputTokens ?? 0,
              outputTokens: r.usage.outputTokens ?? 0,
              reasoningTokens: r.usage.outputTokenDetails?.reasoningTokens ?? 0,
              cachedInputTokens: r.usage.inputTokenDetails?.cacheReadTokens ?? 0,
              callCount: 1,
            });
            runningUSD += callBill.totalUSD;
            const cBill = geminiCost(agenticModelId(), curatorUsage);
            const lBill = geminiCost(agenticLinkerModelId(), linkerUsage);
            emit({
              type: "cost_update",
              phase: "linker",
              frame_pair: [prev.index, idx1],
              call_usd: callBill.totalUSD,
              running_usd: runningUSD,
              breakdown: {
                ocr_pages: ocr.frames.length,
                ocr_usd: ocrRunningBill.totalUSD,
                curator_calls: cBill.callCount,
                curator_usd: cBill.totalUSD,
                linker_calls: lBill.callCount,
                linker_usd: lBill.totalUSD,
              },
            });
          }
        } catch (e) {
          aerr(`linker pair #${prev.index}→#${idx1} fell back`, e);
          decisions = linkFramePairFallback({
            boxesA: prev.boxes,
            boxesB: boxes,
            frameWidthA: prev.width,
            frameHeightA: prev.height,
            frameWidthB: state.width,
            frameHeightB: state.height,
          });
          fallback = true;
          linkerFallbacks += 1;
          error(
            `Linker failed on #${prev.index}→#${idx1}, used fallback: ` +
              (e instanceof Error ? e.message : String(e)),
          );
        }
      }

      // Mint ids + stamp back onto the cached boxes so any downstream
      // read (navigator, export) sees them.
      const trackIds: string[] = new Array(boxes.length);
      const linkNarrative: Array<{
        prev_index: number | null;
        prev_b_index: number | null;
        reason: string | null;
        fallback?: boolean;
      }> = new Array(boxes.length);
      for (let bi = 0; bi < boxes.length; bi++) {
        const d = decisions[bi] ?? {
          b_index: bi,
          a_index: null,
          reason: null,
        };
        let id: string;
        let prevBIndex: number | null = null;
        if (prev && d.a_index != null && d.a_index < prev.trackIds.length) {
          id = prev.trackIds[d.a_index];
          prevBIndex = d.a_index;
        } else {
          id = mintTrack();
        }
        trackIds[bi] = id;
        boxes[bi].track_id = id;
        linkNarrative[bi] = {
          prev_index: prev?.index ?? null,
          prev_b_index: prevBIndex,
          reason: d.reason,
          ...(fallback ? { fallback: true } : {}),
        };
      }

      emit({
        type: "link",
        index: idx1,
        track_ids: trackIds,
        links: linkNarrative,
      });

      prev = {
        index: idx1,
        boxes,
        trackIds,
        jpeg: state.blob,
        width: state.width,
        height: state.height,
      };
    }

    const linkerElapsed = Date.now() - linkerT0;
    const elapsed = curatorElapsed + linkerElapsed;

    // ---- Cost summary ------------------------------------------------
    // Textract bill = one page per OCR frame returned by the Python
    // backend. Gemini curator bill = aggregated input/output tokens
    // across every per-frame curator call. Linker bill = one call per
    // adjacent scanned pair with non-empty A and B.
    const ocrPages = ocr.frames.length;
    const ocrBill = textractCost(ocrPages);
    const curatorBill = geminiCost(agenticModelId(), curatorUsage);
    const linkerBill = geminiCost(agenticLinkerModelId(), linkerUsage);
    const totalUSD =
      ocrBill.totalUSD + curatorBill.totalUSD + linkerBill.totalUSD;

    alog("detect route cost", {
      elapsed_ms: elapsed,
      curator_elapsed_ms: curatorElapsed,
      linker_elapsed_ms: linkerElapsed,
      matched_frames: matchedFrames,
      total_boxes: totalBoxes,
      // --- OCR ---
      ocr_provider: "aws_textract_detect_document_text",
      ocr_pages: ocrPages,
      ocr_total_usd: ocrBill.totalUSD,
      ocr_pricing: `${formatUSD(ocrBill.perPageUSD)} / page`,
      // --- Gemini curator ---
      curator_model: curatorBill.model,
      curator_calls: curatorBill.callCount,
      curator_input_tokens: curatorBill.inputTokens,
      curator_output_tokens: curatorBill.outputTokens,
      curator_reasoning_tokens: curatorBill.reasoningTokens,
      curator_cached_input_tokens: curatorBill.cachedInputTokens,
      curator_input_usd: curatorBill.inputUSD,
      curator_output_usd: curatorBill.outputUSD,
      curator_total_usd: curatorBill.totalUSD,
      curator_tier: curatorBill.tier,
      // --- Gemini linker (phase-1.5) ---
      linker_model: linkerBill.model,
      linker_calls: linkerCalls,
      linker_fallbacks: linkerFallbacks,
      linker_input_tokens: linkerBill.inputTokens,
      linker_output_tokens: linkerBill.outputTokens,
      linker_reasoning_tokens: linkerBill.reasoningTokens,
      linker_cached_input_tokens: linkerBill.cachedInputTokens,
      linker_input_usd: linkerBill.inputUSD,
      linker_output_usd: linkerBill.outputUSD,
      linker_total_usd: linkerBill.totalUSD,
      linker_tier: linkerBill.tier,
      // --- combined ---
      total_usd: totalUSD,
    });
    // Human-readable one-liner for scanning the log:
    alog(
      `detect route done — OCR ${ocrPages} pages @ ${formatUSD(ocrBill.totalUSD)} · ` +
        `curator ${curatorBill.callCount} calls (${formatTokens(curatorBill.inputTokens)} in / ${formatTokens(curatorBill.outputTokens)} out) ` +
        `@ ${formatUSD(curatorBill.totalUSD)} · ` +
        `linker ${linkerCalls} calls` +
        (linkerFallbacks > 0 ? ` (+${linkerFallbacks} fallback)` : "") +
        ` (${formatTokens(linkerBill.inputTokens)} in / ${formatTokens(linkerBill.outputTokens)} out) ` +
        `@ ${formatUSD(linkerBill.totalUSD)} · ` +
        `total ${formatUSD(totalUSD)}`,
    );

    // Emit the final, authoritative cost summary so the client can
    // replace the running estimate with ground-truth numbers.
    emit({
      type: "cost_final",
      phase: "detect",
      total_usd: totalUSD,
      breakdown: {
        ocr_pages: ocrPages,
        ocr_usd: ocrBill.totalUSD,
        curator_calls: curatorBill.callCount,
        curator_input_tokens: curatorBill.inputTokens,
        curator_output_tokens: curatorBill.outputTokens,
        curator_reasoning_tokens: curatorBill.reasoningTokens,
        curator_cached_input_tokens: curatorBill.cachedInputTokens,
        curator_usd: curatorBill.totalUSD,
        linker_calls: linkerCalls,
        linker_fallbacks: linkerFallbacks,
        linker_input_tokens: linkerBill.inputTokens,
        linker_output_tokens: linkerBill.outputTokens,
        linker_usd: linkerBill.totalUSD,
      },
      elapsed_ms: elapsed,
      run_log_path: runLog.path,
    });

    runLog.write({
      kind: "run_end",
      elapsed_ms: elapsed,
      curator_elapsed_ms: curatorElapsed,
      linker_elapsed_ms: linkerElapsed,
      matched_frames: matchedFrames,
      total_boxes: totalBoxes,
      ocr_pages: ocrPages,
      ocr_usd: ocrBill.totalUSD,
      curator_calls: curatorBill.callCount,
      curator_usd: curatorBill.totalUSD,
      linker_calls: linkerCalls,
      linker_usd: linkerBill.totalUSD,
      total_usd: totalUSD,
    });
    await runLog.close();

    emit({
      type: "done",
      matched_frames: matchedFrames,
      total_boxes: totalBoxes,
    });
  });
}

function json(detail: string, status: number): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

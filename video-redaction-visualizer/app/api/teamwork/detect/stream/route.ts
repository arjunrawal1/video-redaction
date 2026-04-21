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
  runGapFiller,
  type KeptFrameForGapFill,
} from "@/lib/server/gap-filler";
import {
  createEntry,
  putFrame,
  type FrameState,
} from "@/lib/server/gemini-cache";
import { linkFramePairFallback } from "@/lib/server/linker-fallback";
import { fetchOcrDetect } from "@/lib/server/ocr-client";
import {
  agenticCuratorConcurrency,
  agenticLinkerConcurrency,
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

  const { file, query, fps, dedupThreshold, maxGap } = parsed;
  let { frameFrom, frameTo } = parsed;
  const qNorm = normalizeQuery(query);

  let framesRes;
  try {
    framesRes = await fetchDeduplicatedFramesServer({
      file,
      fps: fps ?? undefined,
      dedupThreshold,
      maxGap,
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
      maxGap,
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
    maxGap,
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
    max_gap: maxGap,
    source_fps: framesRes.sourceFps,
    kept_source_indices: framesRes.keptSourceIndices,
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
    // Three-phase design (see the analysis in the agent-transcript
    // chat "parallel linker design"):
    //
    //   Phase A  — plan pass. Walk `indices` once and classify every
    //              frame into one of {gap, empty, start, link}. Collect
    //              the pair jobs we'll need to ask Gemini about.
    //   Phase B  — parallel Gemini linker calls. Each pair is
    //              data-independent — the model receives only
    //              boxesA/boxesB and the two JPEGs, and returns
    //              index-to-index mappings. Bound concurrency by
    //              `AGENTIC_LINKER_CONCURRENCY` (default 16).
    //   Phase C  — serial stitch. Walk the plan in frame order, mint
    //              track ids (inheriting from the previous populated
    //              frame's ids whenever a decision says so), stamp them
    //              onto the boxes, and emit the `link` events in frame
    //              order. This phase is O(total boxes) integer
    //              arithmetic and runs in microseconds.
    //
    // Accuracy preservation: each Gemini call sees the exact same
    // inputs it would see in the serial version, so no decision
    // changes. Chain-break semantics (unscanned gap → no event; empty
    // frame → empty-link event + chain reset) fire at the same points.
    // Fallback is per-pair: a throw demotes just that pair to
    // linkFramePairFallback, same as before.
    const linkerT0 = Date.now();
    const linkerUsage = emptyUsage();
    let linkerCalls = 0;
    let linkerFallbacks = 0;
    let nextTrackId = 0;
    const mintTrack = (): string => `t${nextTrackId++}`;

    // ---- Phase A: plan ------------------------------------------------
    type PairJob = {
      bIdx: number;
      jpegA: Uint8Array;
      jpegB: Uint8Array;
      frameIndexA: number;
      frameIndexB: number;
      frameWidthA: number;
      frameHeightA: number;
      frameWidthB: number;
      frameHeightB: number;
      boxesA: ServerBox[];
      boxesB: ServerBox[];
    };
    type PlanEntry =
      | { kind: "gap"; idx: number }
      | { kind: "empty"; idx: number }
      | { kind: "start"; idx: number; boxes: ServerBox[] }
      | { kind: "link"; idx: number; boxes: ServerBox[]; pair: PairJob };

    const plan: PlanEntry[] = [];
    const pairJobs: PairJob[] = [];
    {
      type PlanPrev = {
        index: number;
        boxes: ServerBox[];
        jpeg: Uint8Array;
        width: number;
        height: number;
      };
      let planPrev: PlanPrev | null = null;
      for (const idx1 of indices) {
        const state = entry.perFrame[idx1];
        if (!state) {
          // Unscanned frame: break the chain silently. No `link` event,
          // matching the serial baseline.
          plan.push({ kind: "gap", idx: idx1 });
          planPrev = null;
          continue;
        }
        const boxes = state.matched;
        if (boxes.length === 0) {
          // Zero-box frame: break the chain and emit an empty link
          // event so the client still knows the frame was processed.
          plan.push({ kind: "empty", idx: idx1 });
          planPrev = null;
          continue;
        }
        if (!planPrev) {
          plan.push({ kind: "start", idx: idx1, boxes });
        } else {
          const job: PairJob = {
            bIdx: idx1,
            jpegA: planPrev.jpeg,
            jpegB: state.blob,
            frameIndexA: planPrev.index,
            frameIndexB: idx1,
            frameWidthA: planPrev.width,
            frameHeightA: planPrev.height,
            frameWidthB: state.width,
            frameHeightB: state.height,
            boxesA: planPrev.boxes,
            boxesB: boxes,
          };
          pairJobs.push(job);
          plan.push({ kind: "link", idx: idx1, boxes, pair: job });
        }
        planPrev = {
          index: idx1,
          boxes,
          jpeg: state.blob,
          width: state.width,
          height: state.height,
        };
      }
    }

    // ---- Phase B: parallel Gemini linker calls -----------------------
    // Each worker pulls the next unclaimed pair job, runs Gemini, and
    // stashes its decisions in `decisionsByBIdx` keyed by frame-B index.
    // A thrown error for a given pair writes the deterministic fallback
    // for just that pair; siblings keep running.
    //
    // Cost updates emit as each call completes — the `frame_pair` field
    // will arrive in completion order rather than strict frame order,
    // but `running_usd` / breakdown counters are monotonically growing
    // so the UI state stays consistent.
    const decisionsByBIdx = new Map<number, LinkDecision[]>();
    const fallbackByBIdx = new Set<number>();
    const linkerConcurrency = Math.max(1, agenticLinkerConcurrency());
    let pairCursor = 0;
    const linkerWorker = async (): Promise<void> => {
      while (true) {
        const my = pairCursor++;
        if (my >= pairJobs.length) return;
        const job = pairJobs[my];
        try {
          const r = await linkFramePair({
            jpegA: job.jpegA,
            jpegB: job.jpegB,
            frameIndexA: job.frameIndexA,
            frameIndexB: job.frameIndexB,
            frameWidthA: job.frameWidthA,
            frameHeightA: job.frameHeightA,
            frameWidthB: job.frameWidthB,
            frameHeightB: job.frameHeightB,
            boxesA: job.boxesA,
            boxesB: job.boxesB,
            runLog,
          });
          decisionsByBIdx.set(job.bIdx, r.links);
          addUsage(linkerUsage, r.usage);
          if (r.usage != null) {
            linkerCalls += 1;
            const callBill = geminiCost(agenticLinkerModelId(), {
              inputTokens: r.usage.inputTokens ?? 0,
              outputTokens: r.usage.outputTokens ?? 0,
              reasoningTokens:
                r.usage.outputTokenDetails?.reasoningTokens ?? 0,
              cachedInputTokens:
                r.usage.inputTokenDetails?.cacheReadTokens ?? 0,
              callCount: 1,
            });
            runningUSD += callBill.totalUSD;
            const cBill = geminiCost(agenticModelId(), curatorUsage);
            const lBill = geminiCost(agenticLinkerModelId(), linkerUsage);
            emit({
              type: "cost_update",
              phase: "linker",
              frame_pair: [job.frameIndexA, job.frameIndexB],
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
          aerr(
            `linker pair #${job.frameIndexA}→#${job.frameIndexB} fell back`,
            e,
          );
          decisionsByBIdx.set(
            job.bIdx,
            linkFramePairFallback({
              boxesA: job.boxesA,
              boxesB: job.boxesB,
              frameWidthA: job.frameWidthA,
              frameHeightA: job.frameHeightA,
              frameWidthB: job.frameWidthB,
              frameHeightB: job.frameHeightB,
            }),
          );
          fallbackByBIdx.add(job.bIdx);
          linkerFallbacks += 1;
          error(
            `Linker failed on #${job.frameIndexA}→#${job.frameIndexB}, used fallback: ` +
              (e instanceof Error ? e.message : String(e)),
          );
        }
      }
    };
    await Promise.all(
      Array.from(
        { length: Math.min(linkerConcurrency, pairJobs.length) },
        () => linkerWorker(),
      ),
    );

    // ---- Phase C: serial stitch + emit --------------------------------
    // Walk the plan in frame order, thread track ids through the chain,
    // stamp them onto the cached ServerBox objects in place, and emit
    // `link` events in the same order a fully sequential linker would.
    let prevTrackIds: string[] | null = null;
    let prevIndex: number | null = null;
    for (const pe of plan) {
      if (pe.kind === "gap") {
        // Silent break — no event emitted for unscanned gaps.
        prevTrackIds = null;
        prevIndex = null;
        continue;
      }
      if (pe.kind === "empty") {
        emit({
          type: "link",
          index: pe.idx,
          track_ids: [],
          links: [],
        });
        prevTrackIds = null;
        prevIndex = null;
        continue;
      }

      const boxes = pe.boxes;
      let decisions: LinkDecision[];
      let fallback = false;
      if (pe.kind === "start") {
        decisions = boxes.map((_, bi) => ({
          b_index: bi,
          a_index: null,
          reason: "chain start",
        }));
      } else {
        // Phase B guarantees decisionsByBIdx has an entry for every
        // `link` plan entry — either the Gemini result or the fallback.
        // The nullish coalescing below is defensive belt-and-braces.
        decisions =
          decisionsByBIdx.get(pe.idx) ??
          linkFramePairFallback({
            boxesA: pe.pair.boxesA,
            boxesB: pe.pair.boxesB,
            frameWidthA: pe.pair.frameWidthA,
            frameHeightA: pe.pair.frameHeightA,
            frameWidthB: pe.pair.frameWidthB,
            frameHeightB: pe.pair.frameHeightB,
          });
        fallback = fallbackByBIdx.has(pe.idx);
      }

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
        if (
          prevTrackIds &&
          d.a_index != null &&
          d.a_index < prevTrackIds.length
        ) {
          id = prevTrackIds[d.a_index];
          prevBIndex = d.a_index;
        } else {
          id = mintTrack();
        }
        trackIds[bi] = id;
        boxes[bi].track_id = id;
        linkNarrative[bi] = {
          prev_index: prevIndex,
          prev_b_index: prevBIndex,
          reason: d.reason,
          ...(fallback ? { fallback: true } : {}),
        };
      }

      emit({
        type: "link",
        index: pe.idx,
        track_ids: trackIds,
        links: linkNarrative,
      });

      prevTrackIds = trackIds;
      prevIndex = pe.idx;
    }

    const linkerElapsed = Date.now() - linkerT0;

    // ---- Phase 1.75: gap-filler --------------------------------------
    // For every adjacent kept-frame pair whose same-track motion or
    // resize exceeds the smoothness-invariant thresholds, recursively
    // bisect the gap in source-index space, fetch the raw frame at the
    // midpoint, curate + link it, and recurse on the two halves. This
    // gives the client smooth-tween material without asking it to
    // guess across large drifts. See lib/server/gap-filler.ts for the
    // full algorithm + threshold rationale.
    const gapT0 = Date.now();
    let gapFillerElapsed = 0;
    let gapCuratorUsd = 0;
    let gapLinkerUsd = 0;
    let gapInsertedCount = 0;
    try {
      // Gather each kept frame's current post-linker state. Boxes were
      // mutated in place (track_id stamped onto the ServerBox objects
      // in `entry.perFrame[idx].matched`), so the cache has everything
      // the gap-filler needs.
      const kept: KeptFrameForGapFill[] = [];
      for (const idx1 of indices) {
        const s = entry.perFrame[idx1];
        if (!s) continue;
        // Map 1-based kept index → 0-based source index via the frame
        // set's `keptSourceIndices`. When the backend didn't return
        // that array (older Python without the field), we skip the
        // gap-filler entirely because source-index bisection wouldn't
        // work.
        const srcIdx = framesRes.keptSourceIndices?.[idx1 - 1];
        if (typeof srcIdx !== "number") continue;
        kept.push({
          index: idx1,
          sourceIndex: srcIdx,
          width: s.width,
          height: s.height,
          jpeg: s.blob,
          boxes: s.matched,
          trackIds: s.matched.map((b) => b.track_id ?? ""),
        });
      }

      if (kept.length >= 2) {
        const runningRef = { value: runningUSD };
        const gapResult = await runGapFiller({
          kept,
          videoFile: file,
          fps: fps ?? null,
          query: qNorm,
          mintTrackId: mintTrack,
          emit,
          runningUSDRef: runningRef,
          runLog,
        });
        runningUSD = runningRef.value;
        gapInsertedCount = gapResult.inserts.length;
        gapCuratorUsd = geminiCost(
          agenticModelId(),
          gapResult.curatorUsage,
        ).totalUSD;
        gapLinkerUsd = geminiCost(
          agenticLinkerModelId(),
          gapResult.linkerUsage,
        ).totalUSD;
        // Fold gap-filler tokens into the main aggregates so the
        // cost_final summary counts them. They used the same two
        // models (curator + linker), so the same aggregates are the
        // right destination.
        curatorUsage.inputTokens += gapResult.curatorUsage.inputTokens;
        curatorUsage.outputTokens += gapResult.curatorUsage.outputTokens;
        curatorUsage.reasoningTokens +=
          gapResult.curatorUsage.reasoningTokens;
        curatorUsage.cachedInputTokens +=
          gapResult.curatorUsage.cachedInputTokens;
        curatorUsage.callCount += gapResult.curatorUsage.callCount;
        linkerUsage.inputTokens += gapResult.linkerUsage.inputTokens;
        linkerUsage.outputTokens += gapResult.linkerUsage.outputTokens;
        linkerUsage.reasoningTokens += gapResult.linkerUsage.reasoningTokens;
        linkerUsage.cachedInputTokens +=
          gapResult.linkerUsage.cachedInputTokens;
        linkerUsage.callCount += gapResult.linkerUsage.callCount;
        alog("detect route gap_filler done", {
          inserted_frames: gapInsertedCount,
          top_level_triggered: gapResult.triggeredTopLevelGaps,
          total_probes: gapResult.totalProbes,
          gap_curator_usd: gapCuratorUsd,
          gap_linker_usd: gapLinkerUsd,
        });
      } else {
        alog("detect route gap_filler skipped", {
          reason: "fewer than 2 kept frames",
          kept_frames: kept.length,
        });
      }
    } catch (e) {
      aerr("gap_filler threw at top level", e);
      runLog.write({
        kind: "gap_filler_error",
        error: e instanceof Error ? e.message : String(e),
      });
    }
    gapFillerElapsed = Date.now() - gapT0;

    const elapsed = curatorElapsed + linkerElapsed + gapFillerElapsed;

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
      gap_filler_elapsed_ms: gapFillerElapsed,
      gap_inserted_frames: gapInsertedCount,
      gap_curator_usd: gapCuratorUsd,
      gap_linker_usd: gapLinkerUsd,
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
      gap_filler_elapsed_ms: gapFillerElapsed,
      gap_inserted_frames: gapInsertedCount,
      gap_curator_usd: gapCuratorUsd,
      gap_linker_usd: gapLinkerUsd,
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

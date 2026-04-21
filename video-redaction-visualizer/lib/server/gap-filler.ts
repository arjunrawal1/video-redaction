// Post-detection gap-filler.
//
// Runs after the detect + linker pipeline has produced a per-kept-frame
// box list with `track_id`s linking the same real-world redaction
// across consecutive kept frames. For every adjacent kept-frame pair
// whose same-track motion or resize exceeds the smoothness-invariant
// thresholds, we recursively bisect the gap in source-index space:
// fetch the midpoint raw frame from the backend, OCR + curate it,
// re-link it into both neighbors, and recurse on the two halves until
// every sub-gap is below threshold (or we hit the depth cap).
//
// Design goals — in order:
//
//   1. Accuracy. Every inserted frame gets full OCR + curator + linker
//      treatment, exactly like a kept frame. No interpolation shortcuts.
//   2. Smoothness. The trigger thresholds (below) are tight enough that
//      the tween between adjacent kept frames never has to cover more
//      than ~8 px of motion or ~10 % of resize — the scale where a
//      linear tween starts visibly under-covering moving text.
//   3. Parallelism. Independent gaps are processed concurrently. Each
//      recursion level's two halves are processed concurrently too.
//      Sequential dependencies are only within the local linker chain.
//
// What this module does NOT do:
//   - It does not mutate the kept-frame array. It emits `inserted_frame`
//     events carrying the new frame's JPEG + boxes + track ids; the
//     caller decides how to splice them into downstream state.
//   - It does not handle the navigate route. Navigate runs after detect
//     and can add/remove boxes; threading the gap-filler through it is
//     future work.

import { curateFrame } from "./agentic-curator";
import { aerr, alog } from "./agentic-log";
import { linkFramePair } from "./agentic-linker";
import { addUsage, geminiCost, type AggregateUsage } from "./cost";
import { fetchFramesBySourceIndexServer } from "./frames";
import { fetchOcrDetectSingle } from "./ocr-client";
import {
  agenticLinkerModelId,
  agenticModelId,
  type ServerBox,
} from "./openrouter";
import type { RunLog } from "./run-log";

// ---- Thresholds ---------------------------------------------------------
//
// Tuned from the ~380-pair resize/motion distribution observed on the
// test-word video (see logs/2026-04-21T15-50-45-965Z-detect-test-word-
// gf8u.jsonl). Current values target the ~top 15 % of pairs — above the
// p65 of same-track motion and above the tail of true resizes, without
// firing on pure OCR jitter (median |Δw| = 0, mean = 1.8 px). Expose as
// env vars so operators can tune without a code change.

function envNum(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

const GAP_FILLER_MOVE_PX = envNum("GAP_FILLER_MOVE_PX", 8);
const GAP_FILLER_REL_W = envNum("GAP_FILLER_REL_W", 0.1);
const GAP_FILLER_REL_H = envNum("GAP_FILLER_REL_H", 0.1);
const GAP_FILLER_REL_AREA = envNum("GAP_FILLER_REL_AREA", 0.15);
// Any unmatched new-track box whose area exceeds this fraction of the
// full frame area is considered a "real" new appearance worth bisecting
// for, rather than OCR flicker on a detail.
const GAP_FILLER_NEW_TRACK_AREA_FRAC = envNum(
  "GAP_FILLER_NEW_TRACK_AREA_FRAC",
  0.01,
);
// Recursion cap. At depth 5 we've already inserted up to 2^5-1 = 31
// frames inside the original gap — the OCR/curator/linker on each of
// those is a real call, so this caps the worst-case fan-out per gap.
// For the default `max_gap=1` the gap-filler is mostly a no-op because
// kept frames are already ~66 ms apart; depth 5 is a safety net for
// odd videos where `max_gap` was disabled or the perceptual hash got
// confused by a slow pan.
const GAP_FILLER_MAX_DEPTH = Math.max(
  0,
  Math.floor(envNum("GAP_FILLER_MAX_DEPTH", 5)),
);
// Global cap on total inserts per run, as a fraction of the original
// kept-frame count. Protects against pathological videos (rapid
// continuous motion on every frame) where every gap would otherwise
// trigger. When the budget is hit we stop bisecting and log a
// ``gap_terminate`` with reason=``budget``.
const GAP_FILLER_MAX_INSERT_FRAC = envNum(
  "GAP_FILLER_MAX_INSERT_FRAC",
  1.0, // 100% — default off because cost isn't the constraint here.
);

// ---- Types --------------------------------------------------------------

export type KeptFrameForGapFill = {
  /** 1-based kept-frame index as emitted to the client. */
  index: number;
  /** 0-based index into the original ffmpeg-emitted sequence. */
  sourceIndex: number;
  width: number;
  height: number;
  jpeg: Uint8Array;
  /** Curator boxes after the linker pass. */
  boxes: ServerBox[];
  /** Track ids aligned with `boxes`. */
  trackIds: string[];
};

export type GapFillEmitter = (ev: GapFillEvent) => void;

export type InsertedFrame = {
  /** Virtual kept-frame index assigned to this insert (parent.index < virtualIndex < next.index). */
  virtualIndex: number;
  sourceIndex: number;
  width: number;
  height: number;
  jpeg: Uint8Array;
  boxes: ServerBox[];
  trackIds: string[];
  /** The two kept frames this insert sits between, by kept-frame index. */
  betweenKeptIndices: [number, number];
  /** Recursion depth at which this frame was created (root inserts are depth 0). */
  depth: number;
  /**
   * Why the gap that produced this insert triggered. Useful in logs for
   * post-hoc threshold tuning.
   */
  triggerReason: string;
};

export type GapFillEvent =
  | {
      type: "inserted_frame";
      between: [number, number];
      depth: number;
      trigger_reason: string;
      source_index: number;
      frame: {
        width: number;
        height: number;
        jpeg_base64: string;
        boxes: Array<{
          x: number;
          y: number;
          w: number;
          h: number;
          text: string;
          score: number;
          origin?: string;
          track_id?: string;
        }>;
        track_ids: string[];
      };
    }
  | {
      type: "cost_update";
      phase: "gap_filler";
      call_usd: number;
      running_usd: number;
      breakdown: Record<string, number | string>;
    };

export type GapFillResult = {
  inserts: InsertedFrame[];
  /** Number of gaps that triggered at the top level. */
  triggeredTopLevelGaps: number;
  /** Total gap_probe events written (includes non-triggering ones). */
  totalProbes: number;
  curatorUsage: AggregateUsage;
  linkerUsage: AggregateUsage;
};

// ---- Geometry helpers ---------------------------------------------------

type TrackChange = {
  /** Largest center-to-center displacement (px) across linked tracks. */
  moveMaxPx: number;
  /** Largest |Δw| / w_prev. */
  relWMax: number;
  /** Largest |Δh| / h_prev. */
  relHMax: number;
  /** Largest |ΔArea| / Area_prev. */
  relAreaMax: number;
  /** Fraction of frame area taken by unmatched new-track boxes in B. */
  newTrackAreaFracMax: number;
  /** Number of tracks linked between A and B. */
  linkedTracks: number;
  /** A->B pair count, including unmatched. */
  totalPairs: number;
};

function centerOf(b: ServerBox): [number, number] {
  return [b.x + b.w / 2, b.y + b.h / 2];
}

function measureTrackChange(
  a: KeptFrameForGapFill,
  b: KeptFrameForGapFill,
): TrackChange {
  const byTrackA = new Map<string, ServerBox>();
  for (let i = 0; i < a.boxes.length; i++) {
    const id = a.trackIds[i];
    if (id) byTrackA.set(id, a.boxes[i]);
  }
  let moveMaxPx = 0;
  let relWMax = 0;
  let relHMax = 0;
  let relAreaMax = 0;
  let newTrackAreaFracMax = 0;
  let linkedTracks = 0;
  const frameAreaB = Math.max(1, b.width * b.height);

  for (let j = 0; j < b.boxes.length; j++) {
    const bBox = b.boxes[j];
    const id = b.trackIds[j];
    const matchA = id ? byTrackA.get(id) : undefined;
    if (!matchA) {
      // New track in B relative to A. Only a "big" appearance justifies
      // bisection — tiny detail boxes (OCR picking up a single pixel of
      // drift) shouldn't force an insert.
      const areaFrac = (bBox.w * bBox.h) / frameAreaB;
      if (areaFrac > newTrackAreaFracMax) newTrackAreaFracMax = areaFrac;
      continue;
    }
    linkedTracks += 1;
    const [cax, cay] = centerOf(matchA);
    const [cbx, cby] = centerOf(bBox);
    const move = Math.hypot(cax - cbx, cay - cby);
    if (move > moveMaxPx) moveMaxPx = move;
    const wA = Math.max(1, matchA.w);
    const hA = Math.max(1, matchA.h);
    const aArea = Math.max(1, matchA.w * matchA.h);
    const relW = Math.abs(bBox.w - matchA.w) / wA;
    const relH = Math.abs(bBox.h - matchA.h) / hA;
    const relArea = Math.abs(bBox.w * bBox.h - aArea) / aArea;
    if (relW > relWMax) relWMax = relW;
    if (relH > relHMax) relHMax = relH;
    if (relArea > relAreaMax) relAreaMax = relArea;
  }

  return {
    moveMaxPx,
    relWMax,
    relHMax,
    relAreaMax,
    newTrackAreaFracMax,
    linkedTracks,
    totalPairs: b.boxes.length,
  };
}

/**
 * Returns (a human-readable trigger reason) if the pair should be
 * bisected, otherwise null. Keeping the reason string (rather than a
 * boolean) so log analysis can attribute each insert to a specific
 * threshold that fired.
 */
function triggerReason(
  tc: TrackChange,
  srcGap: number,
): string | null {
  // srcGap == 1 means there is no raw frame to insert between a and b.
  if (srcGap <= 1) return null;
  if (tc.moveMaxPx > GAP_FILLER_MOVE_PX) {
    return `move ${tc.moveMaxPx.toFixed(1)}px > ${GAP_FILLER_MOVE_PX}`;
  }
  if (tc.relWMax > GAP_FILLER_REL_W) {
    return `relW ${(tc.relWMax * 100).toFixed(1)}% > ${Math.round(
      GAP_FILLER_REL_W * 100,
    )}%`;
  }
  if (tc.relHMax > GAP_FILLER_REL_H) {
    return `relH ${(tc.relHMax * 100).toFixed(1)}% > ${Math.round(
      GAP_FILLER_REL_H * 100,
    )}%`;
  }
  if (tc.relAreaMax > GAP_FILLER_REL_AREA) {
    return `relA ${(tc.relAreaMax * 100).toFixed(1)}% > ${Math.round(
      GAP_FILLER_REL_AREA * 100,
    )}%`;
  }
  if (tc.newTrackAreaFracMax > GAP_FILLER_NEW_TRACK_AREA_FRAC) {
    return `new-track ${(tc.newTrackAreaFracMax * 100).toFixed(2)}% of frame > ${Math.round(
      GAP_FILLER_NEW_TRACK_AREA_FRAC * 100,
    )}%`;
  }
  return null;
}

// ---- Base64 helper (same contract as route stream emits) ----------------

function toBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(bytes).toString("base64");
  }
  // Fallback for non-Node runtimes — we only ever run this module in
  // Next.js's Node runtime today, so this is belt-and-braces.
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, no-undef
  return (globalThis as any).btoa(binary);
}

// ---- Main entry ---------------------------------------------------------

export async function runGapFiller(opts: {
  kept: KeptFrameForGapFill[];
  videoFile: File;
  fps: number | null;
  query: string;
  /** Callback to mint the next fresh track id. Shared with the parent
   *  linker chain so no ids collide. */
  mintTrackId: () => string;
  /**
   * Callback to emit NDJSON events to the client stream. The gap-filler
   * emits ``inserted_frame`` events (one per successful insert) and
   * optionally ``cost_update`` (one per curator/linker call).
   */
  emit: GapFillEmitter;
  /** Running cumulative USD so ``cost_update`` events stay monotonic. */
  runningUSDRef: { value: number };
  runLog?: RunLog | null;
}): Promise<GapFillResult> {
  const {
    kept,
    videoFile,
    fps,
    query,
    mintTrackId,
    emit,
    runningUSDRef,
    runLog,
  } = opts;

  const inserts: InsertedFrame[] = [];
  let triggeredTopLevelGaps = 0;
  let totalProbes = 0;
  const curatorUsage: AggregateUsage = {
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    cachedInputTokens: 0,
    callCount: 0,
  };
  const linkerUsage: AggregateUsage = {
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    cachedInputTokens: 0,
    callCount: 0,
  };

  const maxInserts = Math.floor(kept.length * GAP_FILLER_MAX_INSERT_FRAC);
  const budgetExhausted = () =>
    GAP_FILLER_MAX_INSERT_FRAC < 1 && inserts.length >= maxInserts;

  // Virtual kept-frame indices for inserts. To keep downstream ordering
  // consistent, we mint them as `parent_index + offset / totalOffsets`,
  // but emitted as a plain number (we scale up). Simplest: allocate
  // fresh integers past the original kept count. The client splices by
  // `between` metadata rather than by index ordering.
  let nextVirtualIndex = kept.length + 1;
  const mintVirtualIndex = (): number => nextVirtualIndex++;

  /**
   * Emit a cost_update event for the single Gemini call whose usage was
   * just added to the curator/linker aggregate. Keeps the client's
   * running total monotonic.
   */
  const noteCost = (
    phase: "curator" | "linker",
    callUsd: number,
  ): void => {
    runningUSDRef.value += callUsd;
    const cBill = geminiCost(agenticModelId(), curatorUsage);
    const lBill = geminiCost(agenticLinkerModelId(), linkerUsage);
    emit({
      type: "cost_update",
      phase: "gap_filler",
      call_usd: callUsd,
      running_usd: runningUSDRef.value,
      breakdown: {
        gap_phase: phase,
        gap_curator_calls: cBill.callCount,
        gap_curator_usd: cBill.totalUSD,
        gap_linker_calls: lBill.callCount,
        gap_linker_usd: lBill.totalUSD,
        gap_inserts: inserts.length,
      },
    });
  };

  /**
   * Curate + dual-link an inserted frame that sits between ``a`` and
   * ``b`` at source index ``midSrc``. Returns the fully-populated
   * ``InsertedFrame`` or null if we couldn't fetch the underlying
   * image.
   */
  const detectInserted = async (
    a: KeptFrameForGapFill,
    b: KeptFrameForGapFill,
    midSrc: number,
    depth: number,
    reason: string,
  ): Promise<KeptFrameForGapFill | null> => {
    // 1) Fetch raw JPEG for source_index = midSrc.
    const framesRes = await fetchFramesBySourceIndexServer({
      file: videoFile,
      sourceIndices: [midSrc],
      fps,
    });
    const fetched = framesRes.frames[0];
    if (!fetched) {
      alog(`gap_filler: no raw frame at source_index=${midSrc}`, {
        between: [a.index, b.index],
      });
      return null;
    }

    // 2) OCR it. Single-frame Textract call, same data shape as the
    //    stream endpoint's entries so the curator prompt is identical
    //    to the kept-frame path.
    const virtualIndex = mintVirtualIndex();
    let ocr: { boxes: ServerBox[]; raw: unknown };
    try {
      const r = await fetchOcrDetectSingle(fetched.blob, query);
      ocr = { boxes: r.boxes, raw: r.raw };
    } catch (e) {
      aerr(
        `gap_filler: OCR failed for inserted frame src=${midSrc}; proceeding without OCR seeds`,
        e,
      );
      ocr = { boxes: [], raw: null };
    }

    // 3) Curator. Runs with full OCR candidates + raw dump, same
    //    system prompt + behaviour as a kept-frame curator call.
    const curated = await curateFrame({
      jpeg: fetched.blob,
      query,
      frameIndex: virtualIndex,
      frameWidth: fetched.width,
      frameHeight: fetched.height,
      ocrBoxes: ocr.boxes,
      ocrRaw: ocr.raw,
      runLog,
    });
    addUsage(curatorUsage, curated.usage);
    if (curated.usage) {
      const callBill = geminiCost(agenticModelId(), {
        inputTokens: curated.usage.inputTokens ?? 0,
        outputTokens: curated.usage.outputTokens ?? 0,
        reasoningTokens:
          curated.usage.outputTokenDetails?.reasoningTokens ?? 0,
        cachedInputTokens:
          curated.usage.inputTokenDetails?.cacheReadTokens ?? 0,
        callCount: 1,
      });
      noteCost("curator", callBill.totalUSD);
    }
    const midBoxes: ServerBox[] = [
      ...curated.kept.map((x) => ({ ...x, origin: undefined })),
      ...curated.added,
    ];

    // Assign track ids for `mid`: inherit from A where the linker matches,
    // mint fresh otherwise. We do NOT also run (mid -> b) linker: the
    // original a→b pass already assigned b's ids, and introducing a
    // second opinion on b would just create duplicate-id chaos. The
    // natural outcome is:
    //   - if mid has the track through it (same id as a), tween chain
    //     a→mid→b is continuous when b also shares that id.
    //   - if mid doesn't have the track, the tween gracefully breaks
    //     at mid (safer than a long tween across a→b).
    const midTrackIds: string[] = new Array(midBoxes.length);
    if (midBoxes.length === 0) {
      // Fast path — skip the linker entirely.
    } else if (a.boxes.length === 0) {
      for (let i = 0; i < midBoxes.length; i++) {
        midTrackIds[i] = mintTrackId();
      }
    } else {
      try {
        const r = await linkFramePair({
          jpegA: a.jpeg,
          jpegB: fetched.blob,
          frameIndexA: a.index,
          frameIndexB: virtualIndex,
          frameWidthA: a.width,
          frameHeightA: a.height,
          frameWidthB: fetched.width,
          frameHeightB: fetched.height,
          boxesA: a.boxes,
          boxesB: midBoxes,
          runLog,
        });
        addUsage(linkerUsage, r.usage);
        if (r.usage != null) {
          const callBill = geminiCost(agenticLinkerModelId(), {
            inputTokens: r.usage.inputTokens ?? 0,
            outputTokens: r.usage.outputTokens ?? 0,
            reasoningTokens:
              r.usage.outputTokenDetails?.reasoningTokens ?? 0,
            cachedInputTokens:
              r.usage.inputTokenDetails?.cacheReadTokens ?? 0,
            callCount: 1,
          });
          noteCost("linker", callBill.totalUSD);
        }
        for (let i = 0; i < midBoxes.length; i++) {
          const d = r.links[i];
          if (
            d &&
            d.a_index != null &&
            d.a_index >= 0 &&
            d.a_index < a.trackIds.length
          ) {
            midTrackIds[i] = a.trackIds[d.a_index];
          } else {
            midTrackIds[i] = mintTrackId();
          }
        }
      } catch (e) {
        aerr(
          `gap_filler: linker a→mid failed for src=${midSrc}; minting fresh ids`,
          e,
        );
        for (let i = 0; i < midBoxes.length; i++) {
          midTrackIds[i] = mintTrackId();
        }
      }
    }

    // Stamp ids back onto the mid boxes so serialization round-trips
    // agree with the emitted track_ids array.
    for (let i = 0; i < midBoxes.length; i++) {
      midBoxes[i].track_id = midTrackIds[i];
    }

    const inserted: InsertedFrame = {
      virtualIndex,
      sourceIndex: midSrc,
      width: fetched.width,
      height: fetched.height,
      jpeg: fetched.blob,
      boxes: midBoxes,
      trackIds: midTrackIds,
      betweenKeptIndices: [a.index, b.index],
      depth,
      triggerReason: reason,
    };
    inserts.push(inserted);

    runLog?.write({
      kind: "gap_bisect",
      between: [a.index, b.index],
      source_index: midSrc,
      virtual_index: virtualIndex,
      depth,
      reason,
      boxes: midBoxes.length,
      track_ids: midTrackIds,
    });
    emit({
      type: "inserted_frame",
      between: [a.index, b.index],
      depth,
      trigger_reason: reason,
      source_index: midSrc,
      frame: {
        width: fetched.width,
        height: fetched.height,
        jpeg_base64: toBase64(fetched.blob),
        boxes: midBoxes.map((b2) => ({
          x: b2.x,
          y: b2.y,
          w: b2.w,
          h: b2.h,
          text: b2.text,
          score: Math.round(b2.score * 1000) / 1000,
          origin: b2.origin,
          track_id: b2.track_id,
        })),
        track_ids: midTrackIds,
      },
    });

    // Return the inserted frame as a KeptFrameForGapFill view so the
    // recursion can treat it as an ordinary endpoint of the next
    // sub-gap. We use the virtual index both as `index` and anywhere
    // downstream that keys off kept-frame index.
    return {
      index: virtualIndex,
      sourceIndex: midSrc,
      width: fetched.width,
      height: fetched.height,
      jpeg: fetched.blob,
      boxes: midBoxes,
      trackIds: midTrackIds,
    };
  };

  /**
   * Recurse into a gap: probe, insert if triggered, then bisect both
   * halves. Parallelizes the two halves because they're independent
   * once the mid frame exists.
   */
  const smoothGap = async (
    a: KeptFrameForGapFill,
    b: KeptFrameForGapFill,
    depth: number,
  ): Promise<void> => {
    totalProbes += 1;
    const srcGap = b.sourceIndex - a.sourceIndex;
    const tc = measureTrackChange(a, b);
    const reason = triggerReason(tc, srcGap);
    runLog?.write({
      kind: "gap_probe",
      between: [a.index, b.index],
      source_indices: [a.sourceIndex, b.sourceIndex],
      depth,
      src_gap: srcGap,
      move_max_px: Math.round(tc.moveMaxPx * 100) / 100,
      rel_w_max: Math.round(tc.relWMax * 1000) / 1000,
      rel_h_max: Math.round(tc.relHMax * 1000) / 1000,
      rel_area_max: Math.round(tc.relAreaMax * 1000) / 1000,
      new_track_area_frac_max:
        Math.round(tc.newTrackAreaFracMax * 10000) / 10000,
      linked_tracks: tc.linkedTracks,
      total_pairs: tc.totalPairs,
      triggered: reason != null,
      reason,
    });
    if (reason == null) return;
    if (depth >= GAP_FILLER_MAX_DEPTH) {
      runLog?.write({
        kind: "gap_terminate",
        between: [a.index, b.index],
        depth,
        reason: "depth_cap",
        final_move_max_px: tc.moveMaxPx,
      });
      return;
    }
    if (budgetExhausted()) {
      runLog?.write({
        kind: "gap_terminate",
        between: [a.index, b.index],
        depth,
        reason: "budget",
        inserts_so_far: inserts.length,
      });
      return;
    }
    const midSrc = Math.floor((a.sourceIndex + b.sourceIndex) / 2);
    if (midSrc <= a.sourceIndex || midSrc >= b.sourceIndex) {
      // Nothing to insert between — adjacent raw frames.
      runLog?.write({
        kind: "gap_terminate",
        between: [a.index, b.index],
        depth,
        reason: "no_raw_between",
      });
      return;
    }
    const mid = await detectInserted(a, b, midSrc, depth, reason);
    if (!mid) {
      runLog?.write({
        kind: "gap_terminate",
        between: [a.index, b.index],
        depth,
        reason: "fetch_failed",
      });
      return;
    }
    // Recurse on both halves. Parallel: they're independent now that
    // `mid` is fully populated. Both may recurse further.
    await Promise.all([
      smoothGap(a, mid, depth + 1),
      smoothGap(mid, b, depth + 1),
    ]);
  };

  // Top-level walk. Process all adjacent gaps in parallel. Within each
  // triggered gap the recursion itself parallelizes per-level. Failures
  // on one gap don't stop the others.
  const tasks: Promise<void>[] = [];
  for (let k = 0; k + 1 < kept.length; k++) {
    const a = kept[k];
    const b = kept[k + 1];
    // Cheap pre-filter: if source_gap is 1, there's literally no raw
    // frame to insert between them, so skip the trigger evaluation
    // entirely.
    if (b.sourceIndex - a.sourceIndex <= 1) continue;
    tasks.push(
      (async () => {
        const beforeInserts = inserts.length;
        await smoothGap(a, b, 0);
        if (inserts.length > beforeInserts) {
          triggeredTopLevelGaps += 1;
        }
      })().catch((e) => {
        aerr(
          `gap_filler: top-level gap ${a.index}→${b.index} threw`,
          e,
        );
      }),
    );
  }
  await Promise.all(tasks);

  alog("gap_filler: done", {
    kept_frames: kept.length,
    total_probes: totalProbes,
    top_level_triggered: triggeredTopLevelGaps,
    inserted_frames: inserts.length,
    curator_calls: curatorUsage.callCount,
    linker_calls: linkerUsage.callCount,
  });

  return {
    inserts,
    triggeredTopLevelGaps,
    totalProbes,
    curatorUsage,
    linkerUsage,
  };
}

// Gemini phase-1 detection stream.
//
// Emits the same NDJSON envelope as the OCR detect/stream endpoint:
//   {type: "start", ...}
//   {type: "frame", index, width, height, boxes: [...], raw}
//   {type: "done", matched_frames, total_boxes}
//   {type: "error", message}
//
// Frames run sequentially so each call can feed the model the running list
// of labels accumulated in prior frames, giving us consistent cross-frame
// identity without a separate reconciliation pass.

import {
  createEntry,
  putFrame,
  type FrameState,
} from "@/lib/server/gemini-cache";
import { fetchDeduplicatedFramesServer } from "@/lib/server/frames";
import { detectFrame, type KnownLabel } from "@/lib/server/openrouter";
import {
  ndjsonStreamResponse,
  normalizeQuery,
  readFormInputs,
} from "@/lib/server/route-helpers";

export const runtime = "nodejs";
// Give the model time on long-ish videos. 10 minutes is enough for a few
// dozen frames at flash speeds.
export const maxDuration = 600;

export async function POST(req: Request): Promise<Response> {
  const parsed = await readFormInputs(req);
  if ("error" in parsed) {
    return new Response(JSON.stringify({ detail: parsed.error }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const { file, query, fps, dedupThreshold, maxGap } = parsed;
  let { frameFrom, frameTo } = parsed;
  const qNorm = normalizeQuery(query);

  // We need frames before we can decide lo/hi, so fetch up front. Python
  // caches by SHA256(video) so this is cheap on re-run.
  let framesRes;
  try {
    framesRes = await fetchDeduplicatedFramesServer({
      file,
      fps: fps ?? undefined,
      dedupThreshold,
      maxGap,
    });
  } catch (e) {
    return new Response(
      JSON.stringify({
        detail: e instanceof Error ? e.message : "frame extraction failed",
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  const n = framesRes.deduplicatedCount;
  if (n === 0) {
    return new Response(JSON.stringify({ detail: "No frames to process." }), {
      status: 422,
      headers: { "Content-Type": "application/json" },
    });
  }

  frameFrom = Math.max(1, Math.min(n, frameFrom));
  frameTo = frameTo == null ? n : Math.max(1, Math.min(n, frameTo));
  if (frameFrom > frameTo) {
    return new Response(
      JSON.stringify({
        detail: `Invalid frame range: from=${frameFrom} to=${frameTo}`,
      }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }
  const lo = frameFrom;
  const hi = frameTo;

  const entry = createEntry({
    engine: "gemini",
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

  return ndjsonStreamResponse(async ({ emit, error }) => {
    emit({
      type: "start",
      video_hash: framesRes.videoHash,
      query,
      deduplicated_count: n,
      frame_from: lo,
      frame_to: hi,
      frame_indices: indices,
      total: indices.length,
    });

    if (!qNorm) {
      emit({ type: "done", matched_frames: 0, total_boxes: 0 });
      return;
    }

    const known: KnownLabel[] = [];
    let matchedFrames = 0;
    let totalBoxes = 0;

    for (const idx1 of indices) {
      const src = framesRes.frames[idx1 - 1];
      if (!src) continue;

      let boxes;
      let raw;
      try {
        const r = await detectFrame(src.blob, qNorm, known, src.width, src.height);
        boxes = r.boxes;
        raw = r.raw;
      } catch (e) {
        error(
          `Gemini detect failed on frame ${idx1}: ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
        return;
      }

      const state: FrameState = {
        width: src.width,
        height: src.height,
        matched: boxes,
        raw,
        blob: src.blob,
      };
      putFrame(entry, idx1, state);

      // Accumulate labels for the next frame's context. Encode each hit's
      // center in [ymin, xmin, ymax, xmax] normalized 0..1000 so the model
      // gets Gemini-native geometry back.
      for (const b of boxes) {
        if (!b.label) continue;
        known.push({
          label: b.label,
          text: b.text,
          bbox: [
            Math.round((b.y / Math.max(1, src.height)) * 1000),
            Math.round((b.x / Math.max(1, src.width)) * 1000),
            Math.round(((b.y + b.h) / Math.max(1, src.height)) * 1000),
            Math.round(((b.x + b.w) / Math.max(1, src.width)) * 1000),
          ],
        });
      }

      if (boxes.length > 0) matchedFrames += 1;
      totalBoxes += boxes.length;

      emit({
        type: "frame",
        index: idx1,
        width: src.width,
        height: src.height,
        boxes: boxes.map((b) => ({
          x: b.x,
          y: b.y,
          w: b.w,
          h: b.h,
          text: b.text,
          score: Math.round(b.score * 1000) / 1000,
          label: b.label,
        })),
        raw,
      });
    }

    emit({
      type: "done",
      matched_frames: matchedFrames,
      total_boxes: totalBoxes,
    });
  });
}

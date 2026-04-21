// Agentic teamwork phase-1 (per-frame curator).
//
// Pipeline for each requested frame:
//   1. Python OCR (Textract) has already produced candidate boxes for this
//      frame. We fan those out per-frame (parallel).
//   2. For each frame, ask Gemini 3 Pro: keep/drop every OCR box, and
//      propose any box OCR missed entirely.
//   3. Stream the curated result to the client as soon as the per-frame
//      Gemini call resolves (out-of-order completion is fine; events carry
//      their own frame `index`).
//
// Cross-frame labeling is NOT done here — phase-2 (the navigator) handles
// identity bookkeeping when it actually matters. The client's existing
// assignLabels pass still provides letter labels visually until then.

import { curateFrame } from "@/lib/server/agentic-curator";
import { alog } from "@/lib/server/agentic-log";
import { fetchDeduplicatedFramesServer } from "@/lib/server/frames";
import {
  createEntry,
  putFrame,
  type FrameState,
} from "@/lib/server/gemini-cache";
import { fetchOcrDetect } from "@/lib/server/ocr-client";
import { agenticCuratorConcurrency, type ServerBox } from "@/lib/server/openrouter";
import {
  ndjsonStreamResponse,
  normalizeQuery,
  readFormInputs,
} from "@/lib/server/route-helpers";

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
    });

    if (!qNorm) {
      emit({ type: "done", matched_frames: 0, total_boxes: 0 });
      return;
    }

    // Accumulators are updated from the async worker pool; we rely on the
    // single-threaded event loop to make these writes safe.
    let matchedFrames = 0;
    let totalBoxes = 0;
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
          });
          kept = r.kept;
          added = r.added;
          dropped = r.dropped;
          curatorRaw = r.raw;
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

    const elapsed = Date.now() - t0;
    alog("detect route done", {
      elapsed_ms: elapsed,
      matched_frames: matchedFrames,
      total_boxes: totalBoxes,
    });

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

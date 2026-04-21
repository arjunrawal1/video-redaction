// Shared body for /api/gemini/backtrack and /api/gemini/forward routes.
//
// Both endpoints: validate inputs, fetch frames, locate or populate the
// Gemini cache entry (transparently running phase-1 if the cache missed),
// then invoke the pass driver with the appropriate direction and emit one
// NDJSON frame event per added hit. Keeps backtrack.ts and forward.ts as
// thin wrappers.

import { compareFrames, detectFrame, type KnownLabel } from "./openrouter";
import {
  createEntry,
  getEntry,
  putFrame,
  type CacheEntry,
  type FrameState,
} from "./gemini-cache";
import { fetchDeduplicatedFramesServer, type ServerFrame } from "./frames";
import { runPass, type PassDirection } from "./pass-driver";
import {
  ndjsonStreamResponse,
  normalizeQuery,
  readFormInputs,
} from "./route-helpers";

export async function runPassRoute(
  req: Request,
  direction: PassDirection,
): Promise<Response> {
  const parsed = await readFormInputs(req);
  if ("error" in parsed) {
    return new Response(JSON.stringify({ detail: parsed.error }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }
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

  return ndjsonStreamResponse(async ({ emit, error }) => {
    emit({
      type: "start",
      video_hash: framesRes.videoHash,
      query,
      frame_from: lo,
      frame_to: hi,
      total_frames: hi - lo + 1,
    });

    if (!qNorm) {
      emit({ type: "done", added_frames: 0, added_boxes: 0 });
      return;
    }

    // Cache hit or transparent phase-1 population.
    let entry = getEntry({
      engine: "gemini",
      videoHash: framesRes.videoHash,
      queryNorm: qNorm,
      fps: fps ?? null,
      dedupThreshold,
      frameFrom: lo,
      frameTo: hi,
    });
    if (!entry) {
      try {
        entry = await populatePhase1({
          framesRes,
          query,
          qNorm,
          fps: fps ?? null,
          dedupThreshold,
          lo,
          hi,
        });
      } catch (e) {
        error(
          "Inline phase-1 failed: " +
            (e instanceof Error ? e.message : String(e)),
        );
        return;
      }
    }

    const touched = new Set<number>();
    let addedBoxes = 0;

    try {
      await runPass(
        entry,
        direction,
        async (anchor, srcFrame, targetFrame, queryNorm) => {
          const r = await compareFrames(
            srcFrame.blob,
            targetFrame.blob,
            queryNorm,
            anchor,
            srcFrame.width,
            srcFrame.height,
            targetFrame.width,
            targetFrame.height,
          );
          return r.box;
        },
        (hit) => {
          touched.add(hit.frameIdx1);
          addedBoxes += 1;
          const f = entry!.perFrame[hit.frameIdx1];
          emit({
            type: "frame",
            index: hit.frameIdx1,
            width: f?.width ?? 0,
            height: f?.height ?? 0,
            box: {
              x: hit.box.x,
              y: hit.box.y,
              w: hit.box.w,
              h: hit.box.h,
              text: hit.box.text,
              score: Math.round(hit.box.score * 1000) / 1000,
              label: hit.box.label,
            },
            origin: direction === "backward" ? "backtrack" : "forward",
          });
        },
      );
    } catch (e) {
      error(
        "Pass driver failed: " +
          (e instanceof Error ? e.message : String(e)),
      );
      return;
    }

    emit({
      type: "done",
      added_frames: touched.size,
      added_boxes: addedBoxes,
    });
  });
}

async function populatePhase1(opts: {
  framesRes: { videoHash: string; frames: ServerFrame[] };
  query: string;
  qNorm: string;
  fps: number | null;
  dedupThreshold: number;
  lo: number;
  hi: number;
}): Promise<CacheEntry> {
  const entry = createEntry({
    engine: "gemini",
    videoHash: opts.framesRes.videoHash,
    queryNorm: opts.qNorm,
    fps: opts.fps,
    dedupThreshold: opts.dedupThreshold,
    frameFrom: opts.lo,
    frameTo: opts.hi,
  });
  const known: KnownLabel[] = [];
  for (let idx1 = opts.lo; idx1 <= opts.hi; idx1++) {
    const src = opts.framesRes.frames[idx1 - 1];
    if (!src) continue;
    const r = await detectFrame(
      src.blob,
      opts.qNorm,
      known,
      src.width,
      src.height,
    );
    const state: FrameState = {
      width: src.width,
      height: src.height,
      matched: r.boxes,
      raw: r.raw,
      blob: src.blob,
    };
    putFrame(entry, idx1, state);
    for (const b of r.boxes) {
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
  }
  return entry;
}

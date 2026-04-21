// Agentic teamwork phase-2: the navigator.
//
// Picks up the phase-1 curator output from the teamwork cache and hands
// it to GPT-5.4 as one of two architectures:
//
//   - "cascade" (default): multi-agent. One focused agent per phase-1
//     transition; each cascades forward until corrections stop
//     propagating. See lib/server/agentic-cascade.ts.
//   - "single": legacy free-roaming single-agent navigator. Kept behind
//     AGENTIC_NAV_MODE=single for fallback / A/B comparison.
//
// Both modes stream the same NDJSON event protocol; cascade emits
// `agent_start` / `agent_end` envelopes and tags every event with an
// `agent_id` so the UI can distinguish work done by different agents.

import { alog, aerr } from "@/lib/server/agentic-log";
import { runCascadeNavigator } from "@/lib/server/agentic-cascade";
import {
  runNavigator,
  type NavFrameState,
  type NavHit,
  type NavigatorEvent,
} from "@/lib/server/agentic-navigator";
import { fetchDeduplicatedFramesServer } from "@/lib/server/frames";
import { getEntry } from "@/lib/server/gemini-cache";
import { agenticNavMode } from "@/lib/server/openrouter";
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

  return ndjsonStreamResponse(async ({ emit, error }) => {
    alog("navigate route start", {
      video_hash: framesRes.videoHash,
      query,
      query_norm: qNorm,
      frame_from: lo,
      frame_to: hi,
    });

    emit({
      type: "start",
      video_hash: framesRes.videoHash,
      query,
      frame_from: lo,
      frame_to: hi,
      total_frames: hi - lo + 1,
    });
    if (!qNorm) {
      emit({ type: "done", added_boxes: 0, removed_boxes: 0, total_steps: 0 });
      return;
    }

    const entry = getEntry({
      engine: "teamwork",
      videoHash: framesRes.videoHash,
      queryNorm: qNorm,
      fps: fps ?? null,
      dedupThreshold,
      frameFrom: lo,
      frameTo: hi,
    });
    if (!entry) {
      alog("navigate route cache miss", {
        video_hash: framesRes.videoHash,
        query_norm: qNorm,
        frame_from: lo,
        frame_to: hi,
      });
      error(
        "No teamwork cache entry — run /api/teamwork/detect/stream first for this range.",
      );
      return;
    }

    const navFrames: NavFrameState[] = [];
    for (let idx1 = lo; idx1 <= hi; idx1++) {
      const f = entry.perFrame[idx1];
      if (!f) continue;
      const hits: NavHit[] = f.matched.map((b, i) => ({
        ...b,
        hit_id: `H${idx1}-${i}`,
      }));
      navFrames.push({
        index: idx1,
        width: f.width,
        height: f.height,
        blob: f.blob,
        hits,
        ocrBoxes: f.ocrMatched ? f.ocrMatched.map((b) => ({ ...b })) : [],
        ocrRaw: f.ocrRaw ?? null,
      });
    }

    if (navFrames.length === 0) {
      emit({ type: "done", added_boxes: 0, removed_boxes: 0, total_steps: 0 });
      return;
    }

    // Single NDJSON emitter — handles every NavigatorEvent variant the
    // cascade or single-agent navigator can produce. Keeping one forwarder
    // makes the two modes observationally identical from the client's
    // point of view except for the agent-scoped event envelopes.
    const forward = (ev: NavigatorEvent): void => {
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

    const mode = agenticNavMode();
    alog("navigate route: running", { mode });

    try {
      let added = 0;
      let removed = 0;
      let totalSteps = 0;
      let finishSummary: string | null = null;

      if (mode === "cascade") {
        const result = await runCascadeNavigator({
          query: qNorm,
          frames: navFrames,
          onEvent: forward,
        });
        added = result.added;
        removed = result.removed;
        totalSteps = result.totalSteps;
        finishSummary = result.finishSummary;
      } else {
        const result = await runNavigator({
          query: qNorm,
          frames: navFrames,
          onEvent: forward,
        });
        added = result.added;
        removed = result.removed;
        totalSteps = result.totalSteps;
        finishSummary = result.finishSummary;
      }

      // Mirror mutations back into the shared cache so a downstream
      // re-run of another pass (or a page reload) sees the navigator's
      // work. FrameState.matched is the source of truth for phase-1.
      for (const nf of navFrames) {
        const f = entry.perFrame[nf.index];
        if (!f) continue;
        f.matched = nf.hits.map(({ hit_id: _hit_id, ...rest }) => rest);
      }

      alog("navigate route done", {
        mode,
        added_boxes: added,
        removed_boxes: removed,
        total_steps: totalSteps,
        finish_summary: finishSummary,
      });

      emit({
        type: "done",
        added_boxes: added,
        removed_boxes: removed,
        total_steps: totalSteps,
      });
    } catch (e) {
      aerr("navigate route navigator threw", e);
      error(
        "Navigator failed: " + (e instanceof Error ? e.message : String(e)),
      );
    }
  });
}

function json(detail: string, status: number): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

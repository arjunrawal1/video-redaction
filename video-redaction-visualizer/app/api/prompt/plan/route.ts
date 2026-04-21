import { aerr, alog } from "@/lib/server/agentic-log";
import { buildSceneSummary, parsePromptToPredicate } from "@/lib/server/prompt/planner";
import {
  createPlanSession,
  toPublicPlanSession,
} from "@/lib/server/prompt/plan-session";
import { readPromptFormInputs } from "@/lib/server/route-helpers";
import { openRunLog } from "@/lib/server/run-log";

export const runtime = "nodejs";
export const maxDuration = 600;

function jsonBody(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function POST(req: Request): Promise<Response> {
  const parsed = await readPromptFormInputs(req);
  if ("error" in parsed) {
    return jsonBody({ detail: parsed.error }, 400);
  }
  if (!parsed.prompt) {
    return jsonBody({ detail: "Missing prompt." }, 400);
  }

  const runLog = openRunLog("prompt-plan", {
    query: parsed.prompt,
    prompt: parsed.prompt,
    frame_from: parsed.frameFrom,
    frame_to: parsed.frameTo,
    fps: parsed.fps,
    dedup_threshold: parsed.dedupThreshold,
    max_gap: parsed.maxGap,
  });

  alog("prompt plan route start", {
    prompt: parsed.prompt,
    frame_from: parsed.frameFrom,
    frame_to: parsed.frameTo,
    fps: parsed.fps,
    run_log_path: runLog.path,
  });

  const t0 = Date.now();
  try {
    const summary = await buildSceneSummary({
      file: parsed.file,
      frameFrom: parsed.frameFrom,
      frameTo: parsed.frameTo ?? undefined,
      fps: parsed.fps,
      dedupThreshold: parsed.dedupThreshold,
      maxGap: parsed.maxGap,
      runLog,
    });

    const planned = await parsePromptToPredicate({
      prompt: parsed.prompt,
      sceneSummary: summary,
      runLog,
    });

    const session = createPlanSession({
      prompt: parsed.prompt,
      predicate: planned.predicate,
      hash: planned.hash,
      sceneSummary: summary,
    });

    runLog.write({
      kind: "run_end",
      status: "ok",
      session_id: session.session_id,
      predicate_hash: planned.hash,
    });
    await runLog.close();
    alog("prompt plan route done", {
      elapsed_ms: Date.now() - t0,
      session_id: session.session_id,
      predicate_hash: planned.hash,
      predicate_kind:
        (planned.predicate as { kind?: string }).kind ?? "unknown",
    });
    return jsonBody({
      ...toPublicPlanSession(session),
      run_log_path: runLog.path,
    });
  } catch (e) {
    runLog.write({
      kind: "run_end",
      status: "error",
      error: e instanceof Error ? e.message : String(e),
    });
    await runLog.close();
    aerr("prompt plan route threw", e);
    return jsonBody(
      {
        detail:
          e instanceof Error
            ? `Prompt planning failed: ${e.message}`
            : "Prompt planning failed.",
        run_log_path: runLog.path,
      },
      500,
    );
  }
}

import { randomUUID } from "node:crypto";
import type {
  PlanSession,
  Predicate,
  PredicateHash,
  SceneSummary,
} from "./types";

function parseMs(raw: string | undefined, fallback: number): number {
  const n = Number(raw ?? "");
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

export const PLAN_SESSION_TTL_MS = parseMs(
  process.env.PROMPT_PLAN_SESSION_TTL_MS,
  10 * 60 * 1000,
);

export type PlanSessionState = PlanSession & {
  hash: PredicateHash;
  sceneSummary: SceneSummary;
  expires_at: number;
};

const _sessions = new Map<string, PlanSessionState>();

function pruneExpired(now = Date.now()): void {
  for (const [id, sess] of _sessions.entries()) {
    if (sess.expires_at <= now) {
      _sessions.delete(id);
    }
  }
}

export function createPlanSession(args: {
  prompt: string;
  predicate: Predicate;
  hash: PredicateHash;
  sceneSummary: SceneSummary;
}): PlanSessionState {
  pruneExpired();
  const session: PlanSessionState = {
    session_id: randomUUID(),
    prompt: args.prompt,
    predicate: args.predicate,
    hash: args.hash,
    sample_frame_indices: args.sceneSummary.sample_frame_indices,
    sceneSummary: args.sceneSummary,
    expires_at: Date.now() + PLAN_SESSION_TTL_MS,
  };
  _sessions.set(session.session_id, session);
  return session;
}

export function toPublicPlanSession(session: PlanSessionState): PlanSession {
  return {
    session_id: session.session_id,
    prompt: session.prompt,
    predicate: session.predicate,
    hash: session.hash,
    sample_frame_indices: session.sample_frame_indices,
  };
}

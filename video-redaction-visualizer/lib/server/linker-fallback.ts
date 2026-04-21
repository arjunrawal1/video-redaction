// Deterministic IoU + text-similarity fallback for the agentic linker.
//
// When the Gemini call in lib/server/agentic-linker.ts throws (rate
// limit, 5xx, schema violation, timeout), the detect-stream route calls
// this module instead so the pipeline degrades gracefully to "good-
// enough" correspondence rather than losing track ids entirely.
//
// The scoring mirrors the decision a human would make if they only had
// geometry + OCR text to go on:
//   - Strong IoU → same redaction.
//   - Moderate IoU + matching text → same redaction.
//   - Neither → disallow the link (prefer a fresh track).
//
// One-to-one assignment is solved by a greedy pass over the sorted
// (B, A) pairs. That's not globally optimal, but for the usual <20
// boxes/frame it's indistinguishable from Hungarian in practice.

import { alog } from "./agentic-log";
import type { LinkDecision } from "./agentic-linker";
import type { ServerBox } from "./openrouter";

const _IOU_WEIGHT = 0.7;
const _TEXT_WEIGHT = 0.3;
// Minimum score to even consider linking. Tuned so that two visually
// disjoint boxes with coincidentally similar text don't get glued
// together; score < 0.1 is essentially "no real spatial evidence".
const _MIN_LINK_SCORE = 0.12;

export function linkFramePairFallback(opts: {
  boxesA: ServerBox[];
  boxesB: ServerBox[];
  frameWidthA: number;
  frameHeightA: number;
  frameWidthB: number;
  frameHeightB: number;
}): LinkDecision[] {
  const { boxesA, boxesB } = opts;

  if (boxesB.length === 0) return [];
  if (boxesA.length === 0) {
    return boxesB.map((_, i) => ({ b_index: i, a_index: null, reason: "fallback: A empty" }));
  }

  // Use normalized-coord IoU so A/B can have different frame sizes
  // (rare, but the curator path doesn't enforce equal dims).
  const normA = boxesA.map((b) =>
    toNormalized(b, opts.frameWidthA, opts.frameHeightA),
  );
  const normB = boxesB.map((b) =>
    toNormalized(b, opts.frameWidthB, opts.frameHeightB),
  );

  type Candidate = { bi: number; ai: number; score: number };
  const candidates: Candidate[] = [];
  for (let bi = 0; bi < boxesB.length; bi++) {
    for (let ai = 0; ai < boxesA.length; ai++) {
      const iou = normalizedIoU(normA[ai], normB[bi]);
      const tsim = textSim(boxesA[ai].text, boxesB[bi].text);
      const score = _IOU_WEIGHT * iou + _TEXT_WEIGHT * tsim;
      if (score < _MIN_LINK_SCORE) continue;
      candidates.push({ bi, ai, score });
    }
  }
  candidates.sort((a, b) => b.score - a.score);

  const aClaimed = new Set<number>();
  const bClaimed = new Set<number>();
  const chosen = new Map<number, { ai: number; score: number }>();
  for (const c of candidates) {
    if (bClaimed.has(c.bi) || aClaimed.has(c.ai)) continue;
    bClaimed.add(c.bi);
    aClaimed.add(c.ai);
    chosen.set(c.bi, { ai: c.ai, score: c.score });
  }

  const out: LinkDecision[] = [];
  for (let bi = 0; bi < boxesB.length; bi++) {
    const pick = chosen.get(bi);
    if (pick) {
      out.push({
        b_index: bi,
        a_index: pick.ai,
        reason: `fallback: iou+text score=${pick.score.toFixed(3)}`,
      });
    } else {
      out.push({
        b_index: bi,
        a_index: null,
        reason: "fallback: no match above threshold",
      });
    }
  }
  return out;
}

type NormRect = { x: number; y: number; w: number; h: number };

function toNormalized(b: ServerBox, W: number, H: number): NormRect {
  const w = Math.max(1, W);
  const h = Math.max(1, H);
  return { x: b.x / w, y: b.y / h, w: b.w / w, h: b.h / h };
}

function normalizedIoU(a: NormRect, b: NormRect): number {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0;
}

function normalizeText(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

// Levenshtein similarity ratio in [0, 1]. Same definition as
// lib/labeling.ts on the client so fallback behavior matches the UI's
// purely-geometric path when track_ids are absent.
function textSim(a: string, b: string): number {
  const na = normalizeText(a);
  const nb = normalizeText(b);
  if (!na || !nb) return 0;
  if (na === nb) return 1;
  if (na.includes(nb) || nb.includes(na)) return 0.9;
  const la = na.length;
  const lb = nb.length;
  let prev = new Array<number>(lb + 1);
  let curr = new Array<number>(lb + 1);
  for (let j = 0; j <= lb; j++) prev[j] = j;
  for (let i = 1; i <= la; i++) {
    curr[0] = i;
    for (let j = 1; j <= lb; j++) {
      const cost = na[i - 1] === nb[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return 1 - prev[lb] / Math.max(la, lb);
}

// -- Fix-box track-id inheritance -----------------------------------------
//
// The phase-1.5 linker runs once after the curator finishes and stamps
// track_ids on boxes that existed at that moment. The navigator (cascade
// or single-agent) can subsequently add new boxes via add_box /
// adopt_ocr_box; those boxes land without track_ids, which means the UI
// can't interpolate them across frames and the exporter renders them
// statically on just their keyframe window.
//
// This helper closes that gap: when a focused agent adds a box at frame
// F, we look at the already-tracked boxes on frames F-1 and F+1 and
// decide whether this new box is a continuation of one of them. If so,
// it inherits the neighbor's track_id; otherwise the caller mints a
// fresh id for it.
//
// The scoring is deliberately the same IoU+text combination used by the
// phase-1.5 linker fallback, for consistency: if two codepaths disagree
// about "is this the same redaction?", track-based interpolation will
// look wrong. By sharing the primitives we guarantee the navigator's
// identity decisions and the phase-1.5 linker's decisions use the same
// notion of similarity.

const _FIX_MIN_INHERIT_SCORE = 0.15;

/**
 * Try to inherit a ``track_id`` for a newly-added (``origin: "fix"``)
 * box at ``frame_index``.
 *
 * Returns the inherited id when a neighbor in frame F-1 or F+1 scores
 * above the threshold AND that neighbor's id is not already claimed by
 * another box on the current frame (one-to-one). Returns ``null`` when
 * no suitable neighbor exists — the caller then mints a fresh id so the
 * box still participates in tweening across subsequent sibling fix
 * boxes or the client-side label assignment.
 *
 * Prev-frame inheritance is preferred over next-frame: if a box in F-1
 * is tracked and a plausible match, that's the stronger continuity
 * signal because the linker operates strictly forward. Next-frame
 * inheritance is a useful secondary for cascades that run backward
 * (they add on earlier frames and their "continuation" lives in F+1).
 */
export function inheritFixTrackId(opts: {
  newBox: ServerBox;
  newFrameWidth: number;
  newFrameHeight: number;
  prevFrame: null | {
    hits: ServerBox[];
    width: number;
    height: number;
  };
  nextFrame: null | {
    hits: ServerBox[];
    width: number;
    height: number;
  };
  /**
   * Track ids already stamped on boxes in the CURRENT frame. Used to
   * enforce one-to-one — a neighbor's id can only be inherited once per
   * frame, matching the phase-1.5 linker's contract.
   */
  claimedOnCurrentFrame: ReadonlySet<string>;
}): { track_id: string | null; source: "prev" | "next" | null; score: number } {
  const {
    newBox,
    newFrameWidth,
    newFrameHeight,
    prevFrame,
    nextFrame,
    claimedOnCurrentFrame,
  } = opts;

  const newNorm = toNormalized(newBox, newFrameWidth, newFrameHeight);

  const best = (args: {
    hits: ServerBox[];
    width: number;
    height: number;
  }): { track_id: string; score: number } | null => {
    let winner: { track_id: string; score: number } | null = null;
    for (const h of args.hits) {
      const id = h.track_id;
      if (!id) continue;
      if (claimedOnCurrentFrame.has(id)) continue;
      const hn = toNormalized(h, args.width, args.height);
      const iou = normalizedIoU(hn, newNorm);
      const tsim = textSim(h.text, newBox.text);
      const score = 0.7 * iou + 0.3 * tsim;
      if (score < _FIX_MIN_INHERIT_SCORE) continue;
      if (!winner || score > winner.score) {
        winner = { track_id: id, score };
      }
    }
    return winner;
  };

  const prevHit = prevFrame ? best(prevFrame) : null;
  const nextHit = nextFrame ? best(nextFrame) : null;

  // Prefer prev-frame continuity; ties broken by numerical score.
  if (prevHit && (!nextHit || prevHit.score >= nextHit.score)) {
    return { track_id: prevHit.track_id, source: "prev", score: prevHit.score };
  }
  if (nextHit) {
    return { track_id: nextHit.track_id, source: "next", score: nextHit.score };
  }
  return { track_id: null, source: null, score: 0 };
}

/**
 * Convenience wrapper that logs the decision and mints a fresh id when
 * inheritance fails. Use this at ``add_box`` / ``adopt_ocr_box`` sites
 * so the log line tells you whether the new box chains into an existing
 * track and, if not, what its fresh id is.
 *
 * ``mintPrefix`` lets each caller (cascade vs single-agent navigator)
 * tag novel fix-track ids so the logs make the origin obvious —
 * ``f:C...`` from cascade vs ``f:H...`` from the navigator.
 */
export function resolveFixTrackId(opts: {
  agentId?: string;
  frameIndex: number;
  hitId: string;
  newBox: ServerBox;
  newFrameWidth: number;
  newFrameHeight: number;
  prevFrame: null | {
    hits: ServerBox[];
    width: number;
    height: number;
  };
  nextFrame: null | {
    hits: ServerBox[];
    width: number;
    height: number;
  };
  claimedOnCurrentFrame: ReadonlySet<string>;
}): string {
  const { agentId, frameIndex, hitId } = opts;
  const decision = inheritFixTrackId(opts);
  if (decision.track_id) {
    alog(
      `${agentId ? `[${agentId}] ` : ""}fix-box inherit track_id ` +
        `#${frameIndex} ${hitId} → ${decision.track_id}`,
      { source: decision.source, score: Number(decision.score.toFixed(3)) },
    );
    return decision.track_id;
  }
  const minted = `f:${hitId}`;
  alog(
    `${agentId ? `[${agentId}] ` : ""}fix-box mint track_id ` +
      `#${frameIndex} ${hitId} → ${minted}`,
  );
  return minted;
}

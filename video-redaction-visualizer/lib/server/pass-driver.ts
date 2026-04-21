// Engine-neutral iterative pass driver ported from
// backend/app/ocr_backtrack.py. Same correspondence predicates, same
// in-place mutation pattern that turns iteration into recursion.
//
// The Gemini pipeline plugs in its own partial finder (compareFrames over
// two JPEGs). This module does not import any LLM code — it's pure algo.

import type { FrameState, QueryCacheEntry } from "./gemini-cache";
import type { ServerBox } from "./openrouter";

const LINK_CENTER_DIST = 0.08;
const TEXT_THRESHOLD = 0.75;

// --- text similarity -----------------------------------------------------

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function similarityRatio(a: string, b: string): number {
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  const la = a.length;
  const lb = b.length;
  let prev = new Array<number>(lb + 1);
  let curr = new Array<number>(lb + 1);
  for (let j = 0; j <= lb; j++) prev[j] = j;
  for (let i = 1; i <= la; i++) {
    curr[0] = i;
    for (let j = 1; j <= lb; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return 1 - prev[lb] / Math.max(la, lb);
}

function textMatches(a: string, b: string): boolean {
  const na = normalize(a);
  const nb = normalize(b);
  if (!na || !nb) return false;
  if (na === nb || na.includes(nb) || nb.includes(na)) return true;
  return similarityRatio(na, nb) >= TEXT_THRESHOLD;
}

// --- spatial --------------------------------------------------------------

function pixelCenter(
  box: ServerBox,
  frameW: number,
  frameH: number,
): { cx: number; cy: number } {
  return {
    cx: (box.x + box.w / 2) / Math.max(1, frameW),
    cy: (box.y + box.h / 2) / Math.max(1, frameH),
  };
}

function hasCorrespondence(
  current: ServerBox,
  currentFrame: FrameState,
  otherFrame: FrameState,
): boolean {
  const { cx, cy } = pixelCenter(current, currentFrame.width, currentFrame.height);
  for (const p of otherFrame.matched) {
    if (!textMatches(current.text, p.text)) continue;
    const { cx: px, cy: py } = pixelCenter(p, otherFrame.width, otherFrame.height);
    const dist = Math.hypot(cx - px, cy - py);
    if (dist <= LINK_CENTER_DIST) return true;
  }
  return false;
}

function boxOverlapsExisting(box: ServerBox, existing: ServerBox[]): boolean {
  const ax1 = box.x;
  const ay1 = box.y;
  const ax2 = box.x + box.w;
  const ay2 = box.y + box.h;
  const aArea = Math.max(1, box.w * box.h);
  for (const e of existing) {
    const bx1 = e.x;
    const by1 = e.y;
    const bx2 = e.x + e.w;
    const by2 = e.y + e.h;
    const ix1 = Math.max(ax1, bx1);
    const iy1 = Math.max(ay1, by1);
    const ix2 = Math.min(ax2, bx2);
    const iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    if (inter === 0) continue;
    const bArea = Math.max(1, e.w * e.h);
    const union = aArea + bArea - inter;
    if (union > 0 && inter / union >= 0.5) return true;
  }
  return false;
}

// --- iterative driver ----------------------------------------------------

export type AddedHit = { frameIdx1: number; box: ServerBox };

export type PartialFinder = (
  anchor: ServerBox,
  sourceFrame: FrameState,
  targetFrame: FrameState,
  queryNorm: string,
) => Promise<ServerBox | null>;

export type PassDirection = "backward" | "forward";

/**
 * Iterative pass driver. For each frame in iteration order, compare every
 * hit to the adjacent frame; if no correspondence exists, invoke
 * `partialFinder` and, on success, append the returned box to the adjacent
 * frame's matched list in place. That mutation is visible when the outer
 * loop reaches the adjacent frame next, giving recursive walk-back (or
 * walk-forward) through iteration.
 *
 * `onAdded` is called for every appended hit so the caller can stream the
 * event to the client immediately.
 */
export async function runPass(
  entry: QueryCacheEntry,
  direction: PassDirection,
  partialFinder: PartialFinder,
  onAdded: (h: AddedHit) => void,
): Promise<AddedHit[]> {
  const added: AddedHit[] = [];
  const lo = entry.frameFrom;
  const hi = entry.frameTo;

  const range: number[] = [];
  if (direction === "backward") {
    for (let i = hi; i >= lo + 1; i--) range.push(i);
  } else {
    for (let i = lo; i <= hi - 1; i++) range.push(i);
  }

  for (const idx of range) {
    const current = entry.perFrame[idx];
    const otherIdx = direction === "backward" ? idx - 1 : idx + 1;
    const other = entry.perFrame[otherIdx];
    if (!current || !other) continue;

    // Snapshot so we don't re-scan a box we just appended this iteration.
    const currentHits = current.matched.slice();
    for (const hit of currentHits) {
      if (hasCorrespondence(hit, current, other)) continue;
      const partial = await partialFinder(hit, current, other, entry.queryNorm);
      if (!partial) continue;
      if (boxOverlapsExisting(partial, other.matched)) continue;

      // Inherit anchor label so the chain shares identity across frames.
      if (partial.label == null && hit.label != null) {
        partial.label = hit.label;
      }
      partial.origin = direction === "backward" ? "backtrack" : "forward";

      other.matched.push(partial);
      const rec: AddedHit = { frameIdx1: otherIdx, box: partial };
      added.push(rec);
      onAdded(rec);
    }
  }

  return added;
}

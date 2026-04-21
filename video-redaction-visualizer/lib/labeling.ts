// Cross-frame hit labeling.
//
// Walks frames in ascending order and assigns each hit a stable A, B, C, ...
// identity. Two paths, in order of preference:
//
//   1. Server-assigned `track_id` (teamwork engine's phase-1.5 linker):
//      if every box in a frame carries a `track_id`, we use those as the
//      chain identity and letters are just a first-seen dense encoding
//      (`t3` → `A` the first time it appears, then stays `A` forever).
//      Gemini already decided correspondence, we just render it.
//
//   2. Geometric fallback (OCR engine, pre-linker state, mixed frames):
//      a hit in frame k inherits its label from a hit in frame k-1 when
//      both (a) their texts match (normalized equality, substring
//      either way, or Levenshtein similarity >= _TEXT_THRESHOLD) AND
//      (b) their normalized-coordinate centers are within _LINK_DIST.
//      Otherwise it gets the next unused letter. This mirrors the
//      correspondence predicate used by the backend backtrack pass so
//      that backtrack chains -- which are literally the same real-world
//      text across frames -- collapse to one identity.

import type { DetectionBox } from "@/lib/frames-api";

// Keep in sync with BACKTRACK_LINK_CENTER_DIST on the backend.
const _LINK_DIST = 0.08;
// Slightly looser than backtrack's 0.82 because the label's job is to glue
// a chain together, and text-wise adjacent frames are usually nearly
// identical once backtrack has done its work.
const _TEXT_THRESHOLD = 0.75;

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

// Levenshtein similarity ratio in [0, 1]. Reasonable stand-in for Python's
// difflib.SequenceMatcher on the frontend where we can't import difflib.
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
  return similarityRatio(na, nb) >= _TEXT_THRESHOLD;
}

function centerNormalized(
  box: DetectionBox,
  frameW: number,
  frameH: number,
): { cx: number; cy: number } {
  return {
    cx: (box.x + box.w / 2) / Math.max(1, frameW),
    cy: (box.y + box.h / 2) / Math.max(1, frameH),
  };
}

/**
 * Turn a 0-indexed integer into an Excel-style label (A..Z, AA..AZ, BA..).
 * Used so we don't run out of letters on videos with many distinct hits.
 */
export function toLetterLabel(n: number): string {
  let out = "";
  let v = n + 1;
  while (v > 0) {
    const rem = (v - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    v = Math.floor((v - 1) / 26);
  }
  return out;
}

type PrevLabeled = {
  cx: number;
  cy: number;
  text: string;
  label: string;
};

export type FrameLike = {
  width: number;
  height: number;
  boxes: DetectionBox[];
};

export type FrameLabelMap = Record<number, string[]>;

/**
 * Compute labels for every hit in every scanned frame.
 *
 * @param framesByIndex  1-indexed map of scanned frames. Missing keys mean
 *                       the frame wasn't scanned; treated as a chain break.
 * @param totalFrames    Iterate 1..totalFrames so gaps reset continuity
 *                       (strict adjacency). Pass the total number of
 *                       deduplicated frames from the API response.
 */
export function assignLabels(
  framesByIndex: Record<number, FrameLike>,
  totalFrames: number,
): FrameLabelMap {
  const out: FrameLabelMap = {};
  let nextLabelIdx = 0;
  let prevLabeled: PrevLabeled[] = [];

  // First-seen track_id → letter mapping. Shared across the whole range
  // so the same server identity always maps to the same letter on screen.
  const labelByTrack = new Map<string, string>();
  const labelForTrack = (trackId: string): string => {
    const hit = labelByTrack.get(trackId);
    if (hit) return hit;
    const fresh = toLetterLabel(nextLabelIdx++);
    labelByTrack.set(trackId, fresh);
    return fresh;
  };

  for (let k = 1; k <= totalFrames; k++) {
    const entry = framesByIndex[k];
    if (!entry) {
      // Unscanned frame: break the chain so reappearances after a gap get
      // fresh letters (strict adjacency). See the module docstring.
      prevLabeled = [];
      continue;
    }

    // Fast path: server gave us a track_id on every box → render directly.
    // We still append to prevLabeled so a later frame that loses its
    // track_ids (e.g. navigator added a `fix` box after the linker ran)
    // can fall back cleanly to geometric matching against the last frame.
    const allTracked =
      entry.boxes.length > 0 &&
      entry.boxes.every((b) => typeof b.track_id === "string" && b.track_id);
    if (allTracked) {
      const labelsForFrame = entry.boxes.map((b) =>
        labelForTrack(b.track_id as string),
      );
      out[k] = labelsForFrame;
      prevLabeled = entry.boxes.map((b, i) => {
        const { cx, cy } = centerNormalized(b, entry.width, entry.height);
        return { cx, cy, text: b.text, label: labelsForFrame[i] };
      });
      continue;
    }

    const labelsForFrame: string[] = [];
    const nextPrev: PrevLabeled[] = [];
    const claimed = new Set<number>();

    for (const box of entry.boxes) {
      const { cx, cy } = centerNormalized(box, entry.width, entry.height);

      // If this specific box carries a track_id even though the frame
      // as a whole doesn't, still prefer the server identity for it.
      if (typeof box.track_id === "string" && box.track_id) {
        const label = labelForTrack(box.track_id);
        labelsForFrame.push(label);
        nextPrev.push({ cx, cy, text: box.text, label });
        continue;
      }

      // Pick the closest prev hit that satisfies both predicates; first-come
      // wins if distances tie. One-to-one matching via `claimed` prevents a
      // single prev-label from being inherited by two current hits.
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let i = 0; i < prevLabeled.length; i++) {
        if (claimed.has(i)) continue;
        const p = prevLabeled[i];
        const dx = p.cx - cx;
        const dy = p.cy - cy;
        const dist = Math.hypot(dx, dy);
        if (dist > _LINK_DIST) continue;
        if (!textMatches(p.text, box.text)) continue;
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }

      const label =
        bestIdx >= 0
          ? prevLabeled[bestIdx].label
          : toLetterLabel(nextLabelIdx++);
      if (bestIdx >= 0) claimed.add(bestIdx);

      labelsForFrame.push(label);
      nextPrev.push({ cx, cy, text: box.text, label });
    }

    out[k] = labelsForFrame;
    prevLabeled = nextPrev;
  }

  return out;
}

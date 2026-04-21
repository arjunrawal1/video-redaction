// Post-process kept/added redaction boxes to collapse fragments that the
// curator picked at WORD granularity when a single LINE would do. This
// runs AFTER the Gemini tool loop finalizes, so the LLM can't undo it.
//
// Three shape-preserving rules, driven purely by geometry so text / semantic
// queries (where multiple visible matches legitimately share `instance_id`)
// remain untouched:
//
//   1. If every word of an OCR LINE is kept, emit a single box at that
//      LINE's pixel bbox. Equivalent outcome to the curator having picked
//      the LINE index directly.
//
//   2. If a subset of a line's words is kept AND those kept words are
//      spatially adjacent (on the same baseline with near-zero x-gap),
//      emit one box at the union of their bboxes, snapping to the LINE's
//      bbox when IoU is high enough to prove the merge matches a real
//      OCR line.
//
//   3. Non-adjacent kept words on the same line stay as multiple boxes —
//      that's genuinely discontiguous redaction and must not be merged
//      (would bleed over unredacted text).
//
// See conversation "prompt curator coalesce fragments" for the trigger
// case: frames 11–12 of the "redact all text content" run, where the
// curator called `keep_box` once per WORD instead of once per LINE,
// stacking two/three boxes over each cell.

import { extractRawOcrItems } from "@/lib/server/agentic-ocr";
import { bboxToPixels, type ServerBox } from "@/lib/server/openrouter";

// Pixel rectangle used for geometric union + IoU math.
type Rect = { x: number; y: number; w: number; h: number };

/**
 * Minimum intersection-over-union between the geometric union of merged
 * boxes and an OCR LINE bbox for us to consider the merge "equivalent"
 * to the LINE and snap to it. 0.6 is tolerant enough to handle the
 * 2 px padding `bboxToPixels` adds on every box, while still rejecting
 * accidental matches against a larger nearby LINE.
 */
const LINE_SNAP_IOU_THRESHOLD = 0.6;

/**
 * Max vertical-center distance between two boxes to treat them as on the
 * same text line, expressed as a fraction of the shorter box's height.
 * 0.4 covers cell-height jitter from OCR without pulling in the row
 * above or below.
 */
const SAME_LINE_CENTER_RATIO = 0.4;

/**
 * Max x-gap (in pixels of the shorter box's height) between two boxes
 * for us to call them "touching". 0.8 × height ≈ one space character at
 * typical OCR font sizes; wider gaps mean legitimately discontiguous
 * text the user may not want merged.
 */
const TOUCHING_GAP_RATIO = 0.8;

function sameLine(a: ServerBox, b: ServerBox): boolean {
  const minH = Math.min(a.h, b.h);
  if (minH <= 0) return false;
  const centerDy = Math.abs(a.y + a.h / 2 - (b.y + b.h / 2));
  return centerDy < minH * SAME_LINE_CENTER_RATIO;
}

function touching(a: ServerBox, b: ServerBox): boolean {
  const minH = Math.min(a.h, b.h);
  if (minH <= 0) return false;
  const gap = Math.max(a.x, b.x) - Math.min(a.x + a.w, b.x + b.w);
  return gap <= minH * TOUCHING_GAP_RATIO;
}

function unionRect(boxes: ServerBox[]): Rect {
  const x = Math.min(...boxes.map((b) => b.x));
  const y = Math.min(...boxes.map((b) => b.y));
  const right = Math.max(...boxes.map((b) => b.x + b.w));
  const bottom = Math.max(...boxes.map((b) => b.y + b.h));
  return { x, y, w: Math.max(1, right - x), h: Math.max(1, bottom - y) };
}

function iou(a: Rect, b: Rect): number {
  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(a.x + a.w, b.x + b.w);
  const iy2 = Math.min(a.y + a.h, b.y + b.h);
  const iw = ix2 - ix1;
  const ih = iy2 - iy1;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Build a union-find of `boxes`, merging any two boxes that are both on
 * the same baseline and near-touching in x. Returns groups as arrays of
 * the original box references.
 */
function buildAdjacencyGroups(boxes: ServerBox[]): ServerBox[][] {
  const n = boxes.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const union = (i: number, j: number): void => {
    const ri = find(i);
    const rj = find(j);
    if (ri !== rj) parent[ri] = rj;
  };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (sameLine(boxes[i], boxes[j]) && touching(boxes[i], boxes[j])) {
        union(i, j);
      }
    }
  }
  const buckets = new Map<number, ServerBox[]>();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    const arr = buckets.get(r);
    if (arr) arr.push(boxes[i]);
    else buckets.set(r, [boxes[i]]);
  }
  return [...buckets.values()];
}

type LineCandidate = {
  rect: Rect;
  text: string;
};

function precomputeLineRects(
  rawOcr: unknown,
  frameW: number,
  frameH: number,
): LineCandidate[] {
  const items = extractRawOcrItems(rawOcr);
  const out: LineCandidate[] = [];
  for (const it of items) {
    if (it.type !== "line") continue;
    const rect = bboxToPixels(it.bbox, frameW, frameH);
    if (rect.w <= 0 || rect.h <= 0) continue;
    out.push({ rect, text: it.text });
  }
  return out;
}

function findMatchingLine(
  target: Rect,
  lines: LineCandidate[],
): LineCandidate | null {
  let best: { iou: number; line: LineCandidate } | null = null;
  for (const line of lines) {
    const score = iou(target, line.rect);
    if (score >= LINE_SNAP_IOU_THRESHOLD && (!best || score > best.iou)) {
      best = { iou: score, line };
    }
  }
  return best?.line ?? null;
}

export type CoalesceStats = {
  /** Number of boxes in the input. */
  input: number;
  /** Number of boxes after coalescing. */
  output: number;
  /** Merges that snapped to an OCR LINE bbox. */
  snappedToLine: number;
  /** Merges that used the geometric union (no matching LINE). */
  unioned: number;
};

/**
 * Coalesce kept/added boxes. Boxes that are neither same-line nor
 * touching with any other box pass through unchanged. Adjacent runs on
 * the same line collapse into one box whose coords either match an OCR
 * LINE block (when the union matches a real line with IoU ≥ 0.6) or
 * fall back to the geometric union.
 *
 * The function is pure — input boxes are never mutated.
 */
export function coalesceAdjacentBoxes(args: {
  boxes: ServerBox[];
  rawOcr: unknown;
  frameWidth: number;
  frameHeight: number;
}): { boxes: ServerBox[]; stats: CoalesceStats } {
  const { boxes, rawOcr, frameWidth, frameHeight } = args;
  if (boxes.length <= 1) {
    return {
      boxes: boxes.slice(),
      stats: {
        input: boxes.length,
        output: boxes.length,
        snappedToLine: 0,
        unioned: 0,
      },
    };
  }
  const groups = buildAdjacencyGroups(boxes);
  const lines = precomputeLineRects(rawOcr, frameWidth, frameHeight);
  const out: ServerBox[] = [];
  let snappedToLine = 0;
  let unioned = 0;
  for (const group of groups) {
    if (group.length === 1) {
      out.push(group[0]);
      continue;
    }
    // Sort left-to-right for a readable merged `text` value.
    group.sort((a, b) => a.x - b.x || a.y - b.y);
    const target = unionRect(group);
    const line = findMatchingLine(target, lines);
    const rect = line ? line.rect : target;
    const text = line ? line.text : group.map((b) => b.text).join(" ");
    // Inherit metadata (colors, instance_id, branch, origin, track_id)
    // from the first group member — they were all kept for the "same"
    // redaction per the curator, so the metadata should match. We keep
    // origin so "fix" boxes stay "fix" after merging.
    const proto = group[0];
    out.push({
      ...proto,
      x: rect.x,
      y: rect.y,
      w: rect.w,
      h: rect.h,
      text,
      // Widen the bbox slightly by averaging scores of merged members;
      // preserves "pick the tightest" signals downstream.
      score:
        group.reduce((acc, b) => acc + (Number.isFinite(b.score) ? b.score : 0), 0) /
        group.length,
    });
    if (line) snappedToLine++;
    else unioned++;
  }
  return {
    boxes: out,
    stats: {
      input: boxes.length,
      output: out.length,
      snappedToLine,
      unioned,
    },
  };
}

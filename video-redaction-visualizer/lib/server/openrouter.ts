// OpenRouter-backed vision helpers used by the Gemini pipeline.
//
// Two public calls:
//   - detectFrame(jpeg, query, known)  → phase-1 per-frame detection + labeling
//   - compareFrames(ref, target, ...)  → phase-2/3 partial localization
//
// Both use Vercel AI SDK's generateObject with a Zod schema so the model's
// output is structurally validated. Bboxes come back in 0..1000 normalized
// coords and we convert to pixel-space Box via bboxToPixels.

import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { generateObject } from "ai";
import { z } from "zod";

export type ServerBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  text: string;
  score: number;
  label?: string;
  origin?: "backtrack" | "forward";
};

export type KnownLabel = {
  label: string;
  text: string;
  // Normalized [ymin, xmin, ymax, xmax] in 0..1000 so we can feed it back to
  // the model in the next prompt without leaking per-frame pixel coords.
  bbox: [number, number, number, number];
};

const _PAD_PX = 2;

// -- Schemas --------------------------------------------------------------
//
// These match Google's official bounding-box output shape for Gemini 2.5
// (see https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection):
// top-level JSON array of {box_2d, label} objects, with box_2d being
// [y_min, x_min, y_max, x_max] normalized to 0..1000. Diverging from this
// shape costs localization quality because the model was post-trained on
// exactly this schema. We add `text` alongside `label` so our redaction UI
// can carry both the identity letter and the visible text for the box.

const Box2D = z.tuple([z.number(), z.number(), z.number(), z.number()]);

const DetectItemSchema = z.object({
  box_2d: Box2D,
  label: z.string(),
  text: z.string(),
});

const CompareItemSchema = z.object({
  box_2d: Box2D,
  label: z.string(),
  text: z.string(),
});

// -- Client singleton -----------------------------------------------------

let _provider: ReturnType<typeof createOpenRouter> | null = null;

function getProvider(): ReturnType<typeof createOpenRouter> {
  if (_provider) return _provider;
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error(
      "OPENROUTER_API_KEY missing. Add it to video-redaction-visualizer/.env.local.",
    );
  }
  _provider = createOpenRouter({ apiKey });
  return _provider;
}

function modelSlug(): string {
  return process.env.OPENROUTER_MODEL || "google/gemini-2.5-flash";
}

export function openrouterConcurrency(): number {
  const raw = Number(process.env.OPENROUTER_CONCURRENCY || "4");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 4;
}

// -- Bbox conversion ------------------------------------------------------

/**
 * Convert a Gemini-style [ymin, xmin, ymax, xmax] in 0..1000 to pixel-space
 * Box {x, y, w, h} with the same ±2 px pad/clamp the OCR path applies so
 * overlays look consistent across engines.
 */
function bboxToPixels(
  bbox: [number, number, number, number],
  frameW: number,
  frameH: number,
): { x: number; y: number; w: number; h: number } {
  const [ymin, xmin, ymax, xmax] = bbox;
  const left = (xmin / 1000) * frameW;
  const top = (ymin / 1000) * frameH;
  const right = (xmax / 1000) * frameW;
  const bottom = (ymax / 1000) * frameH;
  const x = Math.max(0, Math.round(left) - _PAD_PX);
  const y = Math.max(0, Math.round(top) - _PAD_PX);
  const w = Math.min(frameW - x, Math.round(right - left) + 2 * _PAD_PX);
  const h = Math.min(frameH - y, Math.round(bottom - top) + 2 * _PAD_PX);
  return { x, y, w: Math.max(1, w), h: Math.max(1, h) };
}

function pixelBoxToNormalizedBbox(
  box: ServerBox,
  frameW: number,
  frameH: number,
): [number, number, number, number] {
  const ymin = Math.round((box.y / Math.max(1, frameH)) * 1000);
  const xmin = Math.round((box.x / Math.max(1, frameW)) * 1000);
  const ymax = Math.round(((box.y + box.h) / Math.max(1, frameH)) * 1000);
  const xmax = Math.round(((box.x + box.w) / Math.max(1, frameW)) * 1000);
  return [ymin, xmin, ymax, xmax];
}

// -- Prompts --------------------------------------------------------------

// Following Google's bounding-box-detection guidance we keep the system
// instruction short and declarative, and put the per-call task in the user
// message. The `box_2d` / array-of-objects format is implicit in the
// response schema and matches the model's trained output format, so we
// don't re-describe it in prose.

// Gemini's bbox training uses [y_min, x_min, y_max, x_max] normalized to
// 0..1000. When the model is reached through OpenRouter's proxy the trained
// schema association can drift (we've observed [x_min, y_min, ...] order in
// practice), so we pin the order explicitly in the system prompt. This is
// belt-and-suspenders with the `box_2d` schema name.
const _BOX_FORMAT_NOTE =
  "Each `box_2d` is [y_min, x_min, y_max, x_max] in integer coordinates " +
  "normalized to 0..1000 on the image, where (0, 0) is the top-left corner " +
  "and (1000, 1000) is the bottom-right. y_min < y_max and x_min < x_max.";

function phase1System(): string {
  return [
    "Return bounding boxes as a JSON array with labels.",
    "Never return masks.",
    _BOX_FORMAT_NOTE,
    "If a hit is present multiple times, give each instance a unique label.",
    "Exact matches, substrings, and fuzzy variants of the user's query all count.",
    "Reuse the label from a previously-reported hit when the same on-screen text appears at approximately the same position.",
    "Each array item must include `text` (the visible on-screen text) in addition to `label` and `box_2d`.",
  ].join("\n");
}

function phase2System(): string {
  return [
    "Return bounding boxes as a JSON array with labels.",
    "Never return masks.",
    _BOX_FORMAT_NOTE,
    "You are given two frames, REFERENCE and TARGET. Find the reference hit's on-screen text (or a partial/continuation of it) in the TARGET frame.",
    "Output at most one array element, describing the match in the TARGET frame. Return an empty array if the content is clearly absent.",
    "Prefer a tight `box_2d` that only covers the matched region, not the whole surrounding line.",
    "Each array item must include `text` (the visible on-screen text in the TARGET) in addition to `label` and `box_2d`.",
  ].join("\n");
}

// -- Utilities ------------------------------------------------------------

export function toLetterLabel(n: number): string {
  let out = "";
  let v = n + 1;
  while (v > 0) {
    const r = (v - 1) % 26;
    out = String.fromCharCode(65 + r) + out;
    v = Math.floor((v - 1) / 26);
  }
  return out;
}

// -- Public API -----------------------------------------------------------

/**
 * Phase-1: detect all hits in a single frame and label them consistently
 * with the running list of known labels from earlier frames.
 */
export async function detectFrame(
  jpeg: Uint8Array,
  query: string,
  known: KnownLabel[],
  frameW: number,
  frameH: number,
): Promise<{ boxes: ServerBox[]; raw: unknown }> {
  const provider = getProvider();
  const model = provider(modelSlug());

  const knownSummary =
    known.length === 0
      ? "(none so far)"
      : known
          .map((k) => `${k.label}: "${k.text}" at [${k.bbox.join(", ")}]`)
          .join("\n");

  const userText = [
    `Find every visible instance of the text query "${query}" in this frame.`,
    "Previously-labeled hits across earlier frames (reuse their label when the same on-screen text is present here at roughly the same position):",
    knownSummary,
    `Next unused label: ${toLetterLabel(known.length)}.`,
  ].join("\n");

  const result = await generateObject({
    model,
    output: "array",
    schema: DetectItemSchema,
    temperature: 0.5,
    system: phase1System(),
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: userText },
          { type: "image", image: jpeg, mediaType: "image/jpeg" },
        ],
      },
    ],
  });

  const items = (result.object ?? []) as Array<{
    box_2d: [number, number, number, number];
    label: string;
    text: string;
  }>;
  const boxes: ServerBox[] = items
    .map((h) => {
      const rect = bboxToPixels(h.box_2d, frameW, frameH);
      return {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
        text: h.text,
        score: 1.0,
        label: h.label,
      };
    })
    .filter((b) => b.w > 0 && b.h > 0);

  return { boxes, raw: items };
}

/**
 * Phase-2/3: given a known hit in the reference frame, look for the same or
 * a partial/continuation in the target frame. Returns null if the model says
 * it isn't there.
 */
export async function compareFrames(
  refJpeg: Uint8Array,
  targetJpeg: Uint8Array,
  query: string,
  refHit: ServerBox,
  refFrameW: number,
  refFrameH: number,
  targetFrameW: number,
  targetFrameH: number,
): Promise<{ box: ServerBox | null; raw: unknown }> {
  const provider = getProvider();
  const model = provider(modelSlug());

  const refBbox = pixelBoxToNormalizedBbox(refHit, refFrameW, refFrameH);
  const userText = [
    `Query: "${query}".`,
    `Reference hit: label=${refHit.label ?? "?"} text="${refHit.text}" box_2d=[${refBbox.join(", ")}] (on the REFERENCE frame).`,
    "Find the same on-screen text (or a visible partial/continuation of it) in the TARGET frame. Return an empty array if it is clearly absent.",
  ].join("\n");

  const result = await generateObject({
    model,
    output: "array",
    schema: CompareItemSchema,
    temperature: 0.5,
    system: phase2System(),
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "REFERENCE frame:" },
          { type: "image", image: refJpeg, mediaType: "image/jpeg" },
          { type: "text", text: "TARGET frame:" },
          { type: "image", image: targetJpeg, mediaType: "image/jpeg" },
          { type: "text", text: userText },
        ],
      },
    ],
  });

  const items = (result.object ?? []) as Array<{
    box_2d: [number, number, number, number];
    label: string;
    text: string;
  }>;
  const hit = items[0];
  if (!hit) {
    return { box: null, raw: items };
  }
  const rect = bboxToPixels(hit.box_2d, targetFrameW, targetFrameH);
  if (rect.w <= 0 || rect.h <= 0) {
    return { box: null, raw: items };
  }
  const box: ServerBox = {
    x: rect.x,
    y: rect.y,
    w: rect.w,
    h: rect.h,
    text: hit.text || refHit.text,
    score: 1.0,
    // Inherit the anchor's cross-frame label so chains stay consistent.
    label: refHit.label,
  };
  return { box, raw: items };
}

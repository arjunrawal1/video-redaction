// OpenRouter-backed vision helpers used by the Gemini pipeline.
//
// Two public calls:
//   - detectFrame(jpeg, query, known)  → phase-1 per-frame detection + labeling
//   - compareFrames(ref, target, ...)  → phase-2/3 partial localization
//
// Both use Vercel AI SDK's generateObject with a Zod schema so the model's
// output is structurally validated. Bboxes come back in 0..1000 normalized
// coords and we convert to pixel-space Box via bboxToPixels.

import {
  type GoogleGenerativeAIProviderOptions,
  createGoogleGenerativeAI,
  google,
} from "@ai-sdk/google";
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
  origin?: "backtrack" | "forward" | "fix";
  /**
   * Cross-frame identity assigned by the agentic linker (phase-1.5). Two
   * boxes share a `track_id` iff Gemini judged them to be the same
   * real-world redaction across consecutive scanned frames. The UI uses
   * this to interpolate (tween) box coordinates between keyframes.
   *
   * Minted server-side as `t{n}`; the client never generates these.
   * Undefined until the linker has run (e.g. still streaming phase-1).
   */
  track_id?: string;
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

export function getOpenRouter(): ReturnType<typeof createOpenRouter> {
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

function getProvider(): ReturnType<typeof createOpenRouter> {
  return getOpenRouter();
}

function modelSlug(): string {
  return process.env.OPENROUTER_MODEL || "google/gemini-2.5-flash";
}

// -- Google Gemini (agentic pipeline) -------------------------------------
//
// The agentic pipeline (curator + cascade + navigator) runs directly on
// Google's Gemini API, not through OpenRouter. This is so we can access
// Gemini-3-specific features that OpenRouter strips or doesn't expose:
//   - `thinking_level` (vs. generic reasoning_effort)
//   - `media_resolution_high` (explicit token budget per image — we need
//     the full 1120-token-per-image budget for dense OCR work)
//   - Code execution as a built-in tool — the model can write + run
//     Python to zoom, crop, and annotate images when it needs to ground
//     a small detail precisely.
//   - Automatic Thought Signature handling across tool turns (strictly
//     required for function-calling reasoning continuity).

let _google: ReturnType<typeof createGoogleGenerativeAI> | null = null;

export function getGoogleProvider(): ReturnType<typeof createGoogleGenerativeAI> {
  if (_google) return _google;
  const apiKey =
    process.env.GOOGLE_GENERATIVE_AI_API_KEY ||
    process.env.GEMINI_API_KEY ||
    process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GOOGLE_GENERATIVE_AI_API_KEY missing. Add it to video-redaction-visualizer/.env.local.",
    );
  }
  _google = createGoogleGenerativeAI({ apiKey });
  return _google;
}

/**
 * Model ID for every agentic-pipeline call (curator, cascade, navigator).
 * Defaults to Gemini 3 Flash, which gives Pro-level reasoning at Flash
 * pricing and supports both code execution and media-resolution control.
 */
export function agenticModelId(): string {
  return process.env.AGENTIC_MODEL || "gemini-3-flash-preview";
}

/**
 * Gemini 3 replaces the old `thinking_budget` token count with a
 * qualitative `thinking_level`. For redaction work we want deep reasoning
 * about partial-text visibility, so `high` (Gemini 3 Flash default) is a
 * good fit. Override via AGENTIC_THINKING_LEVEL for faster runs.
 */
export function agenticThinkingLevel(): "minimal" | "low" | "medium" | "high" {
  const raw = (process.env.AGENTIC_THINKING_LEVEL || "high").trim().toLowerCase();
  if (raw === "minimal" || raw === "low" || raw === "medium" || raw === "high") {
    return raw;
  }
  return "high";
}

/**
 * Per-image token budget. Gemini 3 defaults to an optimal value based on
 * media type, but for dense tabular-UI OCR the docs explicitly recommend
 * `MEDIA_RESOLUTION_HIGH` (1120 tokens / image) to preserve small text
 * legibility. Override if frames are unusually small or token cost is a
 * concern.
 */
export function agenticMediaResolution():
  | "MEDIA_RESOLUTION_LOW"
  | "MEDIA_RESOLUTION_MEDIUM"
  | "MEDIA_RESOLUTION_HIGH" {
  const raw = (process.env.AGENTIC_MEDIA_RESOLUTION || "MEDIA_RESOLUTION_HIGH")
    .trim()
    .toUpperCase();
  if (
    raw === "MEDIA_RESOLUTION_LOW" ||
    raw === "MEDIA_RESOLUTION_MEDIUM" ||
    raw === "MEDIA_RESOLUTION_HIGH"
  ) {
    return raw;
  }
  return "MEDIA_RESOLUTION_HIGH";
}

/**
 * When enabled, the cascade / navigator agents get Gemini's built-in
 * `code_execution` tool — letting them write + run Python to crop, zoom,
 * annotate, or measure images before placing a redaction box. Especially
 * useful for tiny text or ambiguous edges. Curator runs don't get this
 * (it's a pure structured-output task; code exec would just add latency).
 *
 * See: https://ai.google.dev/gemini-api/docs/gemini-3#code_execution_with_images
 */
export function agenticCodeExecutionEnabled(): boolean {
  const raw = (process.env.AGENTIC_CODE_EXECUTION || "true").trim().toLowerCase();
  return raw !== "false" && raw !== "0" && raw !== "off";
}

/**
 * Returns the LanguageModelV3 for all agentic calls. Cached via
 * `getGoogleProvider()`.
 */
export function agenticLanguageModel() {
  return getGoogleProvider()(agenticModelId());
}

/**
 * `providerOptions` object to spread into every agentic `generateText` /
 * `generateObject` call. Sets thinking level and media resolution.
 *
 * NOTE: Gemini 3 docs strongly recommend leaving `temperature` at its
 * default (1.0) — tuning it down can cause looping and degraded reasoning.
 * So we do NOT set temperature on these calls.
 */
export function agenticProviderOptions() {
  const google: GoogleGenerativeAIProviderOptions = {
    thinkingConfig: { thinkingLevel: agenticThinkingLevel() },
    mediaResolution: agenticMediaResolution(),
  };
  return { google } as const;
}

/**
 * Built-in tools to spread alongside user-defined tools for agents that
 * benefit from code execution (cascade + navigator). The tool name MUST
 * be `code_execution` — Gemini rejects other keys.
 */
export function agenticBuiltinTools(): Record<string, ReturnType<typeof google.tools.codeExecution>> {
  if (!agenticCodeExecutionEnabled()) return {};
  return { code_execution: google.tools.codeExecution({}) };
}

/**
 * Backward-compatible alias used by logging. Returns the model id instead
 * of an OpenRouter slug now that we're on Google direct.
 */
export function agenticModelSlug(): string {
  return agenticModelId();
}

export function openrouterConcurrency(): number {
  const raw = Number(process.env.OPENROUTER_CONCURRENCY || "4");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 4;
}

export function agenticCuratorConcurrency(): number {
  const raw = Number(process.env.AGENTIC_CURATOR_CONCURRENCY || "16");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 16;
}

/**
 * Model ID for the phase-1.5 linker (see lib/server/agentic-linker.ts).
 * The linker is a narrow structured-output task — no reasoning, no code
 * execution — so it benefits from a cheaper/faster tier when available.
 * Defaults to the same model as the curator so it Just Works without
 * extra env wiring; set AGENTIC_LINKER_MODEL=gemini-3.1-flash-lite-preview
 * to save ~half the tokens on identity bookkeeping.
 */
export function agenticLinkerModelId(): string {
  return process.env.AGENTIC_LINKER_MODEL || agenticModelId();
}

export function agenticLinkerLanguageModel() {
  return getGoogleProvider()(agenticLinkerModelId());
}

/**
 * Provider options tuned for the linker. Unlike the curator/navigator
 * we deliberately drop the thinking level — identity matching across two
 * frames is pattern-recognition, not deliberation — and drop media
 * resolution one notch because the model mostly needs layout context,
 * not pixel-perfect glyph reading (box texts are provided in-prompt).
 */
export function agenticLinkerProviderOptions() {
  const google: GoogleGenerativeAIProviderOptions = {
    thinkingConfig: { thinkingLevel: "low" },
    mediaResolution: "MEDIA_RESOLUTION_MEDIUM",
  };
  return { google } as const;
}

/**
 * How many adjacent-frame linker calls run in parallel. Each call is
 * ordering-dependent on the previous call's track-id output, so the
 * linker is driven as a *serial chain* today. Kept for future use if we
 * switch to a windowed approach.
 */
export function agenticLinkerConcurrency(): number {
  const raw = Number(process.env.AGENTIC_LINKER_CONCURRENCY || "1");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 1;
}

export function agenticNavigatorMaxSteps(): number {
  const raw = Number(process.env.AGENTIC_NAVIGATOR_MAX_STEPS || "250");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 250;
}

/**
 * Switches the navigator architecture:
 *   - "cascade" (default): one focused agent per phase-1 transition, each
 *     cascading forward frame-by-frame until no more changes needed.
 *   - "single": legacy free-roaming single-agent navigator (runNavigator).
 */
export function agenticNavMode(): "cascade" | "single" {
  const raw = (process.env.AGENTIC_NAV_MODE || "cascade").trim().toLowerCase();
  return raw === "single" ? "single" : "cascade";
}

/**
 * How many focused agents run in parallel across transitions. The initial
 * wave of transition agents is bounded by this; cascade children within
 * each chain run sequentially after their parent finishes.
 */
export function agenticCascadeConcurrency(): number {
  const raw = Number(process.env.AGENTIC_CASCADE_CONCURRENCY || "8");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 8;
}

/**
 * Max cascade chain length in a single direction from a seed anchor. A
 * chain stops when the agent reports `still_visible=false`, trips the
 * quiet-streak guard, runs into a frame already claimed by another chain
 * in the same direction, reaches end of range, OR this depth ceiling is
 * hit.
 */
export function agenticCascadeMaxDepth(): number {
  const raw = Number(process.env.AGENTIC_CASCADE_MAX_DEPTH || "20");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 20;
}

/**
 * Quiet-streak stop: when N consecutive agents in a chain each report
 * `still_visible=true` AND make zero modifications (no added / removed
 * boxes), we conclude the chain is stuck in "verify stable content" mode
 * and terminate. Transient-partial chains get reset on any modification,
 * so the guard doesn't interfere with the legit cascade use case.
 *
 * Default 2: aggressive but hugely effective on stable-content videos.
 * Bump to 3+ if you want the chain more patient before giving up.
 */
export function agenticCascadeQuietCap(): number {
  const raw = Number(process.env.AGENTIC_CASCADE_QUIET_CAP || "2");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 2;
}

/**
 * Total safety ceiling on how many focused agents a single navigate run
 * can spawn across all chain seeds combined. Guards against runaway
 * fan-out on pathological inputs. Bidirectional chains roughly double
 * the seed count vs. forward-only, so this is set generously.
 *
 * Raised from 120 → 300 now that ``findChainSeeds`` no longer dedups
 * transitions globally by ``(text, direction)``. The old dedup silently
 * swallowed legitimate re-investigations (e.g. same text hidden by a
 * popup later in the same video), so we let every transition seed its
 * own chain and rely on the per-direction ``frame_already_claimed``
 * stop to keep real-agent-count close to the number of distinct
 * transition frames. Accuracy > cost.
 */
export function agenticCascadeMaxAgents(): number {
  const raw = Number(process.env.AGENTIC_CASCADE_MAX_AGENTS || "300");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 300;
}

/** Per-focused-agent step ceiling (single focused agent is narrow, so this
 * is smaller than the free-roaming navigator's cap). */
export function agenticFocusedMaxSteps(): number {
  const raw = Number(process.env.AGENTIC_FOCUSED_MAX_STEPS || "20");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 20;
}

// -- Bbox conversion ------------------------------------------------------

/**
 * Convert a Gemini-style [ymin, xmin, ymax, xmax] in 0..1000 to pixel-space
 * Box {x, y, w, h} with the same ±2 px pad/clamp the OCR path applies so
 * overlays look consistent across engines.
 */
export function bboxToPixels(
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

export function pixelBoxToNormalizedBbox(
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
// 0..1000. We use this format everywhere — agentic pipeline tools,
// prompts, structured outputs, and debug panels — so the model never has
// to re-order / re-scale what it's seeing.
export const BOX_FORMAT_NOTE =
  "Each `box_2d` is [y_min, x_min, y_max, x_max] in integer coordinates " +
  "normalized to 0..1000 on the image, where (0, 0) is the top-left corner " +
  "and (1000, 1000) is the bottom-right. y_min < y_max and x_min < x_max.";
const _BOX_FORMAT_NOTE = BOX_FORMAT_NOTE;

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

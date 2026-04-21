// Vertex-AI-backed vision + agentic helpers.
//
// Two public vision calls (legacy "gemini" engine):
//   - detectFrame(jpeg, query, known)  → phase-1 per-frame detection + labeling
//   - compareFrames(ref, target, ...)  → phase-2/3 partial localization
//
// Both use the Vercel AI SDK's `generateObject` with a Zod schema so the
// model's output is structurally validated. Bboxes come back in 0..1000
// normalized coords and we convert to pixel-space Box via bboxToPixels.
//
// The agentic pipeline (curator, linker, cascade, navigator) is defined
// further down and runs the same Gemini models through the same Vertex
// provider instance so we share auth, quota, and billing with the rest
// of the stack.

import path from "node:path";
import { type GoogleGenerativeAIProviderOptions } from "@ai-sdk/google";
import { createVertex, vertex } from "@ai-sdk/google-vertex";
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
  // Prompt-mode only: stable per-instance identity and branch/category
  // metadata emitted by the predicate pipeline.
  instance_id?: string;
  branch?: string;
  category?: string | null;
  /**
   * Gemini-labeled dominant foreground (text glyph) color of the
   * redacted region, as a 6-digit hex RGB string (e.g. "#111111"). Set
   * by any path that emits a new box with color context — the agentic
   * curator, the cascade/navigator tool calls, and the prompt-mode
   * curator tools. The post-processing box-shrink pass uses this
   * together with `background_color_hex` to iteratively tighten box
   * sides whose two corners both fall on background pixels (see
   * `lib/server/box-shrink.ts`).
   *
   * Missing on code paths that don't have visual context (prompt-mode
   * deterministic fallback, text fast-path, linker fallback). Boxes
   * without both colors are passed through the shrink pass unchanged.
   */
  text_color_hex?: string;
  /** Dominant background (surrounding pixels) color hex. See `text_color_hex`. */
  background_color_hex?: string;
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

// -- Google Gemini via Vertex AI ------------------------------------------
//
// Every Gemini call — both the legacy per-frame detect/compare pipeline
// and the agentic pipeline (curator + linker + cascade + navigator) —
// goes through a single Vertex provider instance authenticated with a
// service-account key. Vertex gives us Gemini-3-specific features the
// AI Studio / OpenRouter edges don't expose:
//   - `thinkingLevel` (vs. generic reasoning_effort)
//   - `MEDIA_RESOLUTION_HIGH` (explicit 1120-tokens-per-image budget,
//     required for dense OCR work)
//   - `code_execution` built-in tool — the model writes + runs Python
//     to crop, zoom, and annotate images when it needs to ground a
//     small detail precisely.
//   - Automatic Thought Signature handling across tool turns (strictly
//     required for function-calling reasoning continuity).
//
// The AI SDK's Vertex provider wraps the same Gemini language-model
// interface as the AI Studio provider, so call sites don't need to
// care which transport is in use. Provider options are still nested
// under `google` in `providerOptions` (the Vertex provider shares the
// underlying Google Gemini language model implementation).

const VERTEX_PROJECT = process.env.GOOGLE_CLOUD_PROJECT
  || process.env.GOOGLE_VERTEX_PROJECT
  || "yarn-391421";
// `global` is the only Vertex location where `gemini-3-flash-preview` is
// served (per https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-flash),
// and `gemini-2.5-flash` / `gemini-2.5-pro` are also available on the
// global endpoint — so routing everything through `global` keeps both
// the vision and agentic pipelines on a single endpoint that gives
// higher availability than any single regional deployment. Override
// with GOOGLE_CLOUD_LOCATION if you specifically need data-residency
// pinning (but note regional endpoints don't serve Gemini 3 yet).
const VERTEX_LOCATION = process.env.GOOGLE_CLOUD_LOCATION
  || process.env.GOOGLE_VERTEX_LOCATION
  || "global";

/**
 * Absolute path to the Vertex service-account key JSON. Override via
 * `GOOGLE_APPLICATION_CREDENTIALS`; otherwise resolve to
 * `<repo-root>/vertexCreds/key.json` — which for our Next.js process
 * (cwd = `video-redaction-visualizer/`) is one directory up.
 */
function resolveVertexKeyFile(): string {
  const env = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  if (env && env.length > 0) return env;
  return path.resolve(process.cwd(), "..", "vertexCreds", "key.json");
}

let _google: ReturnType<typeof createVertex> | null = null;

/**
 * Workaround for vercel/ai#13911 (open as of @ai-sdk/google@3.0.64).
 *
 * When Gemini 3+ is combined with BOTH function tools and a
 * provider-defined tool (our case: `code_execution` alongside
 * `finish`/`add_box`/`remove_box`/...), `@ai-sdk/google`'s
 * `google-prepare-tools.ts` hard-codes
 * `tool_config.includeServerSideToolInvocations: true` on the outgoing
 * request.
 *
 * That flag is valid on the Google AI Studio edge
 * (`generativelanguage.googleapis.com`) but Vertex AI's endpoint
 * (`aiplatform.googleapis.com`) rejects it with
 * `400 Invalid JSON payload received. Unknown name "includeServerSideToolInvocations" at 'tool_config': Cannot find field.`
 *
 * Every cascade + navigator agent call was failing → 0 steps, 0 tokens,
 * $0.00. We intercept outgoing Vertex requests here, strip the field
 * from `tool_config` before forwarding, and let the rest of the payload
 * through unchanged. DELETE THIS WRAPPER once the SDK ships a fix.
 */
function stripVertexIncompatibleToolConfig(bodyText: string): string {
  try {
    const body = JSON.parse(bodyText);
    const cfg = (body as Record<string, unknown>)?.tool_config ??
      (body as Record<string, unknown>)?.toolConfig;
    if (cfg && typeof cfg === "object") {
      const c = cfg as Record<string, unknown>;
      if ("includeServerSideToolInvocations" in c) {
        delete c.includeServerSideToolInvocations;
      }
      if ("include_server_side_tool_invocations" in c) {
        delete c.include_server_side_tool_invocations;
      }
    }
    return JSON.stringify(body);
  } catch {
    return bodyText;
  }
}

/**
 * Workaround for vercel/ai#10344 / #11466 (still leaking on stable
 * `@ai-sdk/google@3.0.64` + `@ai-sdk/google-vertex@4.0.112` as of Apr 2026).
 *
 * Gemini 3 ("thinking") models attach an opaque `thoughtSignature` to every
 * `functionCall` content part they emit. On each replay turn, Vertex AI
 * rejects the request with:
 *
 *   400: Unable to submit request because function call `<name>`
 *        in the N. content block is missing a `thought_signature`.
 *
 * unless every replayed `functionCall` part carries its original signature.
 * The SDK fixed the common-case namespace-desync path in 3.0.27, but the
 * built-in `code_execution` provider-defined tool still slips through on
 * multi-step agent loops: the `code_execution` `functionCall` round-trips
 * without its signature and Vertex rejects turn N+1.
 *
 * This patch walks outgoing `contents[].parts[]` and, for any `model`-role
 * `functionCall` part missing a `thoughtSignature`, clones one from a
 * sibling part in the same turn (any neighbor that still has one — the
 * signature is per-reasoning-chunk, not per-call, and Vertex only validates
 * presence). Also handles the snake_case `thought_signature` spelling that
 * some SDK paths emit. Preserves every other field untouched.
 *
 * DELETE THIS WRAPPER once `@ai-sdk/google` ships the fix on the stable
 * (non-beta) channel.
 */
type PartLike = Record<string, unknown> & {
  functionCall?: unknown;
  function_call?: unknown;
  thoughtSignature?: string;
  thought_signature?: string;
};

function readThoughtSignature(part: PartLike): string | undefined {
  const a = part.thoughtSignature;
  if (typeof a === "string" && a.length > 0) return a;
  const b = part.thought_signature;
  if (typeof b === "string" && b.length > 0) return b;
  return undefined;
}

function hasFunctionCall(part: PartLike): boolean {
  return (
    (part.functionCall !== undefined && part.functionCall !== null) ||
    (part.function_call !== undefined && part.function_call !== null)
  );
}

function backfillThoughtSignatures(bodyText: string): string {
  try {
    const body = JSON.parse(bodyText) as Record<string, unknown>;
    const contents = body.contents;
    if (!Array.isArray(contents)) return bodyText;
    let mutated = false;
    for (const turn of contents) {
      if (!turn || typeof turn !== "object") continue;
      const t = turn as Record<string, unknown>;
      if (t.role !== "model") continue;
      const parts = t.parts;
      if (!Array.isArray(parts)) continue;
      // First pass: collect any signature present in this turn.
      let donor: string | undefined;
      for (const p of parts) {
        if (!p || typeof p !== "object") continue;
        const sig = readThoughtSignature(p as PartLike);
        if (sig) {
          donor = sig;
          break;
        }
      }
      if (!donor) continue;
      for (const p of parts) {
        if (!p || typeof p !== "object") continue;
        const part = p as PartLike;
        if (!hasFunctionCall(part)) continue;
        if (readThoughtSignature(part)) continue;
        part.thoughtSignature = donor;
        mutated = true;
      }
    }
    return mutated ? JSON.stringify(body) : bodyText;
  } catch {
    return bodyText;
  }
}

const vertexPatchedFetch: typeof fetch = async (input, init) => {
  if (init && typeof init.body === "string" && init.body.length > 0) {
    let patched = stripVertexIncompatibleToolConfig(init.body);
    patched = backfillThoughtSignatures(patched);
    if (patched !== init.body) {
      init = { ...init, body: patched };
    }
  }
  return fetch(input as RequestInfo, init);
};

export function getGoogleProvider(): ReturnType<typeof createVertex> {
  if (_google) return _google;
  const keyFile = resolveVertexKeyFile();
  _google = createVertex({
    project: VERTEX_PROJECT,
    location: VERTEX_LOCATION,
    googleAuthOptions: { keyFile },
    fetch: vertexPatchedFetch,
  });
  return _google;
}

/**
 * Model id used by the legacy per-frame detect / compare vision calls
 * (`detectFrame` / `compareFrames`). Defaults to Gemini 2.5 Flash — the
 * cheapest Vertex-hosted Gemini that still produces trustworthy 0..1000
 * bounding-box output. Override via `GEMINI_VISION_MODEL` (e.g. to
 * `gemini-2.5-pro` for tricky low-contrast text).
 */
function visionModelId(): string {
  return process.env.GEMINI_VISION_MODEL || "gemini-2.5-flash";
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
export function agenticBuiltinTools(): Record<string, ReturnType<typeof vertex.tools.codeExecution>> {
  if (!agenticCodeExecutionEnabled()) return {};
  return { code_execution: vertex.tools.codeExecution({}) };
}

/**
 * Backward-compatible alias used by logging. Returns the Vertex Gemini
 * model id; the `*Slug` name is a relic from when these calls went
 * through OpenRouter.
 */
export function agenticModelSlug(): string {
  return agenticModelId();
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
 * How many phase-1.5 linker Gemini calls run in parallel.
 *
 * Contrary to naïve intuition, linker model calls are completely
 * data-independent: each pair call only receives boxesA/boxesB and the
 * two frame JPEGs — it has no knowledge of track ids. Track ids are
 * minted serially AFTER all calls return, by walking the returned
 * index→index mappings in frame order ("stitch" step). So the model
 * calls can safely fan out; the only per-pair ordering requirement is
 * that the stitch phase sees every pair's decisions before threading
 * the chain, which `Promise.all` guarantees.
 *
 * Default 16 mirrors the curator's observed Gemini concurrency
 * ceiling. Bump if your provider rate limit allows more; set to 1 to
 * force fully-serial behavior for debugging or A/B comparisons.
 */
export function agenticLinkerConcurrency(): number {
  const raw = Number(process.env.AGENTIC_LINKER_CONCURRENCY || "16");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 16;
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

/**
 * Speculative cascade lookahead. Each chain runs up to this many
 * focused agents in parallel — the head of the chain plus N-1 speculated
 * descendants. When an agent returns `still_visible=false` or trips the
 * quiet-streak guard, every speculated agent past it is aborted and its
 * mutations are rolled back, so chain behavior is identical to the
 * sequential baseline — just faster when chains are long. Cost scales
 * linearly; set to 1 to disable speculation entirely.
 *
 * Default 4 is a safe sweet spot: long backward/forward chains (depth
 * ≥ 8 is common on scrolling content) get a ~4x wall-time reduction on
 * the critical path, while wasted work is capped because chains that
 * genuinely stop near their anchor only speculate a few frames ahead.
 */
export function agenticCascadeSpeculationDepth(): number {
  const raw = Number(process.env.AGENTIC_CASCADE_SPECULATION_DEPTH || "4");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 4;
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
  const model = getGoogleProvider()(visionModelId());

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
  const model = getGoogleProvider()(visionModelId());

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

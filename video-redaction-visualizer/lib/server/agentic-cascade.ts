// Cascade navigator: multi-agent, bi-directional content propagation.
//
// Architecture
// ------------
//   1. Walk every phase-1 transition (adjacent frame pair where the set of
//      hit texts changes). For every hit that *appeared* in the later
//      frame, spawn a BACKWARD chain to search earlier frames for partial
//      lead-ins. For every hit that *disappeared*, spawn a FORWARD chain
//      to search later frames for partial residuals. Seeds dedup by
//      (text, direction) so repeated cell content doesn't multi-launch.
//
//   2. Each chain propagates one frame at a time in its direction. The
//      focused agent at each step inspects the focus frame for ANY
//      visible partial of the tracked content (not just exact matches):
//      prefix, suffix, substring, edge-clipped, scrolled-out-of-view are
//      all sensitive and must be redacted.
//
//   3. The agent's `finish` tool now takes `still_visible: boolean`:
//
//        true  → tracked content (or any partial) is still visible here,
//                OR could plausibly continue into the next frame. The
//                cascade advances one step further.
//        false → tracked content is COMPLETELY gone on this frame (cell
//                cleared, text replaced, scrolled off-screen entirely).
//                Cascade stops.
//
//   4. Chains are bounded by AGENTIC_CASCADE_MAX_DEPTH per chain and
//      AGENTIC_CASCADE_MAX_AGENTS globally. Initial chains run in
//      parallel up to AGENTIC_CASCADE_CONCURRENCY; steps within a chain
//      run sequentially because each step reads the prior step's output.
//
// Why bi-directional?
//
//   The user's requirement: "if something we have currently redacted is
//   partially visible in any way in a previous or later frame, we need
//   to redact that." Forward-only propagation misses leading partials
//   (letters fading IN before the full text appears). Bidirectional with
//   proper "still visible" stop semantics catches both halves.
//
// Tool contract
//
//   - get_frame(frame_index)      → any frame in contextFrames (read)
//   - get_ocr_text(frame_index)   → any frame in contextFrames (read)
//   - add_box / remove_box /
//     adopt_ocr_box               → LOCKED to focusFrame (mutation)
//   - finish(summary, still_visible) → stops the focused agent and signals
//                                      the cascade whether to advance

import { generateText, stepCountIs, tool, type LanguageModelUsage } from "ai";
import { z } from "zod";
import { aerr, alog } from "./agentic-log";
import { addUsage, emptyUsage, geminiCost, type AggregateUsage } from "./cost";
import type { NavFrameState, NavHit, NavigatorEvent } from "./agentic-navigator";
import {
  extractRawOcrItems,
  filterAndRankOcrItems,
  formatOcrItemsForPrompt,
  type TextractBlock,
} from "./agentic-ocr";
import { resolveFixTrackId } from "./linker-fallback";
import {
  BOX_FORMAT_NOTE,
  agenticBuiltinTools,
  agenticCascadeConcurrency,
  agenticCascadeMaxAgents,
  agenticCascadeMaxDepth,
  agenticCascadeQuietCap,
  agenticCascadeSpeculationDepth,
  agenticCodeExecutionEnabled,
  agenticFocusedMaxSteps,
  agenticLanguageModel,
  agenticModelId,
  agenticProviderOptions,
  bboxToPixels,
  pixelBoxToNormalizedBbox,
} from "./openrouter";
import { compact, type RunLog } from "./run-log";

// -- Tracked content + chain seeds -----------------------------------------

export type TrackedHit = {
  /** Visible on-screen text of the hit we're propagating. */
  text: string;
  /**
   * Frame where the hit was fully visible (the "anchor"). The cascade
   * walks outward from this frame. For backward chains this is a frame
   * where the hit newly appeared; for forward chains it's the last frame
   * where the hit existed before vanishing.
   */
  source_frame: number;
  /** Anchor-frame bbox in Gemini's box_2d format: [y_min, x_min, y_max, x_max] in 0..1000. Gives the agent a spatial prior. */
  source_bbox: [number, number, number, number];
};

export type ChainSeed = {
  tracked: TrackedHit;
  direction: "backward" | "forward";
  /** First frame the cascade will inspect. */
  first_focus: number;
};

/**
 * Build chain seeds from phase-1 hits by walking every adjacent-frame pair
 * and pulling out the hits that appeared / disappeared.
 *
 * Every genuine transition gets its own seed — we deliberately do NOT
 * dedup by ``(text, direction)`` across the run. Distinct transitions of
 * the same text are distinct events with distinct investigations. Two
 * concrete examples this run surfaced:
 *
 *   1. ``test wor`` scrolls out of view at transition 13→14 (intro
 *      horizontal scroll) and the cascade should chase forward from f14
 *      to find the fading-edge partials.
 *   2. Much later, ``test wor`` is covered by a font-picker popup at
 *      52→53 and the cascade should chase forward from f53 to find the
 *      sliver of "or" still visible to the right of the popup.
 *
 * A global ``(text, direction)`` dedup silently swallows case 2 — the
 * key is already claimed by case 1 — so the popup-occluded sliver never
 * gets investigated and leaks through to export. Duplicate seeds that
 * target an already-inspected frame are already cheaply suppressed at
 * runtime by the ``frame_already_claimed`` chain-stop (per-direction
 * `claimed` set in ``runCascadeNavigator``), so removing this upstream
 * dedup costs effectively nothing but gains correctness on any video
 * where the same text disappears/reappears for different reasons.
 */
export function findChainSeeds(frames: NavFrameState[]): ChainSeed[] {
  const seeds: ChainSeed[] = [];

  const toTracked = (h: NavHit, f: NavFrameState): TrackedHit => ({
    text: h.text,
    source_frame: f.index,
    source_bbox: pixelBoxToNormalizedBbox(h, f.width, f.height),
  });

  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1];
    const cur = frames[i];
    const prevTexts = new Set(
      prev.hits.map((h) => h.text.trim().toLowerCase()),
    );
    const curTexts = new Set(cur.hits.map((h) => h.text.trim().toLowerCase()));

    // Appearances — cascade BACKWARD from cur-1 to find lead-in partials.
    for (const h of cur.hits) {
      const t = h.text.trim().toLowerCase();
      if (prevTexts.has(t)) continue;
      seeds.push({
        tracked: toTracked(h, cur),
        direction: "backward",
        first_focus: cur.index - 1,
      });
    }

    // Disappearances — cascade FORWARD from cur to find residual partials.
    for (const h of prev.hits) {
      const t = h.text.trim().toLowerCase();
      if (curTexts.has(t)) continue;
      seeds.push({
        tracked: toTracked(h, prev),
        direction: "forward",
        first_focus: cur.index,
      });
    }
  }
  return seeds;
}

// -- Hit-id minting ---------------------------------------------------------

const HIT_ID_PREFIX = "C";
let _hitCounter = 0;
function mintHitId(): string {
  _hitCounter = (_hitCounter + 1) % 1_000_000;
  return `${HIT_ID_PREFIX}${Date.now().toString(36)}${_hitCounter.toString(36)}`;
}

// -- Focused agent ----------------------------------------------------------

type FocusedAgentOpts = {
  agentId: string;
  focusFrame: number;
  contextFrames: NavFrameState[];
  query: string;
  source: "transition" | "cascade";
  parentAgentId: string | null;
  tracked: TrackedHit;
  direction: "backward" | "forward";
  depthFromSource: number;
  spawnReason: string;
  onEvent: (ev: NavigatorEvent) => void;
  abortSignal?: AbortSignal;
  runLog?: RunLog | null;
};

type FocusedAgentResult = {
  agentId: string;
  focusFrame: number;
  added: number;
  removed: number;
  totalSteps: number;
  finishSummary: string | null;
  /**
   * Agent's judgement on whether the tracked content is still visible on
   * the focus frame (in any form). `null` if the agent never called
   * `finish` — treated as "still visible" so the cascade is conservative.
   */
  stillVisible: boolean | null;
  /** AI SDK token usage summed across this agent's steps. */
  usage: LanguageModelUsage | null;
  /**
   * Undoes every mutation this agent made to the shared frame state
   * (hits added, hits removed). Called by the speculative cascade
   * orchestrator when a parent agent turns out to have stopped the
   * chain, making this agent's work speculative waste.
   *
   * Also emits counter `frame_update` events so the UI can mirror the
   * rollback, and fires an `agent_rolled_back` log event for
   * observability. Idempotent.
   */
  rollback: () => void;
  /** True once rollback has run. */
  isRolledBack: () => boolean;
};

async function runFocusedAgent(opts: FocusedAgentOpts): Promise<FocusedAgentResult> {
  const {
    agentId,
    focusFrame,
    contextFrames,
    query,
    source,
    parentAgentId,
    tracked,
    direction,
    depthFromSource,
    spawnReason,
    onEvent,
    abortSignal,
    runLog,
  } = opts;

  const frameByIndex = new Map<number, NavFrameState>();
  for (const f of contextFrames) frameByIndex.set(f.index, f);
  const focus = frameByIndex.get(focusFrame);
  if (!focus) {
    throw new Error(
      `runFocusedAgent: focusFrame ${focusFrame} missing from contextFrames`,
    );
  }

  let added = 0;
  let removed = 0;
  let finishSummary: string | null = null;
  let stillVisible: boolean | null = null;
  let steps = 0;

  // Rollback bookkeeping. Every successful add_box / adopt_ocr_box /
  // remove_box records a delta so the speculative cascade orchestrator
  // can undo this agent's mutations if it turns out a parent stopped
  // the chain before this agent should have run. Kept in chronological
  // order; rollback() replays in reverse.
  type Delta =
    | { kind: "add"; hit: NavHit }
    | { kind: "remove"; hit: NavHit };
  const deltas: Delta[] = [];
  let rolledBack = false;

  onEvent({
    type: "agent_start",
    agent_id: agentId,
    focus_frame: focusFrame,
    source,
    parent_agent_id: parentAgentId,
    reason: spawnReason,
  });

  const bumpStep = () => {
    steps += 1;
    return steps;
  };

  const enforceFocus = (
    frame_index: number,
  ): { ok: false; error: string } | null => {
    if (frame_index !== focusFrame) {
      return {
        ok: false as const,
        error: `This focused agent can only modify frame ${focusFrame}. You asked to modify frame ${frame_index}. Use get_frame/get_ocr_text for cross-frame reads; spawn a new agent chain for other frames.`,
      };
    }
    return null;
  };

  // Resolve a track_id for a newly-added fix box on the focus frame.
  // Looks at immediate neighbors (both in the agent's context window) to
  // see if the new box is a continuation of an already-tracked
  // redaction; if not, mints a fresh id keyed off hit_id. Called from
  // add_box and adopt_ocr_box right before the hit is committed so the
  // UI / exporter see track_ids on fix boxes and can interpolate them
  // like phase-1.5-linked boxes.
  const trackIdForFix = (hit: NavHit): string => {
    const prev = frameByIndex.get(focusFrame - 1);
    const next = frameByIndex.get(focusFrame + 1);
    const claimed = new Set<string>();
    for (const h of focus.hits) {
      if (h.track_id) claimed.add(h.track_id);
    }
    return resolveFixTrackId({
      agentId,
      frameIndex: focusFrame,
      hitId: hit.hit_id,
      newBox: hit,
      newFrameWidth: focus.width,
      newFrameHeight: focus.height,
      prevFrame: prev
        ? { hits: prev.hits, width: prev.width, height: prev.height }
        : null,
      nextFrame: next
        ? { hits: next.hits, width: next.width, height: next.height }
        : null,
      claimedOnCurrentFrame: claimed,
    });
  };

  // ---- Tools -----------------------------------------------------------

  // Build a compact summary (hits + pre-filtered OCR candidates) for a
  // single frame. Shared between the initial user-prompt inlining and
  // the get_frame tool so the model sees an identical shape either way.
  const buildFrameSummary = (f: NavFrameState) => ({
    frame_index: f.index,
    width: f.width,
    height: f.height,
    current_hits: f.hits.map((h) => ({
      hit_id: h.hit_id,
      text: h.text,
      box_2d: pixelBoxToNormalizedBbox(h, f.width, f.height),
    })),
    ocr_candidates: f.ocrBoxes.map((b, i) => ({
      ocr_index: i,
      text: b.text,
      box_2d: pixelBoxToNormalizedBbox(b, f.width, f.height),
    })),
  });

  // Multi-frame: accepts an array of frame indices and returns the
  // state + JPEG for each in a single tool call. Saves one full LLM
  // round-trip per extra frame the agent would otherwise have fetched
  // sequentially. The legacy single-frame shape (`frame_index: number`)
  // is still accepted for backward compatibility with prompt variants
  // that use it; the schema normalizes both into an array.
  const getFrameTool = tool({
    description:
      "Fetch frame images + current hits + pre-filtered OCR candidates. PREFER `frame_indices` (array) to fetch multiple frames in a SINGLE call — serializing one-at-a-time fetches is wasteful. Note: your FOCUS frame's image and OCR are already in this conversation's initial user message; only call this to (re)inspect it at zoom, or to inspect neighbors in your context window.",
    inputSchema: z.object({
      frame_indices: z.array(z.number().int()).nullable(),
      frame_index: z.number().int().nullable(),
    }),
    execute: async ({ frame_indices, frame_index }) => {
      const requested: number[] = [];
      if (frame_indices && frame_indices.length > 0) {
        requested.push(...frame_indices);
      } else if (typeof frame_index === "number") {
        requested.push(frame_index);
      } else {
        return {
          ok: false as const,
          error:
            "Provide `frame_indices` (array) or `frame_index` (number).",
        };
      }
      alog(`[${agentId}] tool get_frame invoked`, { frame_indices: requested });
      const results: Array<{
        frame_index: number;
        ok: boolean;
        error?: string;
        summary?: ReturnType<typeof buildFrameSummary>;
        image_base64?: string;
      }> = [];
      for (const idx of requested) {
        const f = frameByIndex.get(idx);
        if (!f) {
          results.push({
            frame_index: idx,
            ok: false,
            error: `Frame ${idx} is not in this agent's context window.`,
          });
          continue;
        }
        results.push({
          frame_index: idx,
          ok: true,
          summary: buildFrameSummary(f),
          image_base64: Buffer.from(f.blob).toString("base64"),
        });
      }
      return { ok: true as const, results };
    },
    toModelOutput: ({ output }) => {
      if (!output.ok) {
        return { type: "error-text", value: output.error };
      }
      const parts: Array<
        | { type: "text"; text: string }
        | { type: "image-data"; mediaType: string; data: string }
      > = [];
      for (const r of output.results) {
        if (!r.ok) {
          parts.push({
            type: "text",
            text: `Frame ${r.frame_index}: ERROR ${r.error ?? "unknown"}`,
          });
          continue;
        }
        parts.push({
          type: "text",
          text: `Frame ${r.frame_index}:\n${JSON.stringify(r.summary, null, 2)}`,
        });
        if (r.image_base64) {
          parts.push({
            type: "image-data",
            mediaType: "image/jpeg",
            data: r.image_base64,
          });
        }
      }
      return { type: "content", value: parts };
    },
  });

  // Multi-frame, multi-filter OCR dump. In one call the agent can ask
  // "give me every WORD/LINE on frames 2, 3, 4 that contains ANY of
  // ['ord','wor','est','test']" and get filter-tagged items back. This
  // replaces the extremely common pattern of issuing 5-15 sequential
  // single-filter calls on the same frame, each of which cost one full
  // LLM round-trip.
  const getOcrTextTool = tool({
    description:
      "Fetch RAW OCR detections (every Textract WORD and LINE with exact bboxes) for ONE OR MORE frames, optionally filtered by substrings. PREFER passing ALL substrings you care about as a single `filters` array in ONE call — the tool returns every matching block tagged with which filter(s) hit, so you can explore many fragments in a single round-trip instead of serializing calls. Note: your FOCUS frame's OCR dump is already in this conversation's initial user message; only call this when you need neighbor-frame OCR or additional filters.",
    inputSchema: z.object({
      frame_indices: z.array(z.number().int()).nullable(),
      frame_index: z.number().int().nullable(),
      filters: z.array(z.string()).nullable(),
      filter: z.string().nullable(),
      block_types: z.array(z.enum(["WORD", "LINE"])).nullable(),
    }),
    execute: async ({
      frame_indices,
      frame_index,
      filters,
      filter,
      block_types,
    }) => {
      const requested: number[] = [];
      if (frame_indices && frame_indices.length > 0) {
        requested.push(...frame_indices);
      } else if (typeof frame_index === "number") {
        requested.push(frame_index);
      } else {
        return {
          ok: false as const,
          error: "Provide `frame_indices` (array) or `frame_index` (number).",
        };
      }
      const needles: string[] = [];
      if (filters && filters.length > 0) {
        for (const f of filters) if (f) needles.push(f);
      } else if (filter) {
        needles.push(filter);
      }
      const needlesLower = needles.map((n) => n.toLowerCase());
      alog(`[${agentId}] tool get_ocr_text invoked`, {
        frame_indices: requested,
        filters: needles,
        block_types,
      });
      const wantWord = !block_types || block_types.includes("WORD");
      const wantLine = !block_types || block_types.includes("LINE");

      const frames: Array<{
        frame_index: number;
        ok: boolean;
        error?: string;
        items?: Array<{
          type: "word" | "line";
          text: string;
          confidence: number;
          box_2d: [number, number, number, number];
          matched_filters: string[];
        }>;
      }> = [];
      for (const idx of requested) {
        const f = frameByIndex.get(idx);
        if (!f) {
          frames.push({
            frame_index: idx,
            ok: false,
            error: `Frame ${idx} is not in this agent's context window.`,
          });
          continue;
        }
        const raw = f.ocrRaw as { Blocks?: TextractBlock[] } | null;
        const blocks = raw?.Blocks ?? [];
        const items: Array<{
          type: "word" | "line";
          text: string;
          confidence: number;
          box_2d: [number, number, number, number];
          matched_filters: string[];
        }> = [];
        for (const b of blocks) {
          const btype = b.BlockType;
          if (btype !== "WORD" && btype !== "LINE") continue;
          if (btype === "WORD" && !wantWord) continue;
          if (btype === "LINE" && !wantLine) continue;
          const text = String(b.Text ?? "");
          let matched: string[] = [];
          if (needlesLower.length > 0) {
            const lower = text.toLowerCase();
            for (let i = 0; i < needlesLower.length; i++) {
              if (lower.includes(needlesLower[i])) matched.push(needles[i]);
            }
            if (matched.length === 0) continue;
          } else {
            matched = [];
          }
          const bbox = b.Geometry?.BoundingBox;
          if (!bbox) continue;
          const left = Number(bbox.Left ?? 0);
          const top = Number(bbox.Top ?? 0);
          const w = Number(bbox.Width ?? 0);
          const h = Number(bbox.Height ?? 0);
          // Gemini native box_2d: [y_min, x_min, y_max, x_max] in 0..1000.
          items.push({
            type: btype === "WORD" ? "word" : "line",
            text,
            confidence: Math.round(Number(b.Confidence ?? 0) * 10) / 10,
            box_2d: [
              Math.round(top * 1000),
              Math.round(left * 1000),
              Math.round((top + h) * 1000),
              Math.round((left + w) * 1000),
            ],
            matched_filters: matched,
          });
        }
        frames.push({ frame_index: idx, ok: true, items });
      }
      return {
        ok: true as const,
        filters: needles,
        frames,
      };
    },
  });

  const addBoxTool = tool({
    description: `Add a new redaction box on frame ${focusFrame} (THIS agent's focus frame). Coords are Gemini's native box_2d format: [y_min, x_min, y_max, x_max] in 0..1000 (top-left origin). STRICT coord-source priority: (1) copy from a get_ocr_text WORD/LINE block on this frame; (2) measure from a code_execution crop of this frame; (3) pure estimation ONLY as last resort. DO NOT copy the anchor frame's bbox and DO NOT invent coords from UI semantics ("formula bar should show…", "selected cell F27 would be here…"). The \`reason\` field must name the tier and the evidence (e.g. "Tier 1: OCR WORD 'or' at [y,x,y,x]=..." or "Tier 2: code_execution crop (x,y,w,h)=...").`,
    inputSchema: z.object({
      frame_index: z.number().int(),
      text: z.string(),
      y_min: z.number().int().min(0).max(1000),
      x_min: z.number().int().min(0).max(1000),
      y_max: z.number().int().min(0).max(1000),
      x_max: z.number().int().min(0).max(1000),
      reason: z.string().optional(),
    }),
    execute: async ({ frame_index, text, y_min, x_min, y_max, x_max, reason }) => {
      alog(`[${agentId}] tool add_box invoked`, {
        frame_index,
        text,
        box_2d: [y_min, x_min, y_max, x_max],
        reason,
      });
      const guard = enforceFocus(frame_index);
      if (guard) return guard;
      const rect = bboxToPixels(
        [y_min, x_min, y_max, x_max],
        focus.width,
        focus.height,
      );
      if (rect.w <= 0 || rect.h <= 0) {
        return { ok: false as const, error: "Degenerate box dimensions." };
      }
      const hit: NavHit = {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
        text,
        score: 1.0,
        origin: "fix",
        hit_id: mintHitId(),
        reason: reason ?? undefined,
      };
      hit.track_id = trackIdForFix(hit);
      focus.hits.push(hit);
      deltas.push({ kind: "add", hit });
      added += 1;
      onEvent({
        type: "frame_update",
        index: focusFrame,
        action: "add",
        hit,
        reason,
        agent_id: agentId,
      });
      return {
        ok: true as const,
        hit_id: hit.hit_id,
        track_id: hit.track_id,
        pixel_box: { x: hit.x, y: hit.y, w: hit.w, h: hit.h },
      };
    },
  });

  const removeBoxTool = tool({
    description: `Remove a false-positive hit on frame ${focusFrame} (THIS agent's focus frame). Very rare in cascade mode — only use if you are certain the hit is wrong, and explain in \`reason\`.`,
    inputSchema: z.object({
      frame_index: z.number().int(),
      hit_id: z.string(),
      reason: z.string().optional(),
    }),
    execute: async ({ frame_index, hit_id, reason }) => {
      alog(`[${agentId}] tool remove_box invoked`, { frame_index, hit_id, reason });
      const guard = enforceFocus(frame_index);
      if (guard) return guard;
      const idx = focus.hits.findIndex((h) => h.hit_id === hit_id);
      if (idx < 0) {
        return { ok: false as const, error: `hit_id ${hit_id} not on frame ${focusFrame}` };
      }
      const [removedHit] = focus.hits.splice(idx, 1);
      deltas.push({ kind: "remove", hit: removedHit });
      removed += 1;
      onEvent({
        type: "frame_update",
        index: focusFrame,
        action: "remove",
        hit: removedHit,
        reason,
        agent_id: agentId,
      });
      return { ok: true as const, removed_hit_id: hit_id };
    },
  });

  const adoptOcrTool = tool({
    description: `Promote an OCR candidate rectangle on frame ${focusFrame} (THIS agent's focus frame) into a real hit. ocr_index refers to the pre-filtered candidate list returned by get_frame on the focus frame.`,
    inputSchema: z.object({
      frame_index: z.number().int(),
      ocr_index: z.number().int(),
      text: z.string().optional(),
      reason: z.string().optional(),
    }),
    execute: async ({ frame_index, ocr_index, text, reason }) => {
      alog(`[${agentId}] tool adopt_ocr_box invoked`, {
        frame_index,
        ocr_index,
        text,
        reason,
      });
      const guard = enforceFocus(frame_index);
      if (guard) return guard;
      const src = focus.ocrBoxes[ocr_index];
      if (!src) {
        return { ok: false as const, error: `No OCR candidate at index ${ocr_index}` };
      }
      const hit: NavHit = {
        x: src.x,
        y: src.y,
        w: src.w,
        h: src.h,
        text: text || src.text,
        score: src.score ?? 1.0,
        origin: "fix",
        hit_id: mintHitId(),
        reason: reason ?? undefined,
      };
      hit.track_id = trackIdForFix(hit);
      focus.hits.push(hit);
      deltas.push({ kind: "add", hit });
      added += 1;
      onEvent({
        type: "frame_update",
        index: focusFrame,
        action: "add",
        hit,
        reason,
        agent_id: agentId,
      });
      return {
        ok: true as const,
        hit_id: hit.hit_id,
        track_id: hit.track_id,
        pixel_box: { x: hit.x, y: hit.y, w: hit.w, h: hit.h },
      };
    },
  });

  const finishTool = tool({
    description:
      "End this focused-agent turn. REQUIRED to set `still_visible`: true if the tracked content (in any form — full, partial, prefix, suffix, substring, clipped, OCR-garbled, case-variant) is still visible on the focus frame OR could plausibly continue into the next frame in the cascade direction; false ONLY when the tracked content is COMPLETELY absent (cell cleared, text replaced, scrolled entirely off-screen). When uncertain, prefer `true` — cascade will safely check one more frame. After calling finish, do not issue any more tool calls.",
    inputSchema: z.object({
      summary: z.string(),
      still_visible: z.boolean(),
    }),
    execute: async ({ summary, still_visible }) => {
      alog(`[${agentId}] tool finish invoked`, { summary, still_visible });
      finishSummary = summary;
      stillVisible = still_visible;
      return { ok: true as const, acknowledged: true };
    },
  });

  // Gemini's built-in code-execution tool is included alongside the
  // user-defined tools when AGENTIC_CODE_EXECUTION is enabled. The model
  // can write + run Python to crop, zoom, or annotate the focus frame
  // when it needs higher visual resolution than a single glance. The
  // name must be `code_execution` — Gemini rejects other keys.
  const tools = {
    get_frame: getFrameTool,
    get_ocr_text: getOcrTextTool,
    add_box: addBoxTool,
    remove_box: removeBoxTool,
    adopt_ocr_box: adoptOcrTool,
    finish: finishTool,
    ...agenticBuiltinTools(),
  };

  // ---- Initial prompt --------------------------------------------------

  const focusSummary = buildFrameSummary(focus);

  // Pre-compute the focus frame's full raw-OCR dump (WORDs + LINEs in
  // Gemini's native box_2d format) and inline it in the initial user
  // message. This is the biggest latency win on the cascade path: pre-
  // optimization every agent was spending its first 2–5 LLM turns doing
  // `get_frame` + repeated `get_ocr_text` on its own focus frame before
  // it could even begin reasoning about the tracked content.
  const focusRawItems = extractRawOcrItems(focus.ocrRaw);
  const { relevant: focusRelevantRaw, context: focusContextRaw } =
    filterAndRankOcrItems(focusRawItems, tracked.text);
  const focusRelevantBlock = formatOcrItemsForPrompt(focusRelevantRaw);
  const focusContextBlock = formatOcrItemsForPrompt(focusContextRaw);

  const directionDescription =
    direction === "backward"
      ? "BACKWARD (earlier frames, hunting for lead-in partials that precede the full appearance of the tracked content)"
      : "FORWARD (later frames, hunting for residual / trailing partials that linger after the tracked content started to disappear)";

  const neighborIndices = contextFrames
    .map((f) => f.index)
    .filter((i) => i !== focusFrame);

  const systemPrompt = [
    "You are a FOCUSED TRACKING agent responsible for a single frame in a cascade.",
    `Focus frame: #${focusFrame}. You may ONLY add / remove / adopt boxes on this frame. Attempts to mutate other frames will be rejected.`,
    "",
    "TRACKED CONTENT",
    `You are tracking: "${tracked.text}"`,
    `Anchor: fully visible on frame #${tracked.source_frame} at box_2d=[${tracked.source_bbox.join(", ")}] (Gemini native format: [y_min, x_min, y_max, x_max] in 0..1000).`,
    `Cascade direction: ${directionDescription}.`,
    `Depth from anchor: ${depthFromSource} (distance in frames).`,
    "",
    "WHAT IS ALREADY IN YOUR CONTEXT",
    "The initial user message already contains, for the focus frame:",
    "  - the JPEG image (attached as an image part)",
    "  - the current hits and pre-filtered OCR candidates",
    "  - the FULL raw Textract OCR dump (WORD + LINE blocks with pixel-accurate box_2d coords), split into query-relevant and context blocks",
    "DO NOT issue `get_frame` or `get_ocr_text` calls for the focus frame unless you genuinely need to zoom into a crop or apply a specific substring filter not already covered by the inlined dump. Those extra calls are pure round-trip latency.",
    neighborIndices.length > 0
      ? `Neighbor frames in your context window: ${neighborIndices.join(", ")}. Fetch them with get_frame/get_ocr_text ONLY if you need cross-frame disambiguation — and when you do, pass ALL the indices and substrings you care about as arrays in a SINGLE tool call.`
      : "No neighbor frames are in your context window.",
    "",
    "PARALLEL TOOL CALLS — USE THEM",
    "Both get_frame and get_ocr_text accept arrays (frame_indices, filters). When you need to look up several substrings or peek at several neighbor frames, pass them ALL in one call instead of serializing one-per-turn. Each extra turn costs a full LLM round-trip for no new information.",
    "If in a single turn you need to inspect multiple INDEPENDENT things (e.g. OCR on two different frames with different block_types), issue ALL of those tool calls in the SAME turn — the runtime will execute them in parallel.",
    "",
    "REDACTION SEMANTICS — partial visibility is a leak",
    "ANY of the following on the focus frame counts as the tracked content still being visible and must be redacted:",
    "  - full match of the tracked text",
    "  - case variants (uppercase/lowercase)",
    "  - substring / prefix / suffix (EVEN A SINGLE LETTER if it's the part of the tracked text that's fading in/out at the edge of a cell or the viewport)",
    "  - OCR-garbled forms (e.g. 'test w0rd', 'tes t word', 'testword')",
    "  - spatially-shifted instances (scrolling moved it but it's still on-screen)",
    "Cross-frame reassembly attack: if frame N shows `test wor` and frame N+1 shows `est word`, the viewer reconstructs the full sensitive text. Redact BOTH.",
    "",
    "YOUR PROCESS",
    "1. Read the inlined focus-frame OCR dump + image in the first user message. That is already everything you need for most frames.",
    `2. Only if the inlined dump does not contain a block you suspect exists on the focus frame — e.g. you want to try a substring filter not covered in the pre-filtered "relevant raw blocks" above — call get_ocr_text({frame_indices:[${focusFrame}], filters:[<substring>, <substring>, ...]}) with ALL substrings in one call.`,
    "",
    "COORDINATE SOURCE — STRICT PRIORITY ORDER (do NOT skip tiers)",
    "Tier 1 — OCR coords (REQUIRED FIRST ATTEMPT).",
    "   Every add_box should copy its bbox from a Textract WORD or LINE block returned by get_ocr_text on THIS frame (or from the inlined dump). Prefer the narrowest block that contains the visible fragment (e.g. the standalone 'or' WORD, not the anchor's full 'test wor' width). Use adopt_ocr_box when the block is in the query-filtered candidate list.",
    agenticCodeExecutionEnabled()
      ? "Tier 2 — Visual reasoning with code_execution (use ONLY if no OCR block covers the visible fragment).\n   Write Python that crops & zooms the focus frame image to the region you suspect, inspect the crop, and measure pixel bbox coords from the zoomed image. Report the measured coords in `reason`. This is the only acceptable fallback when the fragment is sub-OCR-threshold or edge-clipped beyond OCR's reach."
      : "Tier 2 — Careful visual re-read of the focus frame (use ONLY if no OCR block covers the visible fragment). Zoom with get_frame again and measure precisely. Describe your measurement method in `reason`.",
    "Tier 3 — Pure estimation (LAST RESORT — avoid unless Tiers 1 and 2 have both been attempted and failed).",
    "   Only permitted when a fragment is clearly visible to you but neither OCR nor a code-execution crop can produce coords (rare). Must be explicitly justified in `reason`, including why Tiers 1 and 2 failed.",
    "",
    "HARD RULES — coordinate invention is a bug",
    "  - NEVER copy the anchor frame's bbox verbatim onto this frame. The anchor bbox is a SPATIAL PRIOR only — useful for deciding where to LOOK, never as add_box coords.",
    "  - NEVER fabricate coords from UI semantics alone (e.g. 'the formula bar shows the selected cell's content, so there must be text there'). If OCR doesn't see it and you can't crop-and-measure it, it is NOT visible for redaction purposes on this frame.",
    "  - NEVER widen an OCR block to match the anchor's width. If only 'or' is visible, redact ONLY the 'or' block's bbox.",
    "  - If you catch yourself about to add a box whose `reason` is 'same position as anchor' or 'should be in <UI element>', STOP. Either find a supporting OCR block or a code-execution measurement, or do not add the box.",
    "",
    "3. If partial content is visible but NOT already covered by an existing hit, add_box using coords from the highest-priority tier that succeeded. Mention the tier and source in `reason` (e.g. 'Tier 1: OCR WORD block `or` at ocr_index=N' / 'Tier 2: code_execution crop measured (x,y,w,h)=…').",
    "4. If the focus frame already has a hit that fully covers the visible partial, no action needed.",
    "5. Call finish(summary, still_visible) to end. If neither OCR nor code-execution found the fragment on this frame, that's strong evidence `still_visible=false` — don't invent coords just to keep the cascade alive.",
    "",
    "STOP SEMANTICS — `still_visible`",
    "true  → tracked content (or any partial / variant / fragment) is still visible on this frame OR could plausibly appear in the next cascade step (viewport edge, fading in/out). Cascade will propagate one more frame.",
    "false → tracked content is COMPLETELY GONE. The cell is cleared, the text is replaced by something unrelated, or it's scrolled fully off-screen. Cascade stops.",
    "When in doubt, return true — it's cheaper to check one extra frame than to leak a partial.",
    BOX_FORMAT_NOTE,
  ].join("\n");

  const userText = [
    `Query: "${query}".`,
    "",
    `TRACKED CONTENT: "${tracked.text}"`,
    `Anchor frame: #${tracked.source_frame}  ·  anchor box_2d: [${tracked.source_bbox.join(", ")}]`,
    `Direction: ${direction}  ·  step ${depthFromSource} from anchor.`,
    "",
    `Why this agent was spawned: ${spawnReason}`,
    "",
    `Focus frame #${focusFrame} state (post phase-1):`,
    JSON.stringify(focusSummary, null, 2),
    "",
    `--- FOCUS FRAME #${focusFrame}: RAW OCR (pixel-accurate box_2d, USE THESE for add_box) ---`,
    'These are every Textract WORD and LINE block on the focus frame. "Query-relevant" blocks are ranked highest by substring/fuzzy match to the tracked content; "context" blocks disambiguate neighboring layout.',
    "",
    "Query-relevant raw blocks (copy bbox directly when a fragment of the tracked content shows up here):",
    focusRelevantBlock,
    "",
    "Other raw blocks on this frame (context, for disambiguation):",
    focusContextBlock,
    "",
    "The focus frame JPEG is attached as an image part in this message.",
    "",
    "Inspect the focus frame for ANY visible partial of the tracked content. Follow the STRICT coord-source priority: (1) OCR block (see the dump above) → (2) code_execution crop measurement → (3) estimation. Never copy the anchor bbox verbatim and never invent coords from UI semantics. If neither OCR nor code-execution finds the fragment on this frame, return still_visible=false rather than fabricating a box. Then call finish.",
  ].join("\n");

  alog(`[${agentId}] start`, {
    focusFrame,
    source,
    parentAgentId,
    direction,
    depthFromSource,
    tracked_text: tracked.text,
    tracked_source_frame: tracked.source_frame,
    max_steps: agenticFocusedMaxSteps(),
    focus_raw_ocr_total: focusRawItems.length,
    focus_raw_ocr_relevant: focusRelevantRaw.length,
    focus_raw_ocr_context: focusContextRaw.length,
  });
  runLog?.write({
    kind: "agent_start",
    agent_id: agentId,
    focus_frame: focusFrame,
    source,
    parent_agent_id: parentAgentId,
    direction,
    depth_from_source: depthFromSource,
    tracked: {
      text: tracked.text,
      source_frame: tracked.source_frame,
      source_bbox: tracked.source_bbox,
    },
    spawn_reason: spawnReason,
    max_steps: agenticFocusedMaxSteps(),
    system_prompt: systemPrompt,
    user_prompt: userText,
    focus_raw_ocr_total: focusRawItems.length,
    focus_raw_ocr_relevant: focusRelevantRaw.length,
    focus_raw_ocr_context: focusContextRaw.length,
  });

  // ---- Run -------------------------------------------------------------

  const model = agenticLanguageModel();
  const agentT0 = Date.now();

  let usage: LanguageModelUsage | null = null;
  let aborted = false;
  try {
    const r = await generateText({
      model,
      tools,
      system: systemPrompt,
      providerOptions: agenticProviderOptions(),
      // Attach the focus-frame JPEG directly so the model has the pixels
      // from step 1 — no `get_frame(focus)` round-trip needed before
      // reasoning can start.
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: userText },
            { type: "image", image: focus.blob, mediaType: "image/jpeg" },
          ],
        },
      ],
      stopWhen: stepCountIs(agenticFocusedMaxSteps()),
      abortSignal,
      onStepFinish: (step) => {
        const stepNum = bumpStep();
        for (const call of step.toolCalls ?? []) {
          onEvent({
            type: "tool_call",
            step: stepNum,
            name: call.toolName,
            input: call.input,
            agent_id: agentId,
          });
        }
        for (const r of step.toolResults ?? []) {
          onEvent({
            type: "tool_result",
            step: stepNum,
            name: r.toolName,
            summary: summarizeToolResult(r.toolName, r.output),
            agent_id: agentId,
          });
        }
        if (step.text && step.text.trim()) {
          onEvent({
            type: "model_text",
            step: stepNum,
            text: step.text,
            agent_id: agentId,
          });
        }
        // Full step payload to the run log (text + tool I/O + usage).
        runLog?.write({
          kind: "agent_step",
          agent_id: agentId,
          step: stepNum,
          text: step.text ?? "",
          reasoning_text: step.reasoningText ?? null,
          finish_reason: step.finishReason,
          tool_calls: step.toolCalls?.map((c) => ({
            name: c.toolName,
            id: c.toolCallId,
            input: compact(c.input),
          })),
          tool_results: step.toolResults?.map((r) => ({
            name: r.toolName,
            id: r.toolCallId,
            output: compact(r.output),
          })),
          usage: step.usage,
        });
      },
    });
    usage = r.usage ?? null;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    const isAbort =
      e instanceof Error &&
      (e.name === "AbortError" ||
        /abort/i.test(msg) ||
        /The operation was aborted/i.test(msg));
    if (isAbort) {
      // Cancellation from the speculative cascade orchestrator. Return
      // the partial result so the caller can still roll back any
      // mutations this agent made before the abort fired. stillVisible
      // stays null (unknown), which the orchestrator treats as "do not
      // commit this agent's state".
      aborted = true;
      alog(`[${agentId}] aborted by orchestrator`, { focusFrame });
      runLog?.write({
        kind: "agent_aborted",
        agent_id: agentId,
        focus_frame: focusFrame,
        total_steps: steps,
        partial_added: added,
        partial_removed: removed,
      });
    } else {
      aerr(`[${agentId}] generateText failed`, e);
      runLog?.write({
        kind: "agent_error",
        agent_id: agentId,
        focus_frame: focusFrame,
        error: msg,
      });
      // Non-abort errors still surface to the caller — but we don't
      // rethrow, we return the partial result with `stillVisible=null`
      // so the orchestrator can roll back any partial state and decide
      // whether to stop the chain. The original behavior of re-throwing
      // was fine when runChain awaited one agent at a time; with
      // speculation in flight we need to keep draining.
    }
  }

  const agentElapsed = Date.now() - agentT0;
  const agentCost = geminiCost(agenticModelId(), {
    inputTokens: usage?.inputTokens ?? 0,
    outputTokens: usage?.outputTokens ?? 0,
    reasoningTokens: usage?.outputTokenDetails?.reasoningTokens ?? 0,
    cachedInputTokens: usage?.inputTokenDetails?.cacheReadTokens ?? 0,
    callCount: 1,
  });

  alog(`[${agentId}] end`, {
    focusFrame,
    added,
    removed,
    totalSteps: steps,
    finishSummary,
    stillVisible,
    input_tokens: usage?.inputTokens ?? null,
    output_tokens: usage?.outputTokens ?? null,
    cost_usd: agentCost.totalUSD,
  });
  runLog?.write({
    kind: "agent_end",
    agent_id: agentId,
    focus_frame: focusFrame,
    added,
    removed,
    total_steps: steps,
    finish_summary: finishSummary,
    still_visible: stillVisible,
    elapsed_ms: agentElapsed,
    usage,
    cost_usd: agentCost.totalUSD,
    input_usd: agentCost.inputUSD,
    output_usd: agentCost.outputUSD,
  });

  onEvent({
    type: "agent_end",
    agent_id: agentId,
    focus_frame: focusFrame,
    added,
    removed,
    total_steps: steps,
    finish_summary: finishSummary,
    usage,
    cost_usd: agentCost.totalUSD,
  });

  // Rollback: undo every add / remove this agent made to focus.hits, in
  // reverse chronological order. Emits counter frame_update events so
  // any UI observer mirrors the reversal. Called by the speculative
  // cascade orchestrator when this agent's chain is determined to have
  // stopped earlier (speculation past a `still_visible=false` /
  // quiet_streak boundary). Idempotent and safe to call never.
  const rollback = (): void => {
    if (rolledBack) return;
    rolledBack = true;
    if (deltas.length === 0) return;
    alog(`[${agentId}] rollback speculative mutations`, {
      focusFrame,
      delta_count: deltas.length,
    });
    for (let i = deltas.length - 1; i >= 0; i--) {
      const d = deltas[i];
      if (d.kind === "add") {
        const idx = focus.hits.findIndex((h) => h.hit_id === d.hit.hit_id);
        if (idx >= 0) {
          focus.hits.splice(idx, 1);
          onEvent({
            type: "frame_update",
            index: focusFrame,
            action: "remove",
            hit: d.hit,
            reason: `rollback of speculative add by ${agentId}`,
            agent_id: agentId,
          });
        }
      } else {
        focus.hits.push(d.hit);
        onEvent({
          type: "frame_update",
          index: focusFrame,
          action: "add",
          hit: d.hit,
          reason: `rollback of speculative remove by ${agentId}`,
          agent_id: agentId,
        });
      }
    }
    runLog?.write({
      kind: "agent_rolled_back",
      agent_id: agentId,
      focus_frame: focusFrame,
      reverted_adds: deltas.filter((d) => d.kind === "add").length,
      reverted_removes: deltas.filter((d) => d.kind === "remove").length,
    });
  };

  return {
    agentId,
    focusFrame,
    added,
    removed,
    totalSteps: steps,
    finishSummary,
    stillVisible: aborted ? null : stillVisible,
    usage,
    rollback,
    isRolledBack: () => rolledBack,
  };
}

function summarizeToolResult(name: string, output: unknown): string {
  if (!output || typeof output !== "object") return String(output);
  const o = output as Record<string, unknown>;
  if (o.ok === false) return `error: ${String(o.error ?? "unknown")}`;
  if (name === "get_frame") {
    const results = o.results as
      | Array<{
          frame_index?: number;
          ok?: boolean;
          summary?: { current_hits?: unknown[]; ocr_candidates?: unknown[] };
        }>
      | undefined;
    if (Array.isArray(results)) {
      const totalHits = results.reduce(
        (s, r) => s + (r.summary?.current_hits?.length ?? 0),
        0,
      );
      const totalOcr = results.reduce(
        (s, r) => s + (r.summary?.ocr_candidates?.length ?? 0),
        0,
      );
      return `frames fetched · ${results.length} frame${results.length === 1 ? "" : "s"} · ${totalHits} hits · ${totalOcr} ocr`;
    }
    // Legacy single-frame shape.
    const s = o.summary as
      | { current_hits?: unknown[]; ocr_candidates?: unknown[] }
      | undefined;
    return `frame fetched · ${s?.current_hits?.length ?? 0} hits · ${
      s?.ocr_candidates?.length ?? 0
    } ocr`;
  }
  if (name === "get_ocr_text") {
    const frames = o.frames as
      | Array<{ frame_index?: number; items?: unknown[] }>
      | undefined;
    if (Array.isArray(frames)) {
      const total = frames.reduce((s, f) => s + (f.items?.length ?? 0), 0);
      const filters = Array.isArray(o.filters) ? (o.filters as string[]) : [];
      return `raw ocr · ${frames.length} frame${frames.length === 1 ? "" : "s"} · ${total} blocks${filters.length ? ` · filters=${filters.join(",")}` : ""}`;
    }
    const items = o.items as unknown[] | undefined;
    return `raw ocr · ${items?.length ?? 0} blocks`;
  }
  if (name === "add_box" || name === "adopt_ocr_box") {
    const pb = o.pixel_box as
      | { x: number; y: number; w: number; h: number }
      | undefined;
    return pb
      ? `added ${String(o.hit_id)} at (${pb.x},${pb.y}) ${pb.w}x${pb.h}`
      : `added ${String(o.hit_id)}`;
  }
  if (name === "remove_box") return `removed ${String(o.removed_hit_id)}`;
  if (name === "finish") return `finished still_visible=${String(o.acknowledged)}`;
  return JSON.stringify(o);
}

// -- Orchestrator -----------------------------------------------------------

export type CascadeOpts = {
  query: string;
  frames: NavFrameState[];
  onEvent: (ev: NavigatorEvent) => void;
  abortSignal?: AbortSignal;
  /**
   * Optional per-run JSONL log. Cascade writes agent_start / agent_step /
   * agent_end / agent_error + cascade_chain_stopped / cascade_summary.
   */
  runLog?: RunLog | null;
};

export type CascadeResult = {
  added: number;
  removed: number;
  totalSteps: number;
  totalAgents: number;
  transitions: number;
  finishSummary: string | null;
  /** Aggregate token usage across every focused agent in the run. */
  usage: AggregateUsage;
};

/**
 * Orchestrate the multi-agent cascade navigator with bidirectional
 * propagation.
 *
 * Invariants:
 *   - Each frame is analyzed by at most one cascade step; chains claim a
 *     frame atomically before spawning an agent and bail if already
 *     claimed.
 *   - Each chain is sequential; chains run in parallel bounded by
 *     AGENTIC_CASCADE_CONCURRENCY.
 *   - Hard safety cap: AGENTIC_CASCADE_MAX_AGENTS across all chains.
 *   - A chain stops on agent `still_visible=false`, on max depth, or on
 *     a claimed next frame.
 */
export async function runCascadeNavigator(
  opts: CascadeOpts,
): Promise<CascadeResult> {
  const { query, frames, onEvent, abortSignal, runLog } = opts;
  const frameByIndex = new Map<number, NavFrameState>();
  for (const f of frames) frameByIndex.set(f.index, f);
  const rangeLo = frames[0]?.index ?? 0;
  const rangeHi = frames[frames.length - 1]?.index ?? 0;

  const seeds = findChainSeeds(frames);
  alog("cascade chain seeds computed", {
    range: [rangeLo, rangeHi],
    total_frames: frames.length,
    total_seeds: seeds.length,
    seeds: seeds.map((s) => ({
      direction: s.direction,
      text: s.tracked.text,
      source_frame: s.tracked.source_frame,
      first_focus: s.first_focus,
    })),
  });
  runLog?.write({
    kind: "cascade_seeds",
    total_frames: frames.length,
    range: [rangeLo, rangeHi],
    seeds: seeds.map((s) => ({
      direction: s.direction,
      text: s.tracked.text,
      source_frame: s.tracked.source_frame,
      source_bbox: s.tracked.source_bbox,
      first_focus: s.first_focus,
    })),
  });

  // Per-direction claim maps. A single frame CAN be visited once by a
  // backward chain and once by a forward chain — they are tracking
  // different content, and claiming globally would starve one direction
  // of coverage. Within a direction, first-come-first-served.
  const claimed: Record<"backward" | "forward", Set<number>> = {
    backward: new Set(),
    forward: new Set(),
  };

  // Shared helper so every chain-stop reason is emitted to both the
  // server console (alog) and the per-run JSONL (runLog) with identical
  // payloads.
  const logChainStop = (payload: Record<string, unknown>): void => {
    alog("cascade chain stopped", payload);
    runLog?.write({ kind: "chain_stopped", ...payload });
  };

  let agentCounter = 0;
  let totalAdded = 0;
  let totalRemoved = 0;
  let totalSteps = 0;
  let totalAgents = 0;
  const totalUsage = emptyUsage();
  const maxAgents = agenticCascadeMaxAgents();
  const maxDepth = agenticCascadeMaxDepth();
  const quietCap = agenticCascadeQuietCap();
  const specDepth = Math.max(1, agenticCascadeSpeculationDepth());

  // Combine the run-wide abort signal with a per-slot cancel controller
  // so the speculative orchestrator can cancel an individual in-flight
  // agent without cancelling the whole run. Prefer AbortSignal.any()
  // when available; fall back to a manually-linked controller on older
  // runtimes.
  const linkAbortSignals = (
    outer: AbortSignal | undefined,
    inner: AbortSignal,
  ): AbortSignal => {
    if (!outer) return inner;
    const anyFn = (
      AbortSignal as unknown as {
        any?: (signals: AbortSignal[]) => AbortSignal;
      }
    ).any;
    if (typeof anyFn === "function") {
      return anyFn.call(AbortSignal, [outer, inner]);
    }
    const ctrl = new AbortController();
    const relay = (src: AbortSignal) => {
      if (src.aborted) ctrl.abort(src.reason);
      else
        src.addEventListener("abort", () => ctrl.abort(src.reason), {
          once: true,
        });
    };
    relay(outer);
    relay(inner);
    return ctrl.signal;
  };

  // Context window for a given focus frame: focus + both immediate
  // neighbors. Small enough to keep per-agent token cost modest, wide
  // enough for the agent to reference the previous and next frame.
  const windowFor = (focusFrame: number): NavFrameState[] => {
    const w: NavFrameState[] = [];
    const prev = frameByIndex.get(focusFrame - 1);
    if (prev) w.push(prev);
    const cur = frameByIndex.get(focusFrame);
    if (cur) w.push(cur);
    const next = frameByIndex.get(focusFrame + 1);
    if (next) w.push(next);
    return w;
  };

  // Run a single bidirectional chain from a seed.
  //
  // Speculative lookahead: the chain maintains an in-flight window of up
  // to `specDepth` focused agents at once. Agent N doesn't need to wait
  // for agent N-1 to return `still_visible=true` before being spawned;
  // we speculate that the chain continues (which, empirically, is true
  // >95% of the time — `still_visible=false` is rare vs. `quiet_streak`
  // / range-edge terminations).
  //
  // Correctness:
  //   - Each speculated agent records its mutations via the delta log
  //     in `runFocusedAgent`. When a committed head returns
  //     `still_visible=false` or trips the quiet-streak guard, we abort
  //     every speculated agent past it AND call `rollback()` on each,
  //     undoing their hits. The net observable state is identical to
  //     the baseline (non-speculative) sequential cascade.
  //   - Agents are committed in chain order (head-of-queue first), so
  //     the `quietStreak` counter advances exactly as it would
  //     sequentially.
  const runChain = async (seed: ChainSeed): Promise<void> => {
    const dir = seed.direction;
    const step = dir === "backward" ? -1 : 1;

    // Cursor tracking the NEXT frame/depth/parent to SPAWN (not await).
    let nextFrame = seed.first_focus;
    let nextDepth = 1;
    let nextSource: "transition" | "cascade" = "transition";
    let nextParentAgentId: string | null = null;
    let nextReason =
      dir === "backward"
        ? `Backward chain for "${seed.tracked.text}" anchored at frame #${seed.tracked.source_frame}: sweeping earlier frames for lead-in partials.`
        : `Forward chain for "${seed.tracked.text}" anchored at frame #${seed.tracked.source_frame}: sweeping later frames for residual partials.`;

    let quietStreak = 0;

    type Slot = {
      agentId: string;
      focusFrame: number;
      depth: number;
      speculative: boolean;
      cancel: AbortController;
      promise: Promise<FocusedAgentResult | null>;
    };
    const slots: Slot[] = [];

    // A non-fatal "can't spawn more" reason discovered during speculation.
    // Emitted at drain time IF the chain doesn't stop earlier via
    // still_visible=false / quiet_streak.
    let deferredStop: Record<string, unknown> | null = null;

    const trySpawnNext = (): boolean => {
      if (deferredStop) return false;
      if (nextDepth > maxDepth) {
        deferredStop = {
          reason: "max_depth",
          direction: dir,
          tracked_text: seed.tracked.text,
          anchor: seed.tracked.source_frame,
          depth: nextDepth,
        };
        return false;
      }
      if (totalAgents >= maxAgents) {
        deferredStop = {
          reason: "max_agents_global",
          direction: dir,
          tracked_text: seed.tracked.text,
          totalAgents,
        };
        return false;
      }
      if (nextFrame < rangeLo || nextFrame > rangeHi) {
        deferredStop = {
          reason: "out_of_range",
          direction: dir,
          tracked_text: seed.tracked.text,
          focus: nextFrame,
          range: [rangeLo, rangeHi],
        };
        return false;
      }
      if (claimed[dir].has(nextFrame)) {
        deferredStop = {
          reason: "frame_already_claimed",
          direction: dir,
          tracked_text: seed.tracked.text,
          focus: nextFrame,
        };
        return false;
      }
      const ctx = windowFor(nextFrame);
      if (!ctx.some((f) => f.index === nextFrame)) {
        deferredStop = {
          reason: "out_of_range",
          direction: dir,
          tracked_text: seed.tracked.text,
          focus: nextFrame,
          range: [rangeLo, rangeHi],
        };
        return false;
      }

      claimed[dir].add(nextFrame);
      agentCounter += 1;
      totalAgents += 1;
      const agentId = `A${agentCounter}`;
      const focusFrame = nextFrame;
      const depth = nextDepth;
      const speculative = slots.length > 0;
      const speculativeNote = speculative
        ? ` (SPECULATIVE — parent ${nextParentAgentId ?? "?"} has not yet reported)`
        : "";
      const spawnReason = `${nextReason}${speculativeNote}`;
      const cancel = new AbortController();
      const linked = linkAbortSignals(abortSignal, cancel.signal);

      const promise = (async (): Promise<FocusedAgentResult | null> => {
        try {
          return await runFocusedAgent({
            agentId,
            focusFrame,
            contextFrames: ctx,
            query,
            source: nextSource,
            parentAgentId: nextParentAgentId,
            tracked: seed.tracked,
            direction: dir,
            depthFromSource: depth,
            spawnReason,
            onEvent,
            abortSignal: linked,
            runLog,
          });
        } catch (e) {
          aerr(`[${agentId}] runFocusedAgent threw unexpectedly`, e);
          return null;
        }
      })();

      slots.push({ agentId, focusFrame, depth, speculative, cancel, promise });

      nextParentAgentId = agentId;
      nextSource = "cascade";
      nextReason = `Continuing ${dir} chain for "${seed.tracked.text}" — parent ${agentId} on frame #${focusFrame} reported still visible.`;
      nextFrame += step;
      nextDepth += 1;
      return true;
    };

    const abortAndRollbackRest = async (): Promise<void> => {
      const toCancel = slots.splice(0, slots.length);
      for (const s of toCancel) s.cancel.abort();
      await Promise.all(
        toCancel.map(async (s) => {
          const r = await s.promise;
          if (r) r.rollback();
        }),
      );
    };

    // Seed the initial speculation window.
    while (slots.length < specDepth) {
      if (!trySpawnNext()) break;
    }

    // Drain head-first; refill after each commit.
    while (slots.length > 0) {
      const head = slots.shift()!;
      const result = await head.promise;

      if (result === null) {
        // runFocusedAgent threw something non-abort. Treat as terminal
        // and roll back any speculated descendants.
        logChainStop({
          reason: "agent_failed",
          direction: dir,
          tracked_text: seed.tracked.text,
          focus: head.focusFrame,
          agent_id: head.agentId,
          depth: head.depth,
        });
        await abortAndRollbackRest();
        return;
      }

      // Commit this agent's metrics.
      totalAdded += result.added;
      totalRemoved += result.removed;
      totalSteps += result.totalSteps;
      addUsage(totalUsage, result.usage);

      if (result.stillVisible === false) {
        logChainStop({
          reason: "tracked_content_gone",
          direction: dir,
          tracked_text: seed.tracked.text,
          focus: head.focusFrame,
          agent_id: head.agentId,
          depth: head.depth,
        });
        await abortAndRollbackRest();
        return;
      }
      if (result.stillVisible === null) {
        alog("cascade chain continuing despite missing finish signal", {
          direction: dir,
          tracked_text: seed.tracked.text,
          focus: head.focusFrame,
          agent_id: head.agentId,
        });
      }

      const quiet =
        result.stillVisible === true &&
        result.added === 0 &&
        result.removed === 0;
      if (quiet) {
        quietStreak += 1;
      } else {
        quietStreak = 0;
      }
      if (quietStreak >= quietCap) {
        logChainStop({
          reason: "quiet_streak",
          direction: dir,
          tracked_text: seed.tracked.text,
          anchor: seed.tracked.source_frame,
          focus: head.focusFrame,
          agent_id: head.agentId,
          depth: head.depth,
          quiet_streak: quietStreak,
          quiet_cap: quietCap,
        });
        await abortAndRollbackRest();
        return;
      }

      // Head committed; refill one speculative slot to keep the pipeline full.
      trySpawnNext();
    }

    // Drained cleanly without hitting a runtime stop — emit whatever
    // deferred-stop reason we captured while filling the pipeline.
    if (deferredStop) {
      logChainStop(deferredStop);
    }
  };

  // Parallel workers over the seed list, bounded by concurrency.
  const concurrency = Math.max(1, agenticCascadeConcurrency());
  let cursor = 0;
  const worker = async (): Promise<void> => {
    while (true) {
      const my = cursor++;
      if (my >= seeds.length) return;
      if (totalAgents >= maxAgents) return;
      await runChain(seeds[my]);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(concurrency, seeds.length) }, () =>
      worker(),
    ),
  );

  onEvent({
    type: "finish",
    summary: `Cascade ran ${totalAgents} agents across ${seeds.length} chain seeds. +${totalAdded} / -${totalRemoved} boxes; ${totalSteps} model steps total.`,
    total_steps: totalSteps,
  });

  return {
    added: totalAdded,
    removed: totalRemoved,
    totalSteps,
    totalAgents,
    transitions: seeds.length,
    finishSummary: `Cascade: ${totalAgents} agents · +${totalAdded} / -${totalRemoved}`,
    usage: totalUsage,
  };
}

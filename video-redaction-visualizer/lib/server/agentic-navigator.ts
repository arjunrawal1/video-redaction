// Agentic second pass: the navigator (GPT-5.4 via OpenRouter).
//
// Given the curator's per-frame output (hits + rejected OCR candidates),
// we hand the whole session to GPT-5.4 as a tool-calling agent. The
// model decides which frames to visit, adds missing boxes, removes
// spurious ones, and can adopt the coords of any OCR candidate (rejected
// or not) when it wants to redact a specific rectangle.
//
// The navigator is the "ultimate decision maker": the curator's output
// is just a starting state. The model can freely revise it as it
// navigates.
//
// Box contract (Gemini's native box_2d format):
//   box_2d = [y_min, x_min, y_max, x_max] in integer 0..1000 (top-left origin)
//
// Control flow:
//   - Initial prompt contains a compact summary of every frame in range.
//   - Tools: `get_frame`, `add_box`, `remove_box`, `adopt_ocr_box`,
//     `finish`.
//   - Loop runs until the model stops calling tools OR `finish` is called
//     OR the safety step ceiling is hit.

import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { aerr, alog } from "./agentic-log";
import {
  BOX_FORMAT_NOTE,
  agenticBuiltinTools,
  agenticCodeExecutionEnabled,
  agenticLanguageModel,
  agenticModelSlug,
  agenticNavigatorMaxSteps,
  agenticProviderOptions,
  agenticThinkingLevel,
  bboxToPixels,
  pixelBoxToNormalizedBbox,
  type ServerBox,
} from "./openrouter";

// Subset of the Textract DetectDocumentText Block shape we actually read
// when the navigator asks for raw OCR. Keeps the `get_ocr_text` tool
// self-contained without pulling in boto3 or the AWS SDK just for types.
type TextractBlock = {
  BlockType?: string;
  Text?: string;
  Confidence?: number;
  Geometry?: {
    BoundingBox?: {
      Left?: number;
      Top?: number;
      Width?: number;
      Height?: number;
    };
  };
};

export type NavHit = ServerBox & { hit_id: string; reason?: string };

export type NavFrameState = {
  index: number;
  width: number;
  height: number;
  blob: Uint8Array;
  hits: NavHit[];
  ocrBoxes: ServerBox[];
  // Full Textract response for this frame (Blocks[]). Exposed through the
  // `get_ocr_text` tool so the navigator can see raw OCR — including
  // detections that the fuzzy matcher filtered out — rather than trusting
  // our pre-filtered candidate list.
  ocrRaw: unknown;
};

export type NavigatorEvent =
  // Cascade-only: one of these fires every time a focused agent is about
  // to start. In single-agent mode there's implicitly one agent and
  // these are omitted.
  | {
      type: "agent_start";
      agent_id: string;
      focus_frame: number;
      source: "transition" | "cascade";
      parent_agent_id: string | null;
      reason: string;
    }
  | {
      type: "agent_end";
      agent_id: string;
      focus_frame: number;
      added: number;
      removed: number;
      total_steps: number;
      finish_summary: string | null;
    }
  | {
      type: "tool_call";
      step: number;
      name: string;
      input: unknown;
      agent_id?: string;
    }
  | {
      type: "tool_result";
      step: number;
      name: string;
      summary: string;
      agent_id?: string;
    }
  | {
      type: "frame_update";
      index: number;
      action: "add" | "remove";
      hit: NavHit;
      reason?: string;
      agent_id?: string;
    }
  | {
      type: "model_text";
      step: number;
      text: string;
      agent_id?: string;
    }
  | { type: "finish"; summary: string; total_steps: number };

const HIT_ID_PREFIX = "H";
let _hitCounter = 0;
function mintHitId(): string {
  _hitCounter = (_hitCounter + 1) % 1_000_000;
  return `${HIT_ID_PREFIX}${Date.now().toString(36)}${_hitCounter.toString(36)}`;
}

/**
 * Build the compact summary of the whole pass sent up front so the model
 * can plan which frames to visit. Keeps each line under ~200 chars so the
 * initial prompt stays in the low-thousand-token range even for 40-frame
 * ranges.
 */
function summarizePhase1(frames: NavFrameState[]): string {
  const lines: string[] = [];
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const prev = i > 0 ? frames[i - 1] : null;
    const hitCount = f.hits.length;
    const prevCount = prev?.hits.length ?? 0;
    const delta = prev ? hitCount - prevCount : hitCount;
    const transition = !prev
      ? "[start]"
      : hitCount === prevCount
        ? ""
        : delta > 0
          ? `[+${delta} vs prev]`
          : `[${delta} vs prev]`;
    const texts = f.hits
      .slice(0, 3)
      .map((h) => `"${h.text.slice(0, 40)}"`)
      .join(", ");
    const more = f.hits.length > 3 ? ` (+${f.hits.length - 3} more)` : "";
    const ocrHint = f.ocrBoxes.length
      ? ` · ${f.ocrBoxes.length} ocr candidates`
      : "";
    lines.push(
      `#${f.index}: ${hitCount} hits${ocrHint} ${transition} ${texts}${more}`.trim(),
    );
  }
  return lines.join("\n");
}

function system(): string {
  return [
    "You are the ULTIMATE decision maker for a video redaction pipeline.",
    "A first-pass curator has already produced candidate boxes per frame from OCR + vision review. Your job is to fix the remaining errors by moving between frames freely.",
    "Focus on transitions where the set of hits changes from one frame to the next: those are the most likely places for partial reveals, missed instances, or false positives that need correction.",
    "Tools available:",
    "  - `get_frame(frame_index)` — fetch an image + current hits + the query-matching OCR candidates.",
    "  - `get_ocr_text(frame_index, filter?, block_types?)` — fetch ALL raw OCR (every Textract WORD and LINE with exact bboxes), including ones the fuzzy matcher filtered out. Call this when you suspect OCR did detect the target but it was rejected, or when you want exact pixel-accurate coords for a specific frame before adding a box.",
    "  - `add_box(frame_index, text, bbox)` — add a new redaction rectangle.",
    "  - `remove_box(frame_index, hit_id)` — drop a false positive.",
    "  - `adopt_ocr_box(frame_index, ocr_index)` — promote a pre-filtered OCR candidate into a hit.",
    "  - `finish(summary)` — terminate once every frame is correctly redacted.",
    BOX_FORMAT_NOTE,
    "Prefer tight boxes that wrap just the query text. Do not over-redact whole rows.",
    "Reuse OCR rectangles whenever possible. Use `adopt_ocr_box` when a filtered candidate already exists, and `get_ocr_text` to find bboxes for partials or garbled hits that didn't make it past the fuzzy filter. The on-screen content may SCROLL across frames, so DO NOT blindly copy coordinates from one frame to another — confirm with `get_frame` or `get_ocr_text` on the target frame first.",
    "Call `finish` when no further edits are needed. Do not call it before inspecting any frames that appear suspicious.",
  ].join("\n");
}

export async function runNavigator(opts: {
  query: string;
  frames: NavFrameState[];
  onEvent: (ev: NavigatorEvent) => void;
  abortSignal?: AbortSignal;
}): Promise<{
  totalSteps: number;
  added: number;
  removed: number;
  finishSummary: string | null;
}> {
  const { query, frames, onEvent, abortSignal } = opts;
  const frameByIndex = new Map<number, NavFrameState>();
  for (const f of frames) frameByIndex.set(f.index, f);

  let added = 0;
  let removed = 0;
  let finishSummary: string | null = null;
  let steps = 0;

  const bumpStep = () => {
    steps += 1;
    return steps;
  };

  const getFrameTool = tool({
    description:
      "Fetch a frame image plus its current hits and OCR candidates. Call this before adding or removing boxes so you can see what's actually there.",
    inputSchema: z.object({
      frame_index: z
        .number()
        .int()
        .describe("1-based frame index in the range being navigated."),
    }),
    execute: async ({ frame_index }) => {
      alog(`tool get_frame invoked`, { frame_index });
      const f = frameByIndex.get(frame_index);
      if (!f) {
        alog(`tool get_frame miss`, { frame_index, known: [...frameByIndex.keys()] });
        return {
          ok: false as const,
          error: `Frame ${frame_index} is not in the navigated range.`,
        };
      }
      const summary = {
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
      };
      alog(`tool get_frame returning`, {
        frame_index,
        jpeg_bytes: f.blob.byteLength,
        current_hits: summary.current_hits.length,
        ocr_candidates: summary.ocr_candidates.length,
      });
      return {
        ok: true as const,
        summary,
        image_base64: Buffer.from(f.blob).toString("base64"),
      };
    },
    toModelOutput: ({ output }) => {
      if (!output.ok) {
        return { type: "error-text", value: output.error };
      }
      return {
        type: "content",
        value: [
          {
            type: "text",
            text: JSON.stringify(output.summary, null, 2),
          },
          {
            type: "image-data",
            mediaType: "image/jpeg",
            data: output.image_base64,
          },
        ],
      };
    },
  });

  const addBoxTool = tool({
    description:
      "Add a new redaction box to a frame using Gemini's native box_2d format: [y_min, x_min, y_max, x_max] in 0..1000 (top-left origin). Returns the new hit_id.",
    inputSchema: z.object({
      frame_index: z.number().int(),
      text: z.string().describe("The visible on-screen text being redacted."),
      y_min: z.number().int().min(0).max(1000),
      x_min: z.number().int().min(0).max(1000),
      y_max: z.number().int().min(0).max(1000),
      x_max: z.number().int().min(0).max(1000),
      reason: z.string().optional(),
    }),
    execute: async ({ frame_index, text, y_min, x_min, y_max, x_max, reason }) => {
      alog(`tool add_box invoked`, {
        frame_index,
        text,
        box_2d: [y_min, x_min, y_max, x_max],
        reason,
      });
      const f = frameByIndex.get(frame_index);
      if (!f) return { ok: false as const, error: `Unknown frame ${frame_index}` };
      const rect = bboxToPixels([y_min, x_min, y_max, x_max], f.width, f.height);
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
      f.hits.push(hit);
      added += 1;
      alog(`tool add_box applied`, {
        frame_index,
        hit_id: hit.hit_id,
        pixel_box: { x: hit.x, y: hit.y, w: hit.w, h: hit.h },
      });
      onEvent({
        type: "frame_update",
        index: frame_index,
        action: "add",
        hit,
        reason,
      });
      return {
        ok: true as const,
        hit_id: hit.hit_id,
        pixel_box: { x: hit.x, y: hit.y, w: hit.w, h: hit.h },
      };
    },
  });

  const removeBoxTool = tool({
    description:
      "Remove an existing hit by its hit_id. Use when the curator kept a box that is actually a false positive.",
    inputSchema: z.object({
      frame_index: z.number().int(),
      hit_id: z.string(),
      reason: z.string().optional(),
    }),
    execute: async ({ frame_index, hit_id, reason }) => {
      alog(`tool remove_box invoked`, { frame_index, hit_id, reason });
      const f = frameByIndex.get(frame_index);
      if (!f) return { ok: false as const, error: `Unknown frame ${frame_index}` };
      const idx = f.hits.findIndex((h) => h.hit_id === hit_id);
      if (idx < 0) {
        alog(`tool remove_box miss`, {
          frame_index,
          hit_id,
          available_hit_ids: f.hits.map((h) => h.hit_id),
        });
        return { ok: false as const, error: `hit_id ${hit_id} not on frame` };
      }
      const [removedHit] = f.hits.splice(idx, 1);
      removed += 1;
      alog(`tool remove_box applied`, { frame_index, removed_hit_id: hit_id });
      onEvent({
        type: "frame_update",
        index: frame_index,
        action: "remove",
        hit: removedHit,
        reason,
      });
      return { ok: true as const, removed_hit_id: hit_id };
    },
  });

  const adoptOcrTool = tool({
    description:
      "Promote an OCR candidate rectangle to a real hit. Useful when you want to reuse a tight OCR box rather than describe a new rectangle from scratch.",
    inputSchema: z.object({
      frame_index: z.number().int(),
      ocr_index: z.number().int(),
      text: z.string().optional(),
      reason: z.string().optional(),
    }),
    execute: async ({ frame_index, ocr_index, text, reason }) => {
      alog(`tool adopt_ocr_box invoked`, { frame_index, ocr_index, text, reason });
      const f = frameByIndex.get(frame_index);
      if (!f) return { ok: false as const, error: `Unknown frame ${frame_index}` };
      const src = f.ocrBoxes[ocr_index];
      if (!src) {
        alog(`tool adopt_ocr_box miss`, {
          frame_index,
          ocr_index,
          available: f.ocrBoxes.length,
        });
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
      f.hits.push(hit);
      added += 1;
      alog(`tool adopt_ocr_box applied`, {
        frame_index,
        hit_id: hit.hit_id,
        pixel_box: { x: hit.x, y: hit.y, w: hit.w, h: hit.h },
        source_ocr_text: src.text,
      });
      onEvent({
        type: "frame_update",
        index: frame_index,
        action: "add",
        hit,
        reason,
      });
      return {
        ok: true as const,
        hit_id: hit.hit_id,
        pixel_box: { x: hit.x, y: hit.y, w: hit.w, h: hit.h },
      };
    },
  });

  const getOcrTextTool = tool({
    description:
      "Fetch the RAW OCR text detections for a frame — every WORD and LINE that Textract emitted, including items that the query-matching filter dropped. Use this when you suspect OCR actually detected the query (or a partial) but our fuzzy filter rejected it, or when you want to know exact coordinates before calling add_box. Optionally filter by a substring (case-insensitive) to narrow results.",
    inputSchema: z.object({
      frame_index: z.number().int(),
      filter: z
        .string()
        .nullable()
        .describe(
          "Optional case-insensitive substring filter on detected text. Pass null for all detections.",
        ),
      block_types: z
        .array(z.enum(["WORD", "LINE"]))
        .nullable()
        .describe(
          'Which Textract block types to return. Pass null for both. LINE aggregates the tokens on a row; WORD is per-token.',
        ),
    }),
    execute: async ({ frame_index, filter, block_types }) => {
      alog(`tool get_ocr_text invoked`, { frame_index, filter, block_types });
      const f = frameByIndex.get(frame_index);
      if (!f) {
        return {
          ok: false as const,
          error: `Frame ${frame_index} is not in the navigated range.`,
        };
      }
      const raw = f.ocrRaw as { Blocks?: TextractBlock[] } | null;
      const blocks = raw?.Blocks ?? [];
      const wantWord = !block_types || block_types.includes("WORD");
      const wantLine = !block_types || block_types.includes("LINE");
      const needle = filter ? filter.toLowerCase() : null;

      const items: Array<{
        type: "word" | "line";
        text: string;
        confidence: number;
        box_2d: [number, number, number, number];
      }> = [];
      for (const b of blocks) {
        const btype = b.BlockType;
        if (btype !== "WORD" && btype !== "LINE") continue;
        if (btype === "WORD" && !wantWord) continue;
        if (btype === "LINE" && !wantLine) continue;
        const text = String(b.Text ?? "");
        if (needle && !text.toLowerCase().includes(needle)) continue;
        const bbox = (b.Geometry?.BoundingBox ?? null) as
          | { Left?: number; Top?: number; Width?: number; Height?: number }
          | null;
        if (!bbox) continue;
        const left = Number(bbox.Left ?? 0);
        const top = Number(bbox.Top ?? 0);
        const w = Number(bbox.Width ?? 0);
        const h = Number(bbox.Height ?? 0);
        // Textract returns 0..1. Convert to Gemini's native box_2d
        // contract: [y_min, x_min, y_max, x_max] in 0..1000.
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
        });
      }
      alog(`tool get_ocr_text returning`, {
        frame_index,
        filter,
        total_blocks: blocks.length,
        matched: items.length,
      });
      return { ok: true as const, frame_index, items };
    },
  });

  const finishTool = tool({
    description:
      "Call this when every frame in the range has the correct set of redaction boxes. After calling this, do not issue any more tool calls.",
    inputSchema: z.object({
      summary: z
        .string()
        .describe(
          "Brief summary of what was changed (frames visited, adds/removes).",
        ),
    }),
    execute: async ({ summary }) => {
      alog(`tool finish invoked`, { summary });
      finishSummary = summary;
      return { ok: true as const, acknowledged: true };
    },
  });

  // Include Gemini's built-in code_execution tool so the navigator can
  // crop + zoom + measure images from Python when visual estimation
  // would otherwise be needed. No-op if AGENTIC_CODE_EXECUTION=false.
  const tools = {
    get_frame: getFrameTool,
    get_ocr_text: getOcrTextTool,
    add_box: addBoxTool,
    remove_box: removeBoxTool,
    adopt_ocr_box: adoptOcrTool,
    finish: finishTool,
    ...agenticBuiltinTools(),
  };

  const initialText = [
    `Redaction query: "${query}".`,
    `Frames in range (${frames.length} total, already curated once):`,
    summarizePhase1(frames),
    "",
    agenticCodeExecutionEnabled()
      ? "You may call the built-in code_execution tool to crop / zoom / annotate frames in Python when visual estimation would otherwise be needed."
      : "",
    "Inspect any frame whose transition (vs. previous frame) looks suspicious, then add or remove boxes as needed. Call `finish` when done.",
  ]
    .filter(Boolean)
    .join("\n");

  const model = agenticLanguageModel();

  alog("navigator start", {
    model: agenticModelSlug(),
    max_steps: agenticNavigatorMaxSteps(),
    thinking_level: agenticThinkingLevel(),
    code_execution: agenticCodeExecutionEnabled(),
    query,
    frames_in_range: frames.length,
    total_phase1_hits: frames.reduce((s, f) => s + f.hits.length, 0),
    total_ocr_candidates: frames.reduce((s, f) => s + f.ocrBoxes.length, 0),
    system_prompt: system(),
    initial_user_message: initialText,
  });

  const t0 = Date.now();
  let result;
  try {
    result = await generateText({
      model,
      tools,
      system: system(),
      providerOptions: agenticProviderOptions(),
      messages: [{ role: "user", content: [{ type: "text", text: initialText }] }],
      stopWhen: stepCountIs(agenticNavigatorMaxSteps()),
      abortSignal,
      onStepFinish: (step) => {
        const stepNum = bumpStep();
        alog(`navigator step ${stepNum} finished`, {
          finishReason: step.finishReason,
          usage: step.usage,
          text: step.text,
          reasoningText: step.reasoningText,
          reasoning: step.reasoning,
          content: step.content,
          toolCalls: step.toolCalls?.map((c) => ({
            toolName: c.toolName,
            toolCallId: c.toolCallId,
            input: c.input,
          })),
          toolResults: step.toolResults?.map((r) => ({
            toolName: r.toolName,
            toolCallId: r.toolCallId,
            output: r.output,
          })),
        });
        for (const call of step.toolCalls ?? []) {
          onEvent({
            type: "tool_call",
            step: stepNum,
            name: call.toolName,
            input: call.input,
          });
        }
        for (const r of step.toolResults ?? []) {
          const summary = summarizeToolResult(r.toolName, r.output);
          onEvent({
            type: "tool_result",
            step: stepNum,
            name: r.toolName,
            summary,
          });
        }
        if (step.text && step.text.trim()) {
          onEvent({ type: "model_text", step: stepNum, text: step.text });
        }
      },
    });
  } catch (e) {
    aerr("navigator generateText failed", e);
    throw e;
  }

  const elapsed = Date.now() - t0;
  alog(`navigator finished (${elapsed}ms)`, {
    total_steps: steps,
    added,
    removed,
    finishSummary,
    finalFinishReason: result.finishReason,
    finalUsage: result.usage,
    finalText: result.text,
    warnings: result.warnings,
  });

  onEvent({
    type: "finish",
    summary: finishSummary ?? "(no summary provided)",
    total_steps: steps,
  });

  return { totalSteps: steps, added, removed, finishSummary };
}

function summarizeToolResult(name: string, output: unknown): string {
  if (!output || typeof output !== "object") return String(output);
  const o = output as Record<string, unknown>;
  if (o.ok === false) return `error: ${String(o.error ?? "unknown")}`;
  if (name === "get_frame") {
    const s = o.summary as { current_hits?: unknown[]; ocr_candidates?: unknown[] } | undefined;
    return `frame fetched · ${s?.current_hits?.length ?? 0} hits · ${s?.ocr_candidates?.length ?? 0} ocr`;
  }
  if (name === "get_ocr_text") {
    const items = o.items as unknown[] | undefined;
    return `raw ocr · ${items?.length ?? 0} blocks on frame #${String(o.frame_index)}`;
  }
  if (name === "add_box" || name === "adopt_ocr_box") {
    const pb = o.pixel_box as { x: number; y: number; w: number; h: number } | undefined;
    return pb
      ? `added hit ${String(o.hit_id)} at (${pb.x}, ${pb.y}) ${pb.w}x${pb.h}`
      : `added hit ${String(o.hit_id)}`;
  }
  if (name === "remove_box") return `removed ${String(o.removed_hit_id)}`;
  if (name === "finish") return `finished`;
  return JSON.stringify(o);
}

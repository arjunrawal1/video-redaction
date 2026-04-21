import { createHash } from "node:crypto";
import { generateObject, tool } from "ai";
import { z } from "zod";
import { extractRawOcrItems } from "@/lib/server/agentic-ocr";
import { normalizeHexColor } from "@/lib/server/box-shrink";
import { bboxToPixels, type ServerBox } from "@/lib/server/openrouter";
import {
  agenticLanguageModel,
  agenticModelSlug,
  agenticProviderOptions,
} from "@/lib/server/openrouter";
import { compact, type RunLog } from "@/lib/server/run-log";
import { localizeRegionForFrame } from "./resolver";
import type {
  Instance,
  Predicate,
  RegionBbox,
  ResolvedCandidate,
  PredicateLeaf,
} from "./types";

type RawItem = ReturnType<typeof extractRawOcrItems>[number] & { ocr_index: number };

const BboxSchema = z.array(z.number().int().min(0).max(1000)).length(4);

const SemanticClassifierSchema = z.object({
  scores: z.array(
    z.object({
      ocr_index: z.number().int(),
      confidence: z.number().min(0).max(1),
      ok: z.boolean(),
    }),
  ),
});

const _semanticCache = new Map<string, Array<{ ocr_index: number; confidence: number }>>();

function semanticKey(args: {
  videoHash: string;
  frameIndex: number;
  description: string;
}): string {
  const descHash = createHash("sha256")
    .update(args.description.trim().toLowerCase())
    .digest("hex");
  return `${args.videoHash}|${args.frameIndex}|${descHash}`;
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function intersects(a: [number, number, number, number], b: [number, number, number, number]): boolean {
  const [ay1, ax1, ay2, ax2] = a;
  const [by1, bx1, by2, bx2] = b;
  const iy1 = Math.max(ay1, by1);
  const ix1 = Math.max(ax1, bx1);
  const iy2 = Math.min(ay2, by2);
  const ix2 = Math.min(ax2, bx2);
  return iy2 > iy1 && ix2 > ix1;
}

function toRawItems(rawOcr: unknown): RawItem[] {
  const items = extractRawOcrItems(rawOcr);
  return items.map((item, ocr_index) => ({ ...item, ocr_index }));
}

function toServerBox(
  bbox: [number, number, number, number],
  text: string,
  width: number,
  height: number,
  score = 1,
): ServerBox {
  const rect = bboxToPixels(bbox, width, height);
  return {
    x: rect.x,
    y: rect.y,
    w: rect.w,
    h: rect.h,
    text,
    score,
  };
}

export type PromptToolMutation = {
  kept: ServerBox[];
  added: ServerBox[];
  dropped: number[];
  instances: Instance[];
  finish_summary: string | null;
  any_more_to_check: boolean;
  still_visible: boolean | null;
};

export type PromptToolContext = {
  videoHash: string;
  frameIndex: number;
  frameBlob: Uint8Array;
  width: number;
  height: number;
  rawOcr: unknown;
  predicate: Predicate;
  regions: RegionBbox[];
  textCandidates: ResolvedCandidate[];
  existingHits?: ServerBox[];
  semanticMinConfidenceDefault?: number;
  runLog?: RunLog | null;
};

export type GetOcrItem = {
  ocr_index: number;
  type: "word" | "line";
  text: string;
  confidence: number;
  bbox_2d: [number, number, number, number];
  semantic_score?: number;
};

async function classifySemantic(args: {
  videoHash: string;
  frameIndex: number;
  description: string;
  items: RawItem[];
  runLog?: RunLog | null;
}): Promise<Array<{ ocr_index: number; confidence: number }>> {
  const key = semanticKey(args);
  const cached = _semanticCache.get(key);
  if (cached) {
    args.runLog?.write({
      kind: "semantic_cache_hit",
      frame_index: args.frameIndex,
      description: args.description,
      scored_count: cached.length,
    });
    return cached;
  }

  const model = agenticLanguageModel();
  const system =
    "Classify OCR text blocks against a semantic description. Return confidence 0..1 per block.";
  const prompt = {
    description: args.description,
    items: args.items.slice(0, 300).map((it) => ({
      ocr_index: it.ocr_index,
      type: it.type,
      text: it.text,
      confidence: it.confidence,
    })),
  };
  const promptJson = JSON.stringify(prompt);

  args.runLog?.write({
    kind: "semantic_request",
    frame_index: args.frameIndex,
    model: agenticModelSlug(),
    description: args.description,
    item_count: prompt.items.length,
    system_prompt: system,
    user_text: promptJson,
  });

  const t0 = Date.now();
  try {
    const result = await generateObject({
      model,
      schema: SemanticClassifierSchema,
      providerOptions: agenticProviderOptions(),
      system,
      prompt: promptJson,
    });
    const scored = result.object.scores
      .filter((s) => Number.isFinite(s.confidence))
      .map((s) => ({
        ocr_index: s.ocr_index,
        confidence: Math.max(0, Math.min(1, s.confidence)),
      }));
    _semanticCache.set(key, scored);
    args.runLog?.write({
      kind: "semantic_response",
      frame_index: args.frameIndex,
      description: args.description,
      elapsed_ms: Date.now() - t0,
      finish_reason: result.finishReason,
      warnings: result.warnings,
      usage: result.usage,
      scored: compact(scored),
      scored_count: scored.length,
    });
    return scored;
  } catch (e) {
    const empty: Array<{ ocr_index: number; confidence: number }> = [];
    _semanticCache.set(key, empty);
    args.runLog?.write({
      kind: "semantic_error",
      frame_index: args.frameIndex,
      description: args.description,
      elapsed_ms: Date.now() - t0,
      error: e instanceof Error ? e.message : String(e),
    });
    return empty;
  }
}

export async function filterOcr(args: {
  ctx: PromptToolContext;
  blockTypes?: Array<"WORD" | "LINE">;
  regionBbox?: [number, number, number, number];
  textSubstring?: string;
  semantic?: { description: string; min_confidence?: number };
  minOcrConfidence?: number;
  maxItems?: number;
}): Promise<GetOcrItem[]> {
  const blockTypes = args.blockTypes ?? ["WORD", "LINE"];
  const wantWord = blockTypes.includes("WORD");
  const wantLine = blockTypes.includes("LINE");
  const minOcrConfidence = args.minOcrConfidence ?? 0;
  const maxItems = Math.max(1, args.maxItems ?? 200);
  const items = toRawItems(args.ctx.rawOcr).filter((it) => {
    if (it.type === "word" && !wantWord) return false;
    if (it.type === "line" && !wantLine) return false;
    if (it.confidence < minOcrConfidence) return false;
    return true;
  });

  let filtered = items;
  if (args.regionBbox) {
    filtered = filtered.filter((it) => intersects(it.bbox, args.regionBbox!));
  }

  if (args.textSubstring && args.textSubstring.trim()) {
    const needle = normalize(args.textSubstring);
    filtered = filtered.filter((it) => normalize(it.text).includes(needle));
  }

  let semanticScores: Map<number, number> | null = null;
  if (args.semantic && args.semantic.description.trim()) {
    const minScore =
      args.semantic.min_confidence ?? args.ctx.semanticMinConfidenceDefault ?? 0.6;
    const scored = await classifySemantic({
      videoHash: args.ctx.videoHash,
      frameIndex: args.ctx.frameIndex,
      description: args.semantic.description,
      items: filtered,
      runLog: args.ctx.runLog ?? null,
    });
    semanticScores = new Map(scored.map((s) => [s.ocr_index, s.confidence]));
    filtered = filtered.filter((it) => (semanticScores?.get(it.ocr_index) ?? 0) >= minScore);
  }

  return filtered.slice(0, maxItems).map((it) => ({
    ocr_index: it.ocr_index,
    type: it.type,
    text: it.text,
    confidence: it.confidence,
    bbox_2d: it.bbox,
    semantic_score: semanticScores?.get(it.ocr_index),
  }));
}

function upsertInstance(list: Instance[], candidate: Instance): void {
  if (list.some((i) => i.id === candidate.id)) return;
  list.push(candidate);
}

export function createPromptTools(ctx: PromptToolContext): {
  tools: Record<string, any>;
  getMutation: () => PromptToolMutation;
} {
  const mutation: PromptToolMutation = {
    kept: [],
    added: [],
    dropped: [],
    instances: [],
    finish_summary: null,
    any_more_to_check: false,
    still_visible: null,
  };

  const rawItems = toRawItems(ctx.rawOcr);

  const logTool = (tool_name: string, step: number, phase: "call" | "result", payload: unknown): void => {
    ctx.runLog?.write({
      kind: `curator_tool_${phase}`,
      frame_index: ctx.frameIndex,
      tool_name,
      step,
      payload: compact(payload),
    });
  };

  let toolStep = 0;
  const nextStep = (): number => ++toolStep;

  const localizeRegionTool = tool({
    description: "Localize a described region on the current frame.",
    inputSchema: z.object({
      frame_index: z.number().int(),
      description: z.string(),
      all_instances: z.boolean().default(false),
      branch: z.string().nullable().optional(),
    }),
    execute: async ({ frame_index, description, all_instances, branch }) => {
      const step = nextStep();
      logTool("localize_region", step, "call", { frame_index, description, all_instances, branch });
      if (frame_index !== ctx.frameIndex) {
        const result = {
          ok: false as const,
          error: `Only frame ${ctx.frameIndex} is available in this tool context.`,
        };
        logTool("localize_region", step, "result", result);
        return result;
      }
      const leaf: Extract<PredicateLeaf, { kind: "region" }> = {
        kind: "region",
        description,
        all_instances,
        branch: branch ?? undefined,
      };
      const regions = await localizeRegionForFrame({
        videoHash: ctx.videoHash,
        frameIndex: ctx.frameIndex,
        leaf,
        frameBlob: ctx.frameBlob,
        runLog: ctx.runLog ?? null,
      });
      const result = {
        ok: true as const,
        regions: regions.map((r) => ({
          sub_id: r.sub_id,
          bbox_2d: r.bbox_2d,
          confidence: r.confidence,
          reason: r.reason,
        })),
      };
      logTool("localize_region", step, "result", result);
      return result;
    },
  });

  const getOcrTool = tool({
    description:
      "Fetch OCR blocks filtered by region, substring and/or semantic description. When `precomputed_regions` are provided in the user prompt for this frame, you MUST narrow this call with one of `region_bbox` / `text_substring` / `semantic` — a bare whole-frame fetch on a cluttered screen returns 200+ mixed items (browser chrome, dock, tabs) and downstream `keep_box` selection becomes unreliable.",
    inputSchema: z.object({
      frame_index: z.number().int(),
      block_types: z.array(z.enum(["WORD", "LINE"])) .optional(),
      region_bbox: BboxSchema.optional(),
      text_substring: z.string().optional(),
      semantic: z
        .object({
          description: z.string(),
          min_confidence: z.number().min(0).max(1).optional(),
        })
        .optional(),
      min_ocr_confidence: z.number().min(0).max(100).optional(),
      max_items: z.number().int().min(1).max(500).optional(),
    }),
    execute: async (input) => {
      const step = nextStep();
      logTool("get_ocr", step, "call", input);
      if (input.frame_index !== ctx.frameIndex) {
        const result = {
          ok: false as const,
          error: `Only frame ${ctx.frameIndex} is available in this tool context.`,
        };
        logTool("get_ocr", step, "result", result);
        return result;
      }
      // Gate bare whole-frame calls when the caller has focused regions
      // available. The failure pattern on frame 18 (ztak run) was:
      //   step 1 get_ocr({frame_index}) → 200 items
      //   step 2..N add_box(<region_bbox>, <ocr_text_from_dump>)
      // because once 200 noisy items are in context the model shortcuts
      // past keep_box and echoes the region inputs. Force a spatial
      // filter so get_ocr returns a tractable slice and the OCR items
      // survive to keep_box selection.
      const hasSpatialFilter =
        input.region_bbox != null ||
        (input.text_substring != null && input.text_substring.trim() !== "") ||
        (input.semantic != null && input.semantic.description.trim() !== "");
      if (!hasSpatialFilter && ctx.regions.length > 0) {
        const hint = ctx.regions.map((r) => ({
          sub_id: r.sub_id,
          bbox_2d: r.bbox_2d,
        }));
        const result = {
          ok: false as const,
          error:
            `This frame has ${ctx.regions.length} precomputed region(s); please re-call get_ocr with ` +
            `\`region_bbox\` set to one of their bboxes (or to a bbox covering all of them). A bare ` +
            `whole-frame fetch returns 200+ OCR items spanning browser chrome, tabs and the dock, ` +
            `which makes keep_box selection error-prone.`,
          precomputed_regions: hint,
        };
        ctx.runLog?.write({
          kind: "curator_get_ocr_gated",
          frame_index: ctx.frameIndex,
          step,
          reason: "bare_call_with_regions",
          region_count: ctx.regions.length,
        });
        logTool("get_ocr", step, "result", result);
        return result;
      }
      const items = await filterOcr({
        ctx,
        blockTypes: input.block_types,
        regionBbox: input.region_bbox as [number, number, number, number] | undefined,
        textSubstring: input.text_substring,
        semantic: input.semantic,
        minOcrConfidence: input.min_ocr_confidence,
        maxItems: input.max_items,
      });
      const result = {
        ok: true as const,
        items,
      };
      logTool("get_ocr", step, "result", { item_count: items.length, items });
      return result;
    },
  });

  const keepBoxTool = tool({
    description:
      "Adopt a raw OCR block as a redaction hit. You MUST label `text_color_hex` + `background_color_hex` — dominant hex RGB colors of the text glyphs and the surrounding background (e.g. '#111111' / '#ffffff'). These feed the downstream pixel-level shrink pass that tightens boxes whose corners poke into margin.",
    inputSchema: z.object({
      ocr_index: z.number().int(),
      branch: z.string(),
      instance_id: z.string(),
      category: z.string().nullable().optional(),
      text_color_hex: z
        .string()
        .describe("Dominant 6-digit hex RGB of the text glyphs (e.g. '#111111')."),
      background_color_hex: z
        .string()
        .describe("Dominant 6-digit hex RGB of the surrounding background (e.g. '#ffffff')."),
      reason: z.string().optional(),
    }),
    execute: async ({
      ocr_index,
      branch,
      instance_id,
      category,
      text_color_hex,
      background_color_hex,
      reason,
    }) => {
      const step = nextStep();
      logTool("keep_box", step, "call", {
        ocr_index,
        branch,
        instance_id,
        category,
        text_color_hex,
        background_color_hex,
        reason,
      });
      const src = rawItems.find((r) => r.ocr_index === ocr_index);
      if (!src) {
        const result = { ok: false as const, error: `No OCR item at index ${ocr_index}.` };
        logTool("keep_box", step, "result", result);
        return result;
      }
      const box = toServerBox(src.bbox, src.text, ctx.width, ctx.height, Math.min(1, src.confidence / 100));
      box.instance_id = instance_id;
      box.branch = branch;
      box.category = category ?? null;
      box.text_color_hex = normalizeHexColor(text_color_hex);
      box.background_color_hex = normalizeHexColor(background_color_hex);
      mutation.kept.push(box);
      upsertInstance(mutation.instances, {
        id: instance_id,
        branch,
        descriptor: `${category ?? "value"}: ${src.text}`,
        category: category ?? null,
      });
      const result = { ok: true as const, kept: true };
      logTool("keep_box", step, "result", {
        ...result,
        text: src.text,
        bbox_2d: src.bbox,
      });
      return result;
    },
  });

  const addBoxTool = tool({
    description:
      "Add a new redaction box using explicit bbox_2d coordinates. You MUST label `text_color_hex` + `background_color_hex` — dominant hex RGB colors of the text glyphs and the surrounding background (e.g. '#111111' / '#ffffff'). These feed the downstream pixel-level shrink pass.",
    inputSchema: z.object({
      text: z.string(),
      bbox_2d: BboxSchema,
      branch: z.string(),
      instance_id: z.string(),
      category: z.string().nullable().optional(),
      text_color_hex: z
        .string()
        .describe("Dominant 6-digit hex RGB of the text glyphs (e.g. '#111111')."),
      background_color_hex: z
        .string()
        .describe("Dominant 6-digit hex RGB of the surrounding background (e.g. '#ffffff')."),
      reason: z.string().optional(),
    }),
    execute: async ({
      text,
      bbox_2d,
      branch,
      instance_id,
      category,
      text_color_hex,
      background_color_hex,
      reason,
    }) => {
      const step = nextStep();
      logTool("add_box", step, "call", {
        text,
        bbox_2d,
        branch,
        instance_id,
        category,
        text_color_hex,
        background_color_hex,
        reason,
      });
      const box = toServerBox(
        bbox_2d as [number, number, number, number],
        text,
        ctx.width,
        ctx.height,
        1,
      );
      box.origin = "fix";
      box.instance_id = instance_id;
      box.branch = branch;
      box.category = category ?? null;
      box.text_color_hex = normalizeHexColor(text_color_hex);
      box.background_color_hex = normalizeHexColor(background_color_hex);
      mutation.added.push(box);
      upsertInstance(mutation.instances, {
        id: instance_id,
        branch,
        descriptor: `${category ?? "value"}: ${text}`,
        category: category ?? null,
      });
      const result = { ok: true as const, added: true };
      logTool("add_box", step, "result", result);
      return result;
    },
  });

  const dropBoxTool = tool({
    description: "Explicitly drop an OCR candidate index.",
    inputSchema: z.object({
      ocr_index: z.number().int(),
      reason: z.string().optional(),
    }),
    execute: async ({ ocr_index, reason }) => {
      const step = nextStep();
      logTool("drop_box", step, "call", { ocr_index, reason });
      if (!mutation.dropped.includes(ocr_index)) mutation.dropped.push(ocr_index);
      const result = { ok: true as const, dropped: true };
      logTool("drop_box", step, "result", result);
      return result;
    },
  });

  const finishTool = tool({
    description: "Finish the current frame reasoning loop.",
    inputSchema: z.object({
      summary: z.string(),
      any_more_to_check: z.boolean().optional(),
      still_visible: z.boolean().optional(),
    }),
    execute: async ({ summary, any_more_to_check, still_visible }) => {
      const step = nextStep();
      logTool("finish", step, "call", { summary, any_more_to_check, still_visible });
      mutation.finish_summary = summary;
      mutation.any_more_to_check = Boolean(any_more_to_check);
      mutation.still_visible =
        typeof still_visible === "boolean" ? still_visible : mutation.still_visible;
      const result = { ok: true as const };
      logTool("finish", step, "result", result);
      return result;
    },
  });

  return {
    tools: {
      localize_region: localizeRegionTool,
      get_ocr: getOcrTool,
      keep_box: keepBoxTool,
      add_box: addBoxTool,
      drop_box: dropBoxTool,
      finish: finishTool,
    },
    getMutation: () => mutation,
  };
}

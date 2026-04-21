import { createHash } from "node:crypto";
import { generateText, stepCountIs, type LanguageModelUsage } from "ai";
import { extractRawOcrItems } from "@/lib/server/agentic-ocr";
import {
  agenticLanguageModel,
  agenticModelSlug,
  agenticProviderOptions,
  agenticThinkingLevel,
  bboxToPixels,
  type ServerBox,
} from "@/lib/server/openrouter";
import { compact, type RunLog } from "@/lib/server/run-log";
import { coalesceAdjacentBoxes } from "./coalesce";
import { filterOcr, createPromptTools } from "./tools";
import { collectLeafPredicates, type Instance, type Predicate, type RegionBbox, type ResolvedCandidate } from "./types";

function parseIntEnv(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : fallback;
}

function parseBoolEnv(name: string, fallback: boolean): boolean {
  const raw = (process.env[name] ?? "").trim().toLowerCase();
  if (!raw) return fallback;
  return !(raw === "0" || raw === "false" || raw === "off");
}

const PROMPT_CURATOR_MAX_STEPS = parseIntEnv("PROMPT_CURATOR_MAX_STEPS", 6);
const PROMPT_CURATOR_TEXT_FASTPATH = parseBoolEnv(
  "PROMPT_CURATOR_TEXT_FASTPATH",
  true,
);

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function textInstanceId(branch: string, text: string): string {
  return `${branch}:${normalize(text)}`;
}

function semanticInstanceId(branch: string, category: string, text: string): string {
  const hash = createHash("sha256")
    .update(`${category}:${normalize(text)}`)
    .digest("hex")
    .slice(0, 12);
  return `${branch}:${hash}`;
}

function regionInstanceId(branch: string, subId: string): string {
  return `${branch}:${subId}`;
}

function asPixelBox(args: {
  bbox: [number, number, number, number];
  text: string;
  width: number;
  height: number;
  score: number;
  branch: string;
  instanceId: string;
  category?: string | null;
  textColorHex?: string | null;
  backgroundColorHex?: string | null;
}): ServerBox {
  const rect = bboxToPixels(args.bbox, args.width, args.height);
  return {
    x: rect.x,
    y: rect.y,
    w: rect.w,
    h: rect.h,
    text: args.text,
    score: args.score,
    branch: args.branch,
    instance_id: args.instanceId,
    category: args.category ?? null,
    text_color_hex: args.textColorHex ?? undefined,
    background_color_hex: args.backgroundColorHex ?? undefined,
  };
}

function systemForPredicate(predicate: Predicate): string {
  const leaves = collectLeafPredicates(predicate);
  const lines: string[] = [
    "You are a prompt-mode redaction curator.",
    "Use tools to gather OCR evidence and emit keep/add/drop mutations.",
    "Call finish when done.",
    "Use instance_id exactly as provided in your tool calls.",
    "",
    "TOOL PRIORITY — keep_box over add_box. `keep_box` adopts an OCR block's pixel-accurate bbox; `add_box` paints an arbitrary rectangle you specify. Pixel-tight coverage matters: loose boxes flicker visibly in the exported video. When `precomputed_regions` are provided, for EACH region look up the OCR LINE item whose bbox falls inside that region and call `keep_box` on it — do NOT copy the region's bbox into `add_box`. The region bbox is a COARSE hint from an upstream localizer; the OCR LINE bbox is the ground-truth pixel rectangle around the glyphs. Only use `add_box` as a true OCR-miss fallback: the text is visibly in the frame (you can read it in the image) but `get_ocr` returned no item covering it. If you are tempted to `add_box` with coordinates that match a `precomputed_regions[i].bbox_2d` byte-for-byte, stop and re-check with a narrower `get_ocr` call first — the OCR block you need is almost certainly there.",
    "",
    "LINE vs WORD blocks: OCR returns both LINE-level and WORD-level items for the same visible text. When every word of a LINE belongs to the redaction, call `keep_box` ONCE on the LINE-level item — do NOT stack a `keep_box` on each child WORD. Only pick WORD-level items when you are redacting a proper subset of the line (e.g. one name inside a longer sentence). The post-processor will merge adjacent kept WORDs on the same line anyway, but picking the LINE directly is cheaper and gives you tighter coords.",
    "",
    "COLOR LABELING: for EVERY `keep_box` and `add_box` call, you MUST include `text_color_hex` (dominant color of the text glyphs, as 6-digit hex RGB like '#111111') AND `background_color_hex` (dominant color of the surrounding / inter-glyph pixels, like '#ffffff'). These drive a deterministic pixel-level shrink pass that trims box sides whose corners poke into margin. Guessing wrong makes the pass over-shrink or no-op; be deliberate and pick the most prevalent color by area.",
    "",
    "MOUSE CURSOR obstruction: if the mouse pointer / text cursor / I-beam / hand cursor partially covers a sensitive text region, KEEP the full OCR box over the whole text — do NOT shrink it to the visible sliver and do NOT split it into boxes around the cursor. It is FINE for the mouse cursor to get redacted along with the text (a few harmless pixels, not a privacy leak). This exception applies ONLY to the mouse cursor; for real UI surfaces (popups, modals, dropdowns, tooltips, menus, autocomplete, other windows) continue to redact only the visible unoccluded pixels.",
  ];
  for (const leaf of leaves) {
    if (leaf.kind === "text") {
      lines.push(
        `Leaf ${leaf.branch ?? "?"}: text '${leaf.text}' fuzzy=${leaf.fuzzy}.`,
      );
    } else if (leaf.kind === "region") {
      lines.push(
        `Leaf ${leaf.branch ?? "?"}: region '${leaf.description}', all_instances=${leaf.all_instances}.`,
      );
    } else {
      lines.push(
        `Leaf ${leaf.branch ?? "?"}: semantic ${leaf.category} - ${leaf.description}.`,
      );
    }
  }
  return lines.join("\n");
}

function makeInstances(boxes: ServerBox[]): Instance[] {
  const seen = new Set<string>();
  const out: Instance[] = [];
  for (const b of boxes) {
    if (!b.instance_id || !b.branch) continue;
    if (seen.has(b.instance_id)) continue;
    seen.add(b.instance_id);
    out.push({
      id: b.instance_id,
      branch: b.branch,
      descriptor: `${b.category ?? "value"}: ${b.text}`,
      category: b.category ?? null,
    });
  }
  return out;
}

export type CuratorResult = {
  kept: ServerBox[];
  added: ServerBox[];
  dropped: number[];
  instances: Instance[];
  raw: unknown;
  usage: LanguageModelUsage | null;
};

function singleTextLeaf(predicate: Predicate): Extract<ReturnType<typeof collectLeafPredicates>[number], { kind: "text" }> | null {
  const leaves = collectLeafPredicates(predicate);
  if (leaves.length !== 1) return null;
  const [leaf] = leaves;
  return leaf.kind === "text" ? leaf : null;
}

async function deterministicFallback(args: {
  videoHash: string;
  frameIndex: number;
  predicate: Predicate;
  rawOcr: unknown;
  width: number;
  height: number;
  regions: RegionBbox[];
  textCandidates: ResolvedCandidate[];
  runLog?: RunLog | null;
}): Promise<{ kept: ServerBox[]; added: ServerBox[]; dropped: number[] }> {
  const rawItems = extractRawOcrItems(args.rawOcr);
  const kept: ServerBox[] = [];
  const dropped: number[] = [];

  // Text leaves from resolver candidates.
  for (const c of args.textCandidates) {
    const leafText = rawItems[c.ocr_index]?.text ?? c.text;
    kept.push(
      asPixelBox({
        bbox: c.bbox,
        text: leafText,
        width: args.width,
        height: args.height,
        score: c.prior_score,
        branch: c.branch,
        instanceId: textInstanceId(c.branch, leafText),
      }),
    );
  }

  // Region leaves: keep every OCR block inside each localized region.
  const leaves = collectLeafPredicates(args.predicate);
  for (const leaf of leaves) {
    if (leaf.kind !== "region") continue;
    const branch = leaf.branch ?? "L0";
    const branchRegions = args.regions.filter((r) => r.branch === branch);
    args.runLog?.write({
      kind: "deterministic_fallback_region_leaf",
      frame_index: args.frameIndex,
      branch,
      region_count: branchRegions.length,
      regions: compact(branchRegions),
    });
    for (const region of branchRegions) {
      const items = await filterOcr({
        ctx: {
          videoHash: args.videoHash,
          frameIndex: args.frameIndex,
          frameBlob: new Uint8Array(),
          width: args.width,
          height: args.height,
          rawOcr: args.rawOcr,
          predicate: args.predicate,
          regions: args.regions,
          textCandidates: args.textCandidates,
          runLog: args.runLog ?? null,
        },
        regionBbox: region.bbox_2d,
        maxItems: 300,
      });
      const iid = regionInstanceId(branch, region.sub_id);
      args.runLog?.write({
        kind: "deterministic_fallback_region_items",
        frame_index: args.frameIndex,
        branch,
        sub_id: region.sub_id,
        bbox_2d: region.bbox_2d,
        item_count: items.length,
        items: compact(items),
      });
      for (const item of items) {
        kept.push(
          asPixelBox({
            bbox: item.bbox_2d,
            text: item.text,
            width: args.width,
            height: args.height,
            score: Math.min(1, item.confidence / 100),
            branch,
            instanceId: iid,
            category: leaf.category ?? null,
          }),
        );
      }
    }
  }

  // Semantic leaves: semantic filter over full frame.
  for (const leaf of leaves) {
    if (leaf.kind !== "semantic") continue;
    const branch = leaf.branch ?? "L0";
    const items = await filterOcr({
      ctx: {
        videoHash: args.videoHash,
        frameIndex: args.frameIndex,
        frameBlob: new Uint8Array(),
        width: args.width,
        height: args.height,
        rawOcr: args.rawOcr,
        predicate: args.predicate,
        regions: args.regions,
        textCandidates: args.textCandidates,
        runLog: args.runLog ?? null,
      },
      semantic: { description: leaf.description, min_confidence: 0.6 },
      maxItems: 300,
    });
    args.runLog?.write({
      kind: "deterministic_fallback_semantic_leaf",
      frame_index: args.frameIndex,
      branch,
      category: leaf.category,
      description: leaf.description,
      item_count: items.length,
      items: compact(items),
    });
    for (const item of items) {
      kept.push(
        asPixelBox({
          bbox: item.bbox_2d,
          text: item.text,
          width: args.width,
          height: args.height,
          score: item.semantic_score ?? Math.min(1, item.confidence / 100),
          branch,
          instanceId: semanticInstanceId(branch, leaf.category, item.text),
          category: leaf.category,
        }),
      );
    }
  }

  return { kept, added: [], dropped };
}

export async function curateFramePrompt(args: {
  videoHash: string;
  frameIndex: number;
  jpeg: Uint8Array;
  width: number;
  height: number;
  predicate: Predicate;
  regions: RegionBbox[];
  textCandidates: ResolvedCandidate[];
  rawOcr: unknown;
  runLog?: RunLog | null;
}): Promise<CuratorResult> {
  const runLog = args.runLog ?? null;
  const leaf = singleTextLeaf(args.predicate);
  if (leaf && leaf.fuzzy && PROMPT_CURATOR_TEXT_FASTPATH) {
    const rawItems = extractRawOcrItems(args.rawOcr);
    const kept = args.textCandidates.map((c) => {
      const text = rawItems[c.ocr_index]?.text ?? c.text;
      return asPixelBox({
        bbox: c.bbox,
        text,
        width: args.width,
        height: args.height,
        score: c.prior_score,
        branch: c.branch,
        instanceId: textInstanceId(c.branch, text),
      });
    });
    const instances = makeInstances(kept);
    runLog?.write({
      kind: "curator_fastpath",
      frame_index: args.frameIndex,
      branch: leaf.branch ?? "L0",
      text: leaf.text,
      fuzzy: leaf.fuzzy,
      kept_count: kept.length,
      kept_texts: kept.map((b) => b.text),
    });
    return {
      kept,
      added: [],
      dropped: [],
      instances,
      raw: { mode: "text_fastpath", kept: kept.length },
      usage: null,
    };
  }

  const runtime = createPromptTools({
    videoHash: args.videoHash,
    frameIndex: args.frameIndex,
    frameBlob: args.jpeg,
    width: args.width,
    height: args.height,
    rawOcr: args.rawOcr,
    predicate: args.predicate,
    regions: args.regions,
    textCandidates: args.textCandidates,
    runLog,
  });

  const model = agenticLanguageModel();
  const systemPrompt = systemForPredicate(args.predicate);
  const userText = JSON.stringify(
    {
      frame_index: args.frameIndex,
      precomputed_regions: args.regions,
      text_candidates: args.textCandidates,
      hint:
        "Use keep_box for OCR indices, add_box for missing bboxes, and finish when done.",
    },
    null,
    2,
  );

  runLog?.write({
    kind: "curator_request",
    frame_index: args.frameIndex,
    model: agenticModelSlug(),
    thinking_level: agenticThinkingLevel(),
    predicate: compact(args.predicate),
    region_count: args.regions.length,
    text_candidate_count: args.textCandidates.length,
    regions: compact(args.regions),
    text_candidates: compact(args.textCandidates),
    jpeg_bytes: args.jpeg.byteLength,
    max_steps: PROMPT_CURATOR_MAX_STEPS,
    system_prompt: systemPrompt,
    user_text: userText,
  });

  let usage: LanguageModelUsage | null = null;
  let raw: unknown = null;
  let finishReason: string | null = null;
  let warnings: unknown = null;
  let generateError: string | null = null;

  const t0 = Date.now();
  try {
    const r = await generateText({
      model,
      providerOptions: agenticProviderOptions(),
      system: systemPrompt,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: userText,
            },
            { type: "image", image: args.jpeg, mediaType: "image/jpeg" },
          ],
        },
      ],
      tools: runtime.tools,
      stopWhen: stepCountIs(PROMPT_CURATOR_MAX_STEPS),
    });
    usage = r.usage ?? null;
    raw = r.text;
    finishReason = r.finishReason ?? null;
    warnings = r.warnings ?? null;
  } catch (e) {
    generateError = e instanceof Error ? e.message : String(e);
    runLog?.write({
      kind: "curator_error",
      frame_index: args.frameIndex,
      elapsed_ms: Date.now() - t0,
      error: generateError,
    });
  }

  const mutation = runtime.getMutation();

  runLog?.write({
    kind: "curator_response",
    frame_index: args.frameIndex,
    elapsed_ms: Date.now() - t0,
    finish_reason: finishReason,
    warnings: warnings,
    usage,
    generate_error: generateError,
    model_text: typeof raw === "string" ? compact(raw) : raw,
    tool_finish_summary: mutation.finish_summary,
    kept_count: mutation.kept.length,
    added_count: mutation.added.length,
    dropped_count: mutation.dropped.length,
    kept_texts: mutation.kept.map((b) => b.text),
    added_texts: mutation.added.map((b) => b.text),
  });

  let kept = mutation.kept;
  let added = mutation.added;
  let dropped = mutation.dropped;
  let usedDeterministicFallback = false;

  if (kept.length === 0 && added.length === 0) {
    usedDeterministicFallback = true;
    runLog?.write({
      kind: "curator_fallback_deterministic_start",
      frame_index: args.frameIndex,
      reason:
        generateError != null
          ? "generateText threw"
          : "tool loop produced zero mutations",
    });
    const det = await deterministicFallback({ ...args, runLog });
    kept = det.kept;
    added = det.added;
    dropped = det.dropped;
    raw = {
      mode: "deterministic_fallback",
      finish_summary: mutation.finish_summary,
      tool_output: raw,
    };
    runLog?.write({
      kind: "curator_fallback_deterministic_end",
      frame_index: args.frameIndex,
      kept_count: kept.length,
      added_count: added.length,
      dropped_count: dropped.length,
    });
  }

  // Collapse WORD-level fragments the curator picked when a LINE would
  // do. Pure geometry: adjacent boxes on the same baseline merge into
  // one; isolated boxes pass through. When the merged span matches an
  // OCR LINE bbox (IoU ≥ 0.6) we snap to the LINE for pixel-accurate
  // coords. See `coalesce.ts`. Kept and added are coalesced separately
  // so a kept OCR box and a fix-origin add_box don't accidentally fold
  // across origins.
  const keptCoalesce = coalesceAdjacentBoxes({
    boxes: kept,
    rawOcr: args.rawOcr,
    frameWidth: args.width,
    frameHeight: args.height,
  });
  const addedCoalesce = coalesceAdjacentBoxes({
    boxes: added,
    rawOcr: args.rawOcr,
    frameWidth: args.width,
    frameHeight: args.height,
  });
  kept = keptCoalesce.boxes;
  added = addedCoalesce.boxes;
  const coalesceCollapsed =
    keptCoalesce.stats.input -
    keptCoalesce.stats.output +
    (addedCoalesce.stats.input - addedCoalesce.stats.output);
  if (coalesceCollapsed > 0) {
    runLog?.write({
      kind: "curator_coalesce",
      frame_index: args.frameIndex,
      kept: keptCoalesce.stats,
      added: addedCoalesce.stats,
      collapsed_total: coalesceCollapsed,
    });
  }

  const instances = [
    ...mutation.instances,
    ...makeInstances([...kept, ...added]),
  ].filter(
    (inst, idx, arr) => arr.findIndex((i) => i.id === inst.id) === idx,
  );

  runLog?.write({
    kind: "curator_resolved",
    frame_index: args.frameIndex,
    used_deterministic_fallback: usedDeterministicFallback,
    kept_count: kept.length,
    added_count: added.length,
    dropped_count: dropped.length,
    instance_count: instances.length,
  });

  return {
    kept,
    added,
    dropped,
    instances,
    raw,
    usage,
  };
}

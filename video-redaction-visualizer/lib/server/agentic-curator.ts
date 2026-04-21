// Agentic first-pass curator (Gemini 3 Flash via the Google AI SDK).
//
// For each frame, we already ran OCR (Python/Textract). The curator hands
// the frame image plus every OCR-proposed box to Gemini 3 and asks it to
// make two decisions:
//
//   1. For each OCR box: KEEP (it does redact some instance of the query)
//      or DROP (the OCR box is a false positive / unrelated text).
//   2. Any additional boxes the OCR missed — Gemini can list new
//      rectangles with its own `bbox` coords.
//
// The box contract is Gemini's native box_2d format:
//   bbox = [y_min, x_min, y_max, x_max] in integer 0..1000 (top-left origin)
// Gemini was post-trained on this exact shape, so emitting and consuming
// boxes in this order maximizes localization accuracy.
//
// Each curator invocation is independent, so the caller fans these out
// across frames with `AGENTIC_CURATOR_CONCURRENCY`. No cross-frame
// labeling happens here; that's the navigator's job.
//
// Code execution is NOT enabled on the curator path — it's a pure
// structured-output task and the extra tool turns would just add latency.
// Cascade + navigator get code execution for per-box investigation.

import { generateObject, type LanguageModelUsage } from "ai";
import { z } from "zod";
import { aerr, alog } from "./agentic-log";
import { geminiCost } from "./cost";
import {
  extractRawOcrItems,
  filterAndRankOcrItems,
  formatOcrItemsForPrompt,
} from "./agentic-ocr";
import {
  BOX_FORMAT_NOTE,
  agenticLanguageModel,
  agenticModelId,
  agenticModelSlug,
  agenticProviderOptions,
  agenticThinkingLevel,
  bboxToPixels,
  pixelBoxToNormalizedBbox,
  type ServerBox,
} from "./openrouter";
import { compact, type RunLog } from "./run-log";

// 4-integer bbox expressed as a fixed-length homogeneous array. This shape
// passes Gemini's response_json_schema validation cleanly (tuple /
// `prefixItems` is rejected). Range is 0..1000 to match Gemini's trained
// box_2d output scale.
const Bbox = z
  .array(z.number().int().min(0).max(1000))
  .min(4)
  .max(4);

// Nullable (rather than optional) to keep the schema strict: Structured
// Outputs require every property to appear in `required`. Optional fields
// drop from `required`; nullable fields stay there and the model can
// return `null` when it has nothing to contribute.
const DecisionSchema = z.object({
  ocr_index: z.number().int().nonnegative(),
  keep: z.boolean(),
  text: z.string().nullable(),
  reason: z.string().nullable(),
});

const AdditionSchema = z.object({
  bbox: Bbox,
  text: z.string(),
  reason: z.string().nullable(),
});

const CuratorOutputSchema = z.object({
  decisions: z.array(DecisionSchema),
  additions: z.array(AdditionSchema),
});

export type CuratorDecision = z.infer<typeof DecisionSchema>;
export type CuratorAddition = z.infer<typeof AdditionSchema>;

export type CuratorResult = {
  kept: ServerBox[];
  added: ServerBox[];
  dropped: number[];
  raw: unknown;
  /** AI SDK token usage for the single Gemini call. Fed into cost.ts. */
  usage: LanguageModelUsage | null;
};

function system(): string {
  return [
    "You are reviewing OCR-proposed redaction boxes for a SCREEN-RECORDING REDACTION pipeline.",
    "The goal is to hide every pixel that could leak the sensitive query text, INCLUDING partial reveals. This is a security task, not a transcription task.",
    "",
    "REDACTION SEMANTICS — read carefully:",
    "- Any visible substring, prefix, suffix, partial, case-variant, or OCR-garbled version of the query is SENSITIVE and MUST be redacted.",
    '- Concrete example: if the query is "test word" and the frame shows "test wor", "tes", "t word", "TEST WORD", or "testw0rd", ALL of those must be covered.',
    "- Partial reveals across adjacent frames let a viewer reassemble the full sensitive text. A box labeled with a partial string is NOT a reason to drop it — it is the exact reason to keep it.",
    "- When in doubt, KEEP. Over-redaction is safe; under-redaction is a leak. Never drop a box just because what it covers is 'incomplete'.",
    "",
    "OCCLUSION SLIVERS (popups, modals, dropdowns, tooltips, menus):",
    "- A popup/modal/dropdown/tooltip that opens OVER a cell of sensitive text can leave a thin sliver of that text visible at the right, left, top, or bottom EDGE of the popup. Even 1 or 2 characters of the query text remaining visible is a LEAK — a viewer can sometimes reassemble the full string from such slivers across frames.",
    "- Signals that a popup is present on this frame: many short text labels clustered together (font names, menu items, filter options), a rectangular band of text inserted mid-frame that isn't in the surrounding spreadsheet/document content.",
    "- When you notice a popup signature, LOOK CAREFULLY in the raw-OCR dump for SHORT blocks (1-3 characters) whose text is any substring of the query and whose y-coordinate roughly matches where sensitive text normally sits in the layout. Those are almost certainly the uncovered edge of the query text peeking past the popup. ADD a redaction box for EACH such sliver using the raw-OCR block's exact bbox.",
    '- Concrete example: if the query is "test word" and a dropdown covers most of a "test wor" cell, OCR might report just the word "or" (2 chars) with a tiny bbox at the right edge of the popup region. That "or" block IS the sliver — add a redaction for it.',
    "- A 2-character raw-OCR block that is a query substring is a HIGH-PRIORITY addition target, NOT a noise block to ignore — especially when neighboring frames (without the popup) have a larger OCR candidate at the same y-coordinate.",
    "",
    "DECISIONS (strict definitions):",
    "- keep=true → the OCR rectangle covers visible pixels that are the query OR any substring / prefix / suffix / case-variant / OCR-garbled / fuzzy version of it. These pixels must be redacted.",
    "- keep=false → ONLY if the rectangle covers text that is clearly unrelated to the query (no shared visible characters with it). Example: query is 'apple' but OCR flagged a box over the unrelated word 'orange' — drop.",
    "- A box that is wider than ideal (covers the query PLUS surrounding unrelated chars) should still be KEPT. Do not drop it to replace it with a tighter one — instead keep the wider box, and optionally add a tighter box in `additions`. The wider redaction is safe.",
    "",
    "ADDITIONS — critical rule about coordinates:",
    "- You are also given a RAW OCR dump for this frame: every WORD and LINE block Textract emitted, INCLUDING blocks that did NOT make it into the pre-filtered candidate list above. Each block has exact pixel-accurate coords.",
    "- When you want to add a redaction box, FIRST look for a raw OCR block whose text matches what you want to cover (case-insensitive substring either direction, or an OCR-garbled variant). If such a block exists, COPY ITS BBOX EXACTLY into your addition. DO NOT re-estimate the coordinates visually — the raw OCR block is pixel-accurate, your visual estimate is not.",
    "- Union multiple raw blocks when the sensitive text spans several — take the minimum y_min/x_min and maximum y_max/x_max across the relevant blocks.",
    "- Only INVENT bbox coordinates as a true last resort: the text is visibly on screen but NO raw OCR block overlaps it (OCR genuinely missed it). State this explicitly in `reason` when you do.",
    "- Additions must tightly wrap just the sensitive pixels, not a whole row.",
    BOX_FORMAT_NOTE,
    "",
    "Return JSON only, matching the provided schema. For decisions where you have no comment, set `reason` and `text` to null.",
  ].join("\n");
}

/**
 * Run the curator on a single frame. Parallel-safe: no cross-frame state.
 */
export async function curateFrame(opts: {
  jpeg: Uint8Array;
  query: string;
  frameIndex: number;
  frameWidth: number;
  frameHeight: number;
  ocrBoxes: ServerBox[];
  /**
   * Full Textract response for this frame. The fuzzy-matched subset lives
   * in `ocrBoxes`; passing the raw blob lets the curator reference
   * pixel-accurate bboxes of blocks that didn't pass the query filter
   * (e.g. standalone substrings like "WORD" when the query is "test word")
   * and copy those coords into `additions` instead of guessing.
   */
  ocrRaw?: unknown;
  /** Optional per-run JSONL log. Curator writes one entry per call. */
  runLog?: RunLog | null;
}): Promise<CuratorResult> {
  const {
    jpeg,
    query,
    frameIndex,
    frameWidth,
    frameHeight,
    ocrBoxes,
    ocrRaw,
    runLog,
  } = opts;
  const model = agenticLanguageModel();

  const ocrSummary =
    ocrBoxes.length === 0
      ? "(OCR returned no candidate boxes for this frame.)"
      : ocrBoxes
          .map((b, i) => {
            const [ymin, xmin, ymax, xmax] = pixelBoxToNormalizedBbox(
              b,
              frameWidth,
              frameHeight,
            );
            return `${i}: text="${b.text}" box_2d=[${ymin}, ${xmin}, ${ymax}, ${xmax}]`;
          })
          .join("\n");

  // Raw OCR dump: every Textract WORD/LINE block, split into "relevant"
  // (likely matches/substrings of the query) and "context" (other lines on
  // the frame, capped). The model is instructed in the system prompt to
  // COPY exact bboxes from this list whenever it wants to add a redaction.
  const rawItems = extractRawOcrItems(ocrRaw);
  const { relevant: relevantRaw, context: contextRaw } = filterAndRankOcrItems(
    rawItems,
    query,
  );
  const relevantRawBlock = formatOcrItemsForPrompt(relevantRaw);
  const contextRawBlock = formatOcrItemsForPrompt(contextRaw);

  const userText = [
    `Frame #${frameIndex}. Sensitive query to redact: "${query}".`,
    "",
    "OCR has proposed these candidate boxes that FUZZY-MATCHED the query (index: recognized text and normalized bbox):",
    ocrSummary,
    "",
    "For EACH candidate, decide keep=true (redact it) or keep=false (drop it).",
    'IMPORTANT: Keep the box if its recognized text is the query OR any substring, prefix, suffix, case-variant, or fuzzy/OCR-garbled form of it. A box labeled "test wor" when the query is "test word" MUST be kept — partial visible text is a leak across frames.',
    "Drop a box ONLY if its recognized text is clearly unrelated to the query (no shared characters, different content entirely).",
    "If a box is wider than needed, still keep it — you can add a tighter box in `additions`, but do NOT drop a wide-but-correct redaction.",
    "",
    "--- RAW OCR on this frame (pixel-accurate bboxes, USE THESE for additions) ---",
    "These are Textract WORD/LINE blocks that did NOT make it into the fuzzy-matched candidate list above but may still be partial / substring / case-variant matches of the sensitive query. Use their bboxes as ground truth.",
    "",
    "Query-relevant raw blocks (highest-priority when adding a box):",
    relevantRawBlock,
    "",
    "Other raw blocks on this frame (context, for disambiguation):",
    contextRawBlock,
    "",
    "Then list any additional tight boxes for visible query instances / partials / variants that neither the fuzzy candidate list nor existing coverage captures.",
    "CRITICAL: for each addition, COPY the bbox directly from the raw OCR block that covers the sensitive text. Do not estimate coordinates from the image when a raw block already has them. Union bboxes across multiple blocks if the sensitive text spans several.",
    "",
    "Return JSON:",
    '  decisions: one {"ocr_index", "keep", "text", "reason"} entry for EVERY fuzzy-matched OCR candidate above. Use null for text/reason when you have no comment.',
    '  additions: zero or more {"bbox", "text", "reason"} for missed instances. Bbox is [y_min, x_min, y_max, x_max] in 0..1000 (Gemini\'s native box_2d format). In `reason`, note which raw OCR block (WORD[i]/LINE[i]) you copied coords from, or explain why you had to estimate.',
  ].join("\n");

  const systemPrompt = system();
  alog(`curator frame #${frameIndex} request`, {
    model: agenticModelSlug(),
    thinking_level: agenticThinkingLevel(),
    query,
    frame: { w: frameWidth, h: frameHeight, jpeg_bytes: jpeg.byteLength },
    ocr_candidates: ocrBoxes.length,
    raw_ocr_total: rawItems.length,
    raw_ocr_relevant: relevantRaw.length,
    raw_ocr_context: contextRaw.length,
    user_text: userText,
  });
  runLog?.write({
    kind: "curator_request",
    frame_index: frameIndex,
    model: agenticModelSlug(),
    thinking_level: agenticThinkingLevel(),
    query,
    frame_width: frameWidth,
    frame_height: frameHeight,
    jpeg_bytes: jpeg.byteLength,
    ocr_candidates: ocrBoxes.length,
    raw_ocr_total: rawItems.length,
    raw_ocr_relevant: relevantRaw.length,
    raw_ocr_context: contextRaw.length,
    system_prompt: systemPrompt,
    user_text: userText,
  });

  const t0 = Date.now();
  let result;
  try {
    result = await generateObject({
      model,
      schema: CuratorOutputSchema,
      system: systemPrompt,
      providerOptions: agenticProviderOptions(),
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
  } catch (e) {
    aerr(`curator frame #${frameIndex} generateObject failed`, e);
    runLog?.write({
      kind: "curator_error",
      frame_index: frameIndex,
      error: e instanceof Error ? e.message : String(e),
    });
    throw e;
  }
  const elapsed = Date.now() - t0;

  const parsed = result.object;
  alog(`curator frame #${frameIndex} response (${elapsed}ms)`, {
    usage: result.usage,
    finishReason: result.finishReason,
    warnings: result.warnings,
    object: parsed,
  });

  const callCost = geminiCost(agenticModelId(), {
    inputTokens: result.usage?.inputTokens ?? 0,
    outputTokens: result.usage?.outputTokens ?? 0,
    reasoningTokens: result.usage?.outputTokenDetails?.reasoningTokens ?? 0,
    cachedInputTokens: result.usage?.inputTokenDetails?.cacheReadTokens ?? 0,
    callCount: 1,
  });
  runLog?.write({
    kind: "curator_response",
    frame_index: frameIndex,
    elapsed_ms: elapsed,
    finish_reason: result.finishReason,
    warnings: result.warnings,
    object: compact(parsed),
    usage: result.usage,
    cost_usd: callCost.totalUSD,
    input_usd: callCost.inputUSD,
    output_usd: callCost.outputUSD,
  });

  const keepByIndex = new Map<number, CuratorDecision>();
  for (const d of parsed.decisions) keepByIndex.set(d.ocr_index, d);

  const kept: ServerBox[] = [];
  const dropped: number[] = [];
  for (let i = 0; i < ocrBoxes.length; i++) {
    const d = keepByIndex.get(i);
    if (!d || d.keep === false) {
      dropped.push(i);
      continue;
    }
    const src = ocrBoxes[i];
    kept.push({
      ...src,
      text: d.text || src.text,
    });
  }

  const added: ServerBox[] = [];
  for (const a of parsed.additions) {
    // a.bbox is typed as number[] (Zod array with min=max=4). Tighten to
    // a fixed-length tuple at the call boundary; the schema already
    // guarantees length 4 at runtime. Format is Gemini's native box_2d:
    // [y_min, x_min, y_max, x_max] in 0..1000.
    const [ymin, xmin, ymax, xmax] = a.bbox;
    const rect = bboxToPixels(
      [ymin, xmin, ymax, xmax],
      frameWidth,
      frameHeight,
    );
    if (rect.w <= 0 || rect.h <= 0) continue;
    added.push({
      x: rect.x,
      y: rect.y,
      w: rect.w,
      h: rect.h,
      text: a.text,
      score: 1.0,
      origin: "fix",
    });
  }

  alog(`curator frame #${frameIndex} resolved`, {
    kept: kept.length,
    dropped: dropped.length,
    added: added.length,
    dropped_indices: dropped,
    kept_texts: kept.map((b) => b.text),
    added_texts: added.map((b) => b.text),
    input_tokens: result.usage?.inputTokens ?? null,
    output_tokens: result.usage?.outputTokens ?? null,
  });

  return { kept, added, dropped, raw: parsed, usage: result.usage ?? null };
}

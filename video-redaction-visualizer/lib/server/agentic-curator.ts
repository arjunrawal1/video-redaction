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

import { generateObject } from "ai";
import { z } from "zod";
import { aerr, alog } from "./agentic-log";
import {
  BOX_FORMAT_NOTE,
  agenticLanguageModel,
  agenticModelSlug,
  agenticProviderOptions,
  agenticThinkingLevel,
  bboxToPixels,
  pixelBoxToNormalizedBbox,
  type ServerBox,
} from "./openrouter";

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

// -- Raw-OCR prompt helpers ------------------------------------------------

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

type RawOcrItem = {
  type: "word" | "line";
  text: string;
  confidence: number;
  bbox: [number, number, number, number];
};

/**
 * Pull every WORD and LINE Textract block off the raw response and convert
 * to Gemini's native box_2d contract: [y_min, x_min, y_max, x_max] in
 * integer 0..1000 (top-left origin). Used by the curator prompt so the
 * model can copy pixel-accurate coords from Textract instead of
 * estimating them from the image.
 */
function extractRawOcrItems(raw: unknown): RawOcrItem[] {
  if (!raw || typeof raw !== "object") return [];
  const blocks: TextractBlock[] =
    (raw as { Blocks?: TextractBlock[] } | null)?.Blocks ?? [];
  const out: RawOcrItem[] = [];
  for (const b of blocks) {
    const t = b.BlockType;
    if (t !== "WORD" && t !== "LINE") continue;
    const bbox = b.Geometry?.BoundingBox;
    if (!bbox) continue;
    const left = Number(bbox.Left ?? 0);
    const top = Number(bbox.Top ?? 0);
    const w = Number(bbox.Width ?? 0);
    const h = Number(bbox.Height ?? 0);
    out.push({
      type: t === "WORD" ? "word" : "line",
      text: String(b.Text ?? ""),
      confidence: Math.round(Number(b.Confidence ?? 0) * 10) / 10,
      bbox: [
        Math.round(top * 1000),
        Math.round(left * 1000),
        Math.round((top + h) * 1000),
        Math.round((left + w) * 1000),
      ],
    });
  }
  return out;
}

/**
 * Rank raw OCR items by plausible relevance to the query, then trim to a
 * budget so the curator prompt doesn't balloon on text-heavy frames. A
 * block is considered relevant if ANY of the following hold:
 *   (a) its normalized text contains a 3+ char substring of the query
 *   (b) the query's normalized text contains its text as a substring
 *       (case-insensitive, 2+ chars)
 *   (c) fuzzy rank by character-level SequenceMatcher-ish score.
 * Relevance ranking is done by longest shared substring length.
 *
 * Non-relevant blocks are still included (LINE only, not WORD) up to the
 * budget because they help the model localize in context (row labels,
 * neighboring cells, etc.) — but they're emitted after the relevant
 * ones so truncation clips the least-useful rows first.
 */
function filterAndRankOcrItems(
  items: RawOcrItem[],
  query: string,
  maxItems = 120,
): { relevant: RawOcrItem[]; context: RawOcrItem[] } {
  const q = query.toLowerCase();
  const qWords = q.split(/\s+/).filter((w) => w.length >= 2);

  const sharedLen = (a: string, b: string): number => {
    if (!a || !b) return 0;
    const lo = a.toLowerCase();
    const rhs = b.toLowerCase();
    let best = 0;
    for (let i = 0; i < lo.length; i++) {
      for (let j = i + 1; j <= lo.length; j++) {
        const slice = lo.slice(i, j);
        if (slice.length <= best) continue;
        if (rhs.includes(slice)) best = slice.length;
      }
    }
    return best;
  };

  const scored: Array<{ item: RawOcrItem; score: number }> = items.map((it) => {
    const t = it.text.toLowerCase();
    const viaQ = qWords.reduce(
      (m, w) => Math.max(m, t.includes(w) ? w.length : 0),
      0,
    );
    const viaShared = sharedLen(t, q);
    return { item: it, score: Math.max(viaQ, viaShared) };
  });

  const relevantScored = scored.filter((s) => s.score >= 2);
  relevantScored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    // Prefer LINE blocks when scores tie — they're more useful for
    // copying bboxes of multi-token matches.
    if (a.item.type !== b.item.type) return a.item.type === "line" ? -1 : 1;
    return b.item.confidence - a.item.confidence;
  });

  const relevant = relevantScored
    .map((s) => s.item)
    .slice(0, Math.min(maxItems, 60));
  const remaining = maxItems - relevant.length;
  const context = items
    .filter((it) => it.type === "line")
    .filter((it) => !relevant.includes(it))
    .slice(0, Math.max(0, remaining));
  return { relevant, context };
}

function formatOcrItemsForPrompt(items: RawOcrItem[]): string {
  if (items.length === 0) return "(none)";
  return items
    .map(
      (it, i) =>
        `${it.type.toUpperCase()}[${i}] text=${JSON.stringify(it.text)} conf=${it.confidence} bbox=[${it.bbox.join(", ")}]`,
    )
    .join("\n");
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
}): Promise<CuratorResult> {
  const {
    jpeg,
    query,
    frameIndex,
    frameWidth,
    frameHeight,
    ocrBoxes,
    ocrRaw,
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

  const t0 = Date.now();
  let result;
  try {
    result = await generateObject({
      model,
      schema: CuratorOutputSchema,
      system: system(),
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
  });

  return { kept, added, dropped, raw: parsed };
}

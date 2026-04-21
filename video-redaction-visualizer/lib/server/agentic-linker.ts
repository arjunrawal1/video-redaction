// Agentic phase-1.5: adjacent-frame identity linker.
//
// The curator (phase-1) produces a final set of redaction boxes per
// frame, but says nothing about which box in frame N is the SAME
// real-world redaction as a given box in frame N+1. The UI wants that
// correspondence so it can tween box coordinates smoothly between the
// sparse scanned frames at render time.
//
// Rather than a hand-tuned tracker (IoU + text-similarity thresholds),
// we ask Gemini directly. It's the exact kind of "is this the same
// thing?" judgment that LLMs are good at across scrolling, reflow,
// partial reveals, OCR garbling, and re-cropping — all edge cases that
// kill threshold-based trackers.
//
// Call shape: one `generateObject` per adjacent scanned pair (N, N+1).
// Sequential by construction — the track ids in N+1 depend on the track
// ids assigned to N. The linker fallback (lib/server/linker-fallback.ts)
// is invoked when Gemini fails or returns garbage, so a bad response
// can't corrupt the stream.
//
// Output contract to the rest of the pipeline: each ServerBox gets a
// `track_id` like `t{n}`, stable across consecutive frames whenever the
// model judged them the same redaction. Fresh ids are minted by the
// caller; this module just returns link decisions.

import { generateObject, type LanguageModelUsage } from "ai";
import { z } from "zod";
import { aerr, alog } from "./agentic-log";
import { geminiCost } from "./cost";
import {
  agenticLinkerLanguageModel,
  agenticLinkerModelId,
  agenticLinkerProviderOptions,
  pixelBoxToNormalizedBbox,
  type ServerBox,
} from "./openrouter";
import { compact, type RunLog } from "./run-log";

// `a_index: null` means "new track in frame B". Gemini occasionally
// drifts and returns numbers out of range; validation clamps those back
// to null. `reason` is best-effort narration for debug panels.
const LinkItemSchema = z.object({
  b_index: z.number().int().nonnegative(),
  a_index: z.number().int().nullable(),
  reason: z.string().nullable(),
});

const LinkOutputSchema = z.object({
  links: z.array(LinkItemSchema),
});

export type LinkDecision = {
  /** Index of a box in frame B. Always in range [0, boxesB.length). */
  b_index: number;
  /**
   * Index of the matching box in frame A, or null if this is a new
   * redaction that first appears in frame B. After validation, every
   * non-null value is guaranteed in range [0, boxesA.length) and
   * one-to-one (no two b_indices share the same a_index).
   */
  a_index: number | null;
  reason: string | null;
};

export type LinkPairResult = {
  links: LinkDecision[];
  /** AI SDK usage for the single Gemini call. Fed into cost.ts. */
  usage: LanguageModelUsage | null;
  /**
   * Raw model output after Zod parsing, pre-validation. Kept for the
   * debug panel / log; do not rely on its shape beyond `{ links: [...] }`.
   */
  raw: unknown;
};

function system(): string {
  return [
    "You are matching redaction boxes between two consecutive frames of a screen recording.",
    "For each box in FRAME B, decide whether it is the SAME REAL-WORLD REDACTION as one of the boxes in FRAME A, or a NEW box that just appeared in B.",
    "",
    '"Same" means the same on-screen text instance — even if it moved (scroll, reflow, animation), resized, was recropped by OCR, changed visible state, or has garbled/partial text ("test wor" in one frame and "test word" in the next is the SAME redaction).',
    "",
    "RULES:",
    "- Return exactly one `links` entry per box in FRAME B.",
    "- Each FRAME A box can be matched by AT MOST ONE FRAME B box. If two B boxes look like the same A, pick the better fit and mark the other as new.",
    "- When uncertain between 'same' and 'new', prefer NEW. The UI tweens same-track boxes smoothly, so over-splitting a chain just means the tween starts fresh — visually fine. Merging unrelated boxes teleports a rectangle across the screen — visually bad.",
    "- Boxes in FRAME A that have no continuation in B simply go unmatched (they disappeared). Do not emit an entry for them.",
  ].join("\n");
}

/**
 * Link every box in `boxesB` back to a box in `boxesA` (or mark it new).
 * Caller is responsible for minting track ids from the returned decisions.
 *
 * The `null | out-of-range | dup` cases are all normalized here so the
 * caller can treat the result as structurally valid.
 */
export async function linkFramePair(opts: {
  jpegA: Uint8Array;
  jpegB: Uint8Array;
  frameIndexA: number;
  frameIndexB: number;
  frameWidthA: number;
  frameHeightA: number;
  frameWidthB: number;
  frameHeightB: number;
  boxesA: ServerBox[];
  boxesB: ServerBox[];
  runLog?: RunLog | null;
}): Promise<LinkPairResult> {
  const {
    jpegA,
    jpegB,
    frameIndexA,
    frameIndexB,
    frameWidthA,
    frameHeightA,
    frameWidthB,
    frameHeightB,
    boxesA,
    boxesB,
    runLog,
  } = opts;

  // Degenerate cases: skip the model call entirely.
  if (boxesB.length === 0) {
    return { links: [], usage: null, raw: null };
  }
  if (boxesA.length === 0) {
    const links = boxesB.map(
      (_, i): LinkDecision => ({ b_index: i, a_index: null, reason: null }),
    );
    return { links, usage: null, raw: null };
  }

  const model = agenticLinkerLanguageModel();

  const formatBoxes = (
    boxes: ServerBox[],
    w: number,
    h: number,
  ): string =>
    boxes
      .map((b, i) => {
        const [ymin, xmin, ymax, xmax] = pixelBoxToNormalizedBbox(b, w, h);
        return `${i}: text=${JSON.stringify(b.text)} box_2d=[${ymin}, ${xmin}, ${ymax}, ${xmax}]`;
      })
      .join("\n");

  const userText = [
    `FRAME A = frame #${frameIndexA}. Boxes on A:`,
    formatBoxes(boxesA, frameWidthA, frameHeightA),
    "",
    `FRAME B = frame #${frameIndexB}. Boxes on B:`,
    formatBoxes(boxesB, frameWidthB, frameHeightB),
    "",
    `Return a "links" array with exactly ${boxesB.length} entries — one for each FRAME B box index 0..${boxesB.length - 1}.`,
    "For each entry, set `a_index` to the matching FRAME A box index, or null if this is a new redaction in B.",
  ].join("\n");

  const systemPrompt = system();
  alog(`linker pair #${frameIndexA}→#${frameIndexB} request`, {
    model: agenticLinkerModelId(),
    boxesA: boxesA.length,
    boxesB: boxesB.length,
  });
  runLog?.write({
    kind: "linker_request",
    frame_a: frameIndexA,
    frame_b: frameIndexB,
    model: agenticLinkerModelId(),
    boxes_a: boxesA.length,
    boxes_b: boxesB.length,
    system_prompt: systemPrompt,
    user_text: userText,
  });

  const t0 = Date.now();
  let result;
  try {
    result = await generateObject({
      model,
      schema: LinkOutputSchema,
      system: systemPrompt,
      providerOptions: agenticLinkerProviderOptions(),
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "FRAME A:" },
            { type: "image", image: jpegA, mediaType: "image/jpeg" },
            { type: "text", text: "FRAME B:" },
            { type: "image", image: jpegB, mediaType: "image/jpeg" },
            { type: "text", text: userText },
          ],
        },
      ],
    });
  } catch (e) {
    aerr(`linker pair #${frameIndexA}→#${frameIndexB} generateObject failed`, e);
    runLog?.write({
      kind: "linker_error",
      frame_a: frameIndexA,
      frame_b: frameIndexB,
      error: e instanceof Error ? e.message : String(e),
    });
    throw e;
  }
  const elapsed = Date.now() - t0;

  const validated = validateLinks({
    raw: result.object.links,
    boxesA,
    boxesB,
  });

  const callCost = geminiCost(agenticLinkerModelId(), {
    inputTokens: result.usage?.inputTokens ?? 0,
    outputTokens: result.usage?.outputTokens ?? 0,
    reasoningTokens: result.usage?.outputTokenDetails?.reasoningTokens ?? 0,
    cachedInputTokens: result.usage?.inputTokenDetails?.cacheReadTokens ?? 0,
    callCount: 1,
  });
  alog(`linker pair #${frameIndexA}→#${frameIndexB} response (${elapsed}ms)`, {
    usage: result.usage,
    finishReason: result.finishReason,
    warnings: result.warnings,
    input_tokens: result.usage?.inputTokens ?? null,
    output_tokens: result.usage?.outputTokens ?? null,
    matched: validated.filter((d) => d.a_index != null).length,
    new_tracks: validated.filter((d) => d.a_index == null).length,
    links: validated,
    cost_usd: callCost.totalUSD,
  });
  runLog?.write({
    kind: "linker_response",
    frame_a: frameIndexA,
    frame_b: frameIndexB,
    elapsed_ms: elapsed,
    finish_reason: result.finishReason,
    raw_object: compact(result.object),
    links: validated,
    matched: validated.filter((d) => d.a_index != null).length,
    new_tracks: validated.filter((d) => d.a_index == null).length,
    usage: result.usage,
    cost_usd: callCost.totalUSD,
    input_usd: callCost.inputUSD,
    output_usd: callCost.outputUSD,
  });

  return {
    links: validated,
    usage: result.usage ?? null,
    raw: result.object,
  };
}

/**
 * Normalize the model's raw `links` array into a well-formed decision
 * list of length `boxesB.length` with one-to-one A→B. Trust nothing:
 *   - missing b_index entries → filled as `a_index: null` (new track).
 *   - out-of-range a_index → clamped to null.
 *   - duplicate b_index claims on the same a_index → keeper picked by
 *     best IoU+text heuristic; the loser becomes a new track.
 */
function validateLinks(args: {
  raw: Array<{ b_index: number; a_index: number | null; reason: string | null }>;
  boxesA: ServerBox[];
  boxesB: ServerBox[];
}): LinkDecision[] {
  const { raw, boxesA, boxesB } = args;

  // Collapse rows by b_index (last write wins — model sometimes repeats).
  const byB = new Map<number, LinkDecision>();
  for (const row of raw) {
    const bi = row.b_index;
    if (!Number.isInteger(bi) || bi < 0 || bi >= boxesB.length) continue;
    let ai: number | null = row.a_index;
    if (ai != null) {
      if (!Number.isInteger(ai) || ai < 0 || ai >= boxesA.length) {
        ai = null;
      }
    }
    byB.set(bi, { b_index: bi, a_index: ai, reason: row.reason ?? null });
  }

  // Fill in any missing b_index entries as "new track".
  for (let bi = 0; bi < boxesB.length; bi++) {
    if (!byB.has(bi)) {
      byB.set(bi, { b_index: bi, a_index: null, reason: null });
    }
  }

  // Enforce one-to-one: two b_indices can't both claim the same a_index.
  // Winner = highest pairScore; losers get a_index=null.
  const byA = new Map<number, LinkDecision[]>();
  for (const d of byB.values()) {
    if (d.a_index == null) continue;
    const arr = byA.get(d.a_index) ?? [];
    arr.push(d);
    byA.set(d.a_index, arr);
  }
  for (const [ai, dups] of byA) {
    if (dups.length < 2) continue;
    let winner = dups[0];
    let winnerScore = pairScore(boxesA[ai], boxesB[winner.b_index]);
    for (let i = 1; i < dups.length; i++) {
      const s = pairScore(boxesA[ai], boxesB[dups[i].b_index]);
      if (s > winnerScore) {
        winner = dups[i];
        winnerScore = s;
      }
    }
    for (const d of dups) {
      if (d === winner) continue;
      d.a_index = null;
      d.reason = d.reason
        ? `${d.reason} [demoted: duplicate A=${ai} claim]`
        : `demoted: duplicate A=${ai} claim`;
    }
  }

  // Emit in B-order for deterministic downstream indexing.
  const out: LinkDecision[] = [];
  for (let bi = 0; bi < boxesB.length; bi++) {
    const d = byB.get(bi);
    out.push(d ?? { b_index: bi, a_index: null, reason: null });
  }
  return out;
}

/**
 * Cheap heuristic used only as a tiebreaker when Gemini returns two
 * B-boxes claiming the same A-box. Higher = stronger evidence the pair
 * is the same redaction. Weighted 70/30 toward spatial overlap since
 * text equality alone can't distinguish duplicates.
 */
function pairScore(a: ServerBox, b: ServerBox): number {
  const iou = computeIoU(a, b);
  const textEq = normalize(a.text) === normalize(b.text) ? 1 : 0;
  return 0.7 * iou + 0.3 * textEq;
}

function computeIoU(a: ServerBox, b: ServerBox): number {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0;
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

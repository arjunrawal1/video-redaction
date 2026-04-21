// Pricing + cost accumulation for the redaction pipeline.
//
// Two cost sources per run:
//   1. AWS Textract DetectDocumentText — one API call per frame sent to
//      OCR. Billed per 1000 "pages" (we count 1 page per frame).
//   2. Google Gemini 3 — one API call per curator / focused agent /
//      navigator run. Billed per 1M tokens (separate input + output).
//
// All prices are USD per 1M tokens unless otherwise noted, pulled from:
//   - https://ai.google.dev/gemini-api/docs/gemini-3#meet-the-gemini-3-series
//   - https://aws.amazon.com/textract/pricing/
// and should be updated when those pages change.

import type { LanguageModelUsage } from "ai";

// -- Gemini pricing --------------------------------------------------------

type GeminiPricing = {
  /** USD per 1M input tokens (below the over-threshold tier). */
  input: number;
  /** USD per 1M output tokens (below the over-threshold tier). */
  output: number;
  /**
   * If inputTokens for a request exceeds this threshold, the over-threshold
   * prices apply to the whole request. Set both overInput / overOutput
   * when present.
   */
  threshold?: number;
  overInput?: number;
  overOutput?: number;
};

// Prices known as of 2026-04. Preview models; confirm against the pricing
// page when running large workloads.
const GEMINI_PRICES: Record<string, GeminiPricing> = {
  "gemini-3-flash-preview": { input: 0.5, output: 3.0 },
  "gemini-3.1-flash-lite-preview": { input: 0.25, output: 1.5 },
  "gemini-3.1-pro-preview": {
    input: 2.0,
    output: 12.0,
    threshold: 200_000,
    overInput: 4.0,
    overOutput: 18.0,
  },
};

function pricingFor(model: string): GeminiPricing {
  if (GEMINI_PRICES[model]) return GEMINI_PRICES[model];
  // Fallback: log an unknown-model warning via the caller; return zeros so
  // we never crash a run on a pricing gap.
  return { input: 0, output: 0 };
}

// -- Usage aggregation -----------------------------------------------------

export type AggregateUsage = {
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cachedInputTokens: number;
  /** Count of Gemini API calls folded into this aggregate. */
  callCount: number;
};

export function emptyUsage(): AggregateUsage {
  return {
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    cachedInputTokens: 0,
    callCount: 0,
  };
}

/** Add a single AI SDK LanguageModelUsage into an aggregate, in place. */
export function addUsage(
  into: AggregateUsage,
  u: LanguageModelUsage | undefined | null,
): void {
  if (!u) return;
  into.inputTokens += u.inputTokens ?? 0;
  into.outputTokens += u.outputTokens ?? 0;
  into.reasoningTokens += u.outputTokenDetails?.reasoningTokens ?? 0;
  into.cachedInputTokens += u.inputTokenDetails?.cacheReadTokens ?? 0;
  into.callCount += 1;
}

export function mergeUsage(a: AggregateUsage, b: AggregateUsage): AggregateUsage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    reasoningTokens: a.reasoningTokens + b.reasoningTokens,
    cachedInputTokens: a.cachedInputTokens + b.cachedInputTokens,
    callCount: a.callCount + b.callCount,
  };
}

// -- Cost math -------------------------------------------------------------

export type GeminiCost = {
  model: string;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cachedInputTokens: number;
  callCount: number;
  inputUSD: number;
  outputUSD: number;
  totalUSD: number;
  /** Effective tier used. "over" if threshold was tripped by avg request. */
  tier: "standard" | "over";
  pricing: GeminiPricing;
};

export function geminiCost(
  model: string,
  usage: AggregateUsage,
): GeminiCost {
  const p = pricingFor(model);
  // Pro-tier threshold is per-request, not cumulative. We don't have
  // per-request breakdowns here, but we CAN approximate by checking the
  // average: if avg input per call > threshold, treat the run as "over"
  // tier. Flash has no threshold and `p.threshold` is undefined, which
  // correctly falls through to the standard rates.
  const avgInput =
    usage.callCount > 0 ? usage.inputTokens / usage.callCount : 0;
  const over =
    p.threshold != null &&
    p.overInput != null &&
    p.overOutput != null &&
    avgInput > p.threshold;
  const inputRate = over ? p.overInput! : p.input;
  const outputRate = over ? p.overOutput! : p.output;
  const inputUSD = (usage.inputTokens / 1_000_000) * inputRate;
  // Reasoning tokens are billed as output tokens on Gemini 3. Our
  // aggregate already folds them into outputTokens (AI SDK counts them
  // there), so don't double-count.
  const outputUSD = (usage.outputTokens / 1_000_000) * outputRate;
  return {
    model,
    inputTokens: usage.inputTokens,
    outputTokens: usage.outputTokens,
    reasoningTokens: usage.reasoningTokens,
    cachedInputTokens: usage.cachedInputTokens,
    callCount: usage.callCount,
    inputUSD,
    outputUSD,
    totalUSD: inputUSD + outputUSD,
    tier: over ? "over" : "standard",
    pricing: p,
  };
}

// -- Textract pricing ------------------------------------------------------

/**
 * AWS Textract DetectDocumentText pricing: $1.50 per 1,000 pages for the
 * first 1M pages/month, $0.60 per 1,000 thereafter. We assume first-tier
 * pricing for a single-run estimate; large monthly bills will exceed this
 * estimate by at most 60%.
 */
export function textractCost(pageCount: number): {
  pages: number;
  totalUSD: number;
  perPageUSD: number;
} {
  const perPageUSD = 1.5 / 1000; // $0.0015 / page
  return {
    pages: pageCount,
    totalUSD: pageCount * perPageUSD,
    perPageUSD,
  };
}

// -- Formatting ------------------------------------------------------------

export function formatUSD(n: number): string {
  if (!Number.isFinite(n)) return "$?";
  if (n === 0) return "$0.00";
  if (n < 0.0001) return `$${n.toFixed(8)}`;
  if (n < 0.01) return `$${n.toFixed(6)}`;
  if (n < 1) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

export function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

// Module-level LRU cache of per-frame engine state. Mirrors the Python
// app/ocr_cache.py shape. Used by both the `gemini` and `teamwork`
// engines (the agentic navigator in particular reads phase-1 state from
// here to build its initial context).
//
// Only lives in the Next.js process memory — fine for `next dev` (single
// node). Prod deployments that horizontally scale would swap this for
// Redis / a shared KV, but that's out of scope.

import type { ServerBox } from "./openrouter";
import type { Instance, RegionBbox } from "./prompt/types";

export type Engine = "gemini" | "teamwork" | "prompt";

type CommonKeyParams = {
  videoHash: string;
  fps: number | null;
  dedupThreshold: number;
  // Mirrors `max_gap` in the Python frame cache / ocr cache. Part of
  // the cache identity so runs with different gap caps don't alias.
  maxGap: number;
  frameFrom: number;
  frameTo: number;
};

export type QueryKeyParams = CommonKeyParams & {
  engine: "gemini" | "teamwork";
  queryNorm: string;
};

export type PromptKeyParams = CommonKeyParams & {
  engine: "prompt";
  predicateHash: string;
};

export type KeyParams = QueryKeyParams | PromptKeyParams;

export type FrameState = {
  width: number;
  height: number;
  // Mutated in place by backtrack/forward drivers so that when the outer
  // loop advances to this frame, newly-added anchors are visible.
  matched: ServerBox[];
  raw: unknown | null;
  blob: Uint8Array;
  // Teamwork-only: cached Textract raw response for this frame. Included
  // in the curator's "copy OCR debug" payload on the client.
  ocrRaw?: unknown;
  // Teamwork-only: OCR's phase-1 boxes per frame. The navigator uses
  // these as `ocr_candidates` the model can adopt via `adopt_ocr_box`.
  ocrMatched?: ServerBox[];
  // Unused in the agentic pipeline but kept for schema compatibility
  // with prior runs cached in memory.
  flagged?: boolean;
  // Prompt-mode only. Stable cross-frame identity records emitted by the
  // prompt curator.
  instances?: Instance[];
  // Prompt-mode only. Resolver region-localizer outputs (one per branch /
  // sub_id pair) cached for reuse by navigate.
  regions?: RegionBbox[];
};

type BaseCacheEntry = {
  engine: Engine;
  videoHash: string;
  fps: number | null;
  dedupThreshold: number;
  maxGap: number;
  frameFrom: number;
  frameTo: number;
  // 1-indexed frame number -> state
  perFrame: Record<number, FrameState>;
};

export type QueryCacheEntry = BaseCacheEntry & {
  engine: "gemini" | "teamwork";
  queryNorm: string;
  predicateHash?: never;
};

export type PromptCacheEntry = BaseCacheEntry & {
  engine: "prompt";
  predicateHash: string;
  queryNorm?: never;
};

export type CacheEntry = QueryCacheEntry | PromptCacheEntry;

type Key = string;

function makeKey(p: KeyParams): Key {
  if (p.engine === "prompt") {
    return [
      p.engine,
      p.videoHash,
      p.predicateHash,
      p.fps ?? "null",
      p.dedupThreshold,
      p.maxGap,
      p.frameFrom,
      p.frameTo,
    ].join("|");
  }
  return [
    p.engine,
    p.videoHash,
    p.queryNorm,
    p.fps ?? "null",
    p.dedupThreshold,
    p.maxGap,
    p.frameFrom,
    p.frameTo,
  ].join("|");
}

function parseMaxEntries(raw: string | undefined, fallback: number): number {
  const n = Number(raw ?? "");
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

export const GEMINI_CACHE_MAX_ENTRIES = parseMaxEntries(
  process.env.GEMINI_CACHE_MAX_ENTRIES,
  32,
);

const _cache = new Map<Key, CacheEntry>();

export function createEntry(p: QueryKeyParams): QueryCacheEntry;
export function createEntry(p: PromptKeyParams): PromptCacheEntry;
export function createEntry(p: KeyParams): CacheEntry {
  const entry: CacheEntry =
    p.engine === "prompt"
      ? {
          engine: p.engine,
          videoHash: p.videoHash,
          predicateHash: p.predicateHash,
          fps: p.fps,
          dedupThreshold: p.dedupThreshold,
          maxGap: p.maxGap,
          frameFrom: p.frameFrom,
          frameTo: p.frameTo,
          perFrame: {},
        }
      : {
          engine: p.engine,
          videoHash: p.videoHash,
          queryNorm: p.queryNorm,
          fps: p.fps,
          dedupThreshold: p.dedupThreshold,
          maxGap: p.maxGap,
          frameFrom: p.frameFrom,
          frameTo: p.frameTo,
          perFrame: {},
        };
  const key = makeKey(p);
  // Re-running detect with the same key should reset any augmented state
  // so back/forward don't inherit stale mutations from an earlier run.
  _cache.set(key, entry);
  touch(key);
  evictIfNeeded();
  return entry;
}

export function getEntry(p: QueryKeyParams): QueryCacheEntry | null;
export function getEntry(p: PromptKeyParams): PromptCacheEntry | null;
export function getEntry(p: KeyParams): CacheEntry | null {
  const key = makeKey(p);
  const hit = _cache.get(key) ?? null;
  if (hit) touch(key);
  return hit as CacheEntry | null;
}

export function putFrame(
  entry: CacheEntry,
  frameIdx1: number,
  state: FrameState,
): void {
  entry.perFrame[frameIdx1] = state;
}

function touch(key: Key): void {
  const v = _cache.get(key);
  if (!v) return;
  _cache.delete(key);
  _cache.set(key, v);
}

function evictIfNeeded(): void {
  while (_cache.size > GEMINI_CACHE_MAX_ENTRIES) {
    const firstKey = _cache.keys().next().value;
    if (firstKey === undefined) break;
    _cache.delete(firstKey);
  }
}

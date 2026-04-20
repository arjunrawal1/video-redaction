// Module-level LRU cache of per-frame Gemini state. Mirrors the Python
// app/ocr_cache.py shape so the pass-driver can treat both engines the same.
//
// Only lives in the Next.js process memory — fine for `next dev` (single
// node). Prod deployments that horizontally scale would swap this for
// Redis / a shared KV, but that's out of scope.

import type { ServerBox } from "./openrouter";

export type FrameState = {
  width: number;
  height: number;
  // Mutated in place by backtrack/forward drivers so that when the outer
  // loop advances to this frame, newly-added anchors are visible.
  matched: ServerBox[];
  raw: unknown | null;
  blob: Uint8Array;
};

export type CacheEntry = {
  videoHash: string;
  queryNorm: string;
  fps: number | null;
  dedupThreshold: number;
  frameFrom: number;
  frameTo: number;
  // 1-indexed frame number -> state
  perFrame: Record<number, FrameState>;
};

type Key = string;

function makeKey(p: {
  videoHash: string;
  queryNorm: string;
  fps: number | null;
  dedupThreshold: number;
  frameFrom: number;
  frameTo: number;
}): Key {
  return [
    p.videoHash,
    p.queryNorm,
    p.fps ?? "null",
    p.dedupThreshold,
    p.frameFrom,
    p.frameTo,
  ].join("|");
}

const _MAX_ENTRIES = 8;
const _cache = new Map<Key, CacheEntry>();

export function createEntry(p: {
  videoHash: string;
  queryNorm: string;
  fps: number | null;
  dedupThreshold: number;
  frameFrom: number;
  frameTo: number;
}): CacheEntry {
  const entry: CacheEntry = {
    videoHash: p.videoHash,
    queryNorm: p.queryNorm,
    fps: p.fps,
    dedupThreshold: p.dedupThreshold,
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

export function getEntry(p: {
  videoHash: string;
  queryNorm: string;
  fps: number | null;
  dedupThreshold: number;
  frameFrom: number;
  frameTo: number;
}): CacheEntry | null {
  const key = makeKey(p);
  const hit = _cache.get(key) ?? null;
  if (hit) touch(key);
  return hit;
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
  while (_cache.size > _MAX_ENTRIES) {
    const firstKey = _cache.keys().next().value;
    if (firstKey === undefined) break;
    _cache.delete(firstKey);
  }
}

import type { SceneSummary } from "./types";

function parseMax(raw: string | undefined, fallback: number): number {
  const n = Number(raw ?? "");
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

export const SCENE_SUMMARY_CACHE_MAX_ENTRIES = parseMax(
  process.env.SCENE_SUMMARY_CACHE_MAX_ENTRIES,
  16,
);

type CacheKey = string;
const _cache = new Map<CacheKey, SceneSummary>();

export function makeSceneSummaryCacheKey(args: {
  videoHash: string;
  maxFrames: number;
  frameFrom: number;
  frameTo: number;
  fps: number | null;
  dedupThreshold: number;
  maxGap: number;
}): CacheKey {
  return [
    args.videoHash,
    args.maxFrames,
    args.frameFrom,
    args.frameTo,
    args.fps ?? "null",
    args.dedupThreshold,
    args.maxGap,
  ].join("|");
}

export function getSceneSummary(key: CacheKey): SceneSummary | null {
  const hit = _cache.get(key) ?? null;
  if (!hit) return null;
  _cache.delete(key);
  _cache.set(key, hit);
  return hit;
}

export function putSceneSummary(key: CacheKey, summary: SceneSummary): void {
  _cache.set(key, summary);
  _cache.delete(key);
  _cache.set(key, summary);
  while (_cache.size > SCENE_SUMMARY_CACHE_MAX_ENTRIES) {
    const oldest = _cache.keys().next().value;
    if (oldest == null) break;
    _cache.delete(oldest);
  }
}

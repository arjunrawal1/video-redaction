import { generateObject } from "ai";
import { z } from "zod";
import { extractRawOcrItems } from "@/lib/server/agentic-ocr";
import {
  agenticLanguageModel,
  agenticModelSlug,
  agenticProviderOptions,
  agenticThinkingLevel,
} from "@/lib/server/openrouter";
import { compact, type RunLog } from "@/lib/server/run-log";
import {
  collectLeafPredicates,
  type Predicate,
  type PredicateLeaf,
  type RegionBbox,
  type ResolvedCandidate,
} from "./types";

const RegionBboxSchema = z.object({
  sub_id: z.string(),
  bbox_2d: z.array(z.number().int().min(0).max(1000)).length(4),
  confidence: z.number().min(0).max(1),
  reason: z.string(),
});

const RegionResponseSchema = z.object({
  regions: z.array(RegionBboxSchema).default([]),
});

function parseIntEnv(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : fallback;
}

const REGION_CACHE_MAX_ENTRIES = parseIntEnv("PROMPT_REGION_CACHE_MAX_ENTRIES", 2048);

const _regionCache = new Map<string, RegionBbox[]>();

function regionCacheKey(args: {
  videoHash: string;
  frameIndex: number;
  branch: string;
}): string {
  return `${args.videoHash}|${args.frameIndex}|${args.branch}`;
}

function getRegionCache(key: string): RegionBbox[] | null {
  const hit = _regionCache.get(key) ?? null;
  if (!hit) return null;
  _regionCache.delete(key);
  _regionCache.set(key, hit);
  return hit.map((r) => ({ ...r, bbox_2d: [...r.bbox_2d] as [number, number, number, number] }));
}

function putRegionCache(key: string, regions: RegionBbox[]): void {
  const cloned = regions.map((r) => ({
    ...r,
    bbox_2d: [...r.bbox_2d] as [number, number, number, number],
  }));
  _regionCache.set(key, cloned);
  _regionCache.delete(key);
  _regionCache.set(key, cloned);
  while (_regionCache.size > REGION_CACHE_MAX_ENTRIES) {
    const oldest = _regionCache.keys().next().value;
    if (oldest == null) break;
    _regionCache.delete(oldest);
  }
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function levenshteinRatio(a: string, b: string): number {
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  const la = a.length;
  const lb = b.length;
  const prev = new Array<number>(lb + 1);
  const curr = new Array<number>(lb + 1);
  for (let j = 0; j <= lb; j++) prev[j] = j;
  for (let i = 1; i <= la; i++) {
    curr[0] = i;
    for (let j = 1; j <= lb; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    for (let j = 0; j <= lb; j++) prev[j] = curr[j];
  }
  return 1 - prev[lb] / Math.max(la, lb);
}

function fuzzyContains(haystack: string, needle: string, threshold = 0.82): boolean {
  if (!needle) return false;
  if (haystack.includes(needle)) return true;
  const n = needle.length;
  let best = 0;
  for (const delta of [0, -1, 1, -2, 2]) {
    const windowLen = n + delta;
    if (windowLen < 3 || windowLen > haystack.length) continue;
    for (let i = 0; i <= haystack.length - windowLen; i++) {
      const window = haystack.slice(i, i + windowLen);
      const ratio = levenshteinRatio(needle, window);
      if (ratio > best) {
        best = ratio;
        if (best >= threshold) return true;
      }
    }
  }
  return haystack.length <= n + 2 && levenshteinRatio(needle, haystack) >= threshold;
}

function textLeafCandidates(args: {
  leaf: Extract<PredicateLeaf, { kind: "text" }>;
  rawOcr: unknown;
}): ResolvedCandidate[] {
  const items = extractRawOcrItems(args.rawOcr);
  const needle = normalize(args.leaf.text);
  const out: ResolvedCandidate[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const textNorm = normalize(item.text);
    if (!textNorm) continue;
    const exact = textNorm.includes(needle) || needle.includes(textNorm);
    const fuzzy = args.leaf.fuzzy ? fuzzyContains(textNorm, needle) : false;
    if (!exact && !fuzzy) continue;
    const score = exact ? 1 : Math.max(0.5, levenshteinRatio(textNorm, needle));
    out.push({
      ocr_index: i,
      branch: args.leaf.branch ?? "L0",
      bbox: item.bbox,
      text: item.text,
      prior_score: Math.max(0, Math.min(1, score)),
    });
  }
  return out;
}

async function localizeRegionLive(args: {
  leaf: Extract<PredicateLeaf, { kind: "region" }>;
  frameBlob: Uint8Array;
  frameIndex: number;
  runLog?: RunLog | null;
}): Promise<RegionBbox[]> {
  const model = agenticLanguageModel();
  const system = [
    "Localize UI regions for redaction.",
    "Return bbox_2d in Gemini format [y_min, x_min, y_max, x_max] in 0..1000.",
    "If all_instances=true and the region repeats (cells/rows/cards), return one entry per visible instance with a stable sub_id.",
    "Use sub_id='0' for singleton regions.",
  ].join("\n");

  const userText = JSON.stringify(
    {
      description: args.leaf.description,
      all_instances: args.leaf.all_instances,
      anchors: args.leaf.anchors ?? [],
    },
    null,
    2,
  );

  args.runLog?.write({
    kind: "region_request",
    frame_index: args.frameIndex,
    model: agenticModelSlug(),
    thinking_level: agenticThinkingLevel(),
    branch: args.leaf.branch ?? "L0",
    description: args.leaf.description,
    all_instances: args.leaf.all_instances,
    anchors: args.leaf.anchors ?? [],
    jpeg_bytes: args.frameBlob.byteLength,
    system_prompt: system,
    user_text: userText,
  });

  const t0 = Date.now();
  try {
    const result = await generateObject({
      model,
      schema: RegionResponseSchema,
      system,
      providerOptions: agenticProviderOptions(),
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: userText,
            },
            { type: "image", image: args.frameBlob, mediaType: "image/jpeg" },
          ],
        },
      ],
    });

    const regions = result.object.regions.map((r) => ({
      branch: args.leaf.branch ?? "L0",
      sub_id: r.sub_id || "0",
      bbox_2d: r.bbox_2d as [number, number, number, number],
      confidence: Math.max(0, Math.min(1, Number(r.confidence))),
      reason: r.reason,
    }));

    args.runLog?.write({
      kind: "region_response",
      frame_index: args.frameIndex,
      branch: args.leaf.branch ?? "L0",
      elapsed_ms: Date.now() - t0,
      finish_reason: result.finishReason,
      warnings: result.warnings,
      usage: result.usage,
      regions: compact(regions),
      regions_returned: regions.length,
    });

    if (regions.length > 0) return regions;

    args.runLog?.write({
      kind: "region_fallback_fullframe",
      frame_index: args.frameIndex,
      branch: args.leaf.branch ?? "L0",
      reason: "model returned zero regions",
    });
  } catch (e) {
    args.runLog?.write({
      kind: "region_error",
      frame_index: args.frameIndex,
      branch: args.leaf.branch ?? "L0",
      elapsed_ms: Date.now() - t0,
      error: e instanceof Error ? e.message : String(e),
    });
    args.runLog?.write({
      kind: "region_fallback_fullframe",
      frame_index: args.frameIndex,
      branch: args.leaf.branch ?? "L0",
      reason: "generateObject threw",
    });
  }

  return [
    {
      branch: args.leaf.branch ?? "L0",
      sub_id: "0",
      bbox_2d: [0, 0, 1000, 1000],
      confidence: 0.05,
      reason: "fallback full-frame region",
    },
  ];
}

export async function localizeRegionForFrame(args: {
  videoHash: string;
  frameIndex: number;
  leaf: Extract<PredicateLeaf, { kind: "region" }>;
  frameBlob: Uint8Array;
  runLog?: RunLog | null;
}): Promise<RegionBbox[]> {
  const key = regionCacheKey({
    videoHash: args.videoHash,
    frameIndex: args.frameIndex,
    branch: args.leaf.branch ?? "L0",
  });
  const hit = getRegionCache(key);
  if (hit) {
    args.runLog?.write({
      kind: "region_cache_hit",
      frame_index: args.frameIndex,
      branch: args.leaf.branch ?? "L0",
      regions: compact(hit),
    });
    return hit;
  }

  const regions = await localizeRegionLive({
    leaf: args.leaf,
    frameBlob: args.frameBlob,
    frameIndex: args.frameIndex,
    runLog: args.runLog ?? null,
  });
  putRegionCache(key, regions);
  return regions;
}

export async function precomputeFrame(args: {
  videoHash: string;
  frameIndex: number;
  predicate: Predicate;
  rawOcr: unknown;
  frameImage: Uint8Array;
  width: number;
  height: number;
  runLog?: RunLog | null;
}): Promise<{ regions: RegionBbox[]; textCandidates: ResolvedCandidate[] }> {
  const leaves = collectLeafPredicates(args.predicate);
  const regions: RegionBbox[] = [];
  const textCandidates: ResolvedCandidate[] = [];

  for (const leaf of leaves) {
    if (leaf.kind === "text") {
      const cand = textLeafCandidates({ leaf, rawOcr: args.rawOcr });
      textCandidates.push(...cand);
      args.runLog?.write({
        kind: "text_candidates",
        frame_index: args.frameIndex,
        branch: leaf.branch ?? "L0",
        leaf_text: leaf.text,
        fuzzy: leaf.fuzzy,
        candidate_count: cand.length,
        candidates: compact(cand),
      });
      continue;
    }
    if (leaf.kind === "region") {
      const localized = await localizeRegionForFrame({
        videoHash: args.videoHash,
        frameIndex: args.frameIndex,
        leaf,
        frameBlob: args.frameImage,
        runLog: args.runLog ?? null,
      });
      for (const r of localized) {
        regions.push({
          branch: leaf.branch ?? r.branch,
          sub_id: r.sub_id,
          bbox_2d: r.bbox_2d,
          confidence: r.confidence,
          reason: r.reason,
        });
      }
    }
  }

  args.runLog?.write({
    kind: "precompute_done",
    frame_index: args.frameIndex,
    region_count: regions.length,
    text_candidate_count: textCandidates.length,
    regions: compact(regions),
  });

  return { regions, textCandidates };
}

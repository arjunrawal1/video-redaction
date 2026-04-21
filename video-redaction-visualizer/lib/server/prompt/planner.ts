import { createHash } from "node:crypto";
import { generateObject } from "ai";
import sharp from "sharp";
import { z } from "zod";
import { extractRawOcrItems } from "@/lib/server/agentic-ocr";
import { fetchDeduplicatedFramesServer } from "@/lib/server/frames";
import { fetchOcrRaw } from "@/lib/server/ocr-client";
import {
  agenticLanguageModel,
  agenticModelSlug,
  agenticProviderOptions,
  agenticThinkingLevel,
} from "@/lib/server/openrouter";
import { compact, type RunLog } from "@/lib/server/run-log";
import {
  assignBranchIds,
  hashPredicate,
  PredicateSchema,
  type Predicate,
  type PredicateHash,
  type SceneSummary,
} from "./types";
import {
  getSceneSummary,
  makeSceneSummaryCacheKey,
  putSceneSummary,
} from "./scene-summary-cache";

function parseIntEnv(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : fallback;
}

export const PREDICATE_PLANNER_SAMPLE_FRAMES = parseIntEnv(
  "PREDICATE_PLANNER_SAMPLE_FRAMES",
  8,
);

export const PREDICATE_PLANNER_OCR_CONCURRENCY = parseIntEnv(
  "PREDICATE_PLANNER_OCR_CONCURRENCY",
  4,
);

const PLANNER_CACHE_MAX_ENTRIES = parseIntEnv(
  "PREDICATE_PLANNER_CACHE_MAX_ENTRIES",
  64,
);

type PlannerCacheEntry = { predicate: Predicate; hash: PredicateHash };

const _plannerCache = new Map<string, PlannerCacheEntry>();

function plannerKey(args: { prompt: string; videoHash: string }): string {
  const raw = `${args.prompt.trim().toLowerCase()}|${args.videoHash}`;
  return createHash("sha256").update(raw).digest("hex");
}

function touchPlannerKey(key: string): void {
  const hit = _plannerCache.get(key);
  if (!hit) return;
  _plannerCache.delete(key);
  _plannerCache.set(key, hit);
}

function putPlannerCache(key: string, value: PlannerCacheEntry): void {
  _plannerCache.set(key, value);
  touchPlannerKey(key);
  while (_plannerCache.size > PLANNER_CACHE_MAX_ENTRIES) {
    const oldest = _plannerCache.keys().next().value;
    if (oldest == null) break;
    _plannerCache.delete(oldest);
  }
}

async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (items.length === 0) return [];
  const out = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const idx = cursor++;
      if (idx >= items.length) return;
      out[idx] = await fn(items[idx], idx);
    }
  });
  await Promise.all(workers);
  return out;
}

function sampleEvenly(lo: number, hi: number, maxFrames: number): number[] {
  if (hi < lo) return [];
  const count = hi - lo + 1;
  if (count <= maxFrames) {
    return Array.from({ length: count }, (_, i) => lo + i);
  }
  const step = (count - 1) / (maxFrames - 1);
  const picked = new Set<number>();
  for (let i = 0; i < maxFrames; i++) {
    const idx = Math.round(lo + step * i);
    picked.add(Math.max(lo, Math.min(hi, idx)));
  }
  return Array.from(picked).sort((a, b) => a - b);
}

function extractAnchors(raw: unknown): string[] {
  const items = extractRawOcrItems(raw);
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    const text = item.text.trim();
    if (!text) continue;
    const words = text.split(/\s+/).filter(Boolean);
    const lower = text.toLowerCase();
    const isLikelyHeader =
      item.type === "line" &&
      item.confidence >= 90 &&
      words.length <= 6 &&
      words.length >= 1;
    const isShortKeyword =
      item.type === "word" &&
      item.confidence >= 90 &&
      words.length === 1 &&
      text.length >= 2 &&
      text.length <= 24;
    if (!isLikelyHeader && !isShortKeyword) continue;
    if (seen.has(lower)) continue;
    seen.add(lower);
    out.push(text);
    if (out.length >= 40) break;
  }
  return out;
}

function buildNotes(args: {
  sampleFrameIndices: number[];
  anchorsPerFrame: { frame_index: number; anchors: string[] }[];
  globalAnchors: { text: string; seen_on: number[] }[];
}): string[] {
  const notes: string[] = [];
  notes.push(
    `Sampled frames: ${args.sampleFrameIndices.join(", ")} (1-indexed deduplicated frame indices).`,
  );
  if (args.globalAnchors.length > 0) {
    const top = args.globalAnchors
      .slice(0, 20)
      .map((a) => a.text)
      .join(", ");
    notes.push(`Top visible anchors: ${top}`);
  }
  for (const row of args.anchorsPerFrame) {
    if (row.anchors.length === 0) continue;
    notes.push(`Frame ${row.frame_index} anchors: ${row.anchors.slice(0, 12).join(", ")}`);
  }
  return notes;
}

async function downsampleJpeg(blob: Uint8Array): Promise<Uint8Array> {
  try {
    const out = await sharp(blob)
      .resize({ width: 1280, height: 720, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 80 })
      .toBuffer();
    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
  } catch {
    return blob;
  }
}

export async function buildSceneSummary(args: {
  file: File;
  frameFrom?: number;
  frameTo?: number;
  fps?: number | null;
  dedupThreshold?: number;
  maxGap?: number;
  maxFrames?: number;
  runLog?: RunLog | null;
}): Promise<SceneSummary> {
  const maxFrames = args.maxFrames ?? PREDICATE_PLANNER_SAMPLE_FRAMES;
  const dedupThreshold = args.dedupThreshold ?? 2;
  const maxGap = args.maxGap ?? 1;
  const runLog = args.runLog ?? null;

  const framesRes = await fetchDeduplicatedFramesServer({
    file: args.file,
    fps: args.fps ?? undefined,
    dedupThreshold,
    maxGap,
  });

  const total = framesRes.deduplicatedCount;
  if (total <= 0) {
    throw new Error("No deduplicated frames available for planning.");
  }

  const lo = Math.max(1, Math.min(total, Math.floor(args.frameFrom ?? 1)));
  const hi = Math.max(lo, Math.min(total, Math.floor(args.frameTo ?? total)));

  const sampled = sampleEvenly(lo, hi, Math.max(1, maxFrames));

  const cacheKey = makeSceneSummaryCacheKey({
    videoHash: framesRes.videoHash,
    maxFrames: Math.max(1, maxFrames),
    frameFrom: lo,
    frameTo: hi,
    fps: args.fps ?? null,
    dedupThreshold,
    maxGap,
  });
  const cached = getSceneSummary(cacheKey);
  if (cached) {
    runLog?.write({
      kind: "scene_summary_cache_hit",
      video_hash: framesRes.videoHash,
      cache_key: cacheKey,
      sample_frame_indices: cached.sample_frame_indices,
      total_dedup_frames: cached.total_dedup_frames,
      global_anchor_count: cached.global_anchors.length,
    });
    return cached;
  }

  const ocrResults = await mapLimit(
    sampled,
    PREDICATE_PLANNER_OCR_CONCURRENCY,
    async (idx1) => {
      const ocr = await fetchOcrRaw(args.file, {
        frameFrom: idx1,
        frameTo: idx1,
        fps: args.fps ?? null,
        dedupThreshold,
        maxGap,
      });
      const hit = ocr.frames.find((f) => f.index === idx1);
      return { index: idx1, raw: hit?.raw ?? null };
    },
  );

  const anchorsPerFrame = ocrResults.map((row) => ({
    frame_index: row.index,
    anchors: extractAnchors(row.raw),
  }));

  const anchorToFrames = new Map<string, { text: string; seen_on: number[] }>();
  for (const row of anchorsPerFrame) {
    for (const anchor of row.anchors) {
      const key = anchor.toLowerCase();
      const existing = anchorToFrames.get(key);
      if (existing) {
        if (!existing.seen_on.includes(row.frame_index)) {
          existing.seen_on.push(row.frame_index);
        }
      } else {
        anchorToFrames.set(key, { text: anchor, seen_on: [row.frame_index] });
      }
    }
  }

  const globalAnchors = Array.from(anchorToFrames.values())
    .map((v) => ({ text: v.text, seen_on: v.seen_on.slice().sort((a, b) => a - b) }))
    .sort((a, b) => {
      if (b.seen_on.length !== a.seen_on.length) return b.seen_on.length - a.seen_on.length;
      return a.text.localeCompare(b.text);
    });

  const sampleFramesB64 = await mapLimit(sampled, 3, async (idx1) => {
    const src = framesRes.frames[idx1 - 1];
    const blob = src?.blob ?? new Uint8Array();
    const downsized = await downsampleJpeg(blob);
    return {
      frame_index: idx1,
      jpeg_b64: Buffer.from(downsized).toString("base64"),
    };
  });

  const notes = buildNotes({
    sampleFrameIndices: sampled,
    anchorsPerFrame,
    globalAnchors,
  });

  const summary: SceneSummary = {
    video_hash: framesRes.videoHash,
    total_dedup_frames: total,
    sample_frame_indices: sampled,
    anchors_per_frame: anchorsPerFrame,
    global_anchors: globalAnchors,
    sample_frames_b64: sampleFramesB64,
    notes,
  };

  putSceneSummary(cacheKey, summary);
  runLog?.write({
    kind: "scene_summary_built",
    video_hash: framesRes.videoHash,
    cache_key: cacheKey,
    total_dedup_frames: total,
    sample_frame_indices: sampled,
    global_anchors: compact(globalAnchors),
    anchors_per_frame: compact(anchorsPerFrame),
    notes,
  });
  return summary;
}

const PlannerOutputSchema = z.object({
  predicate: z.any(),
});

function fallbackPredicate(prompt: string): Predicate {
  const trimmed = prompt.trim();
  return {
    kind: "text",
    text: trimmed || "<UNRESOLVED>",
    fuzzy: true,
  };
}

export async function parsePromptToPredicate(opts: {
  prompt: string;
  sceneSummary: SceneSummary;
  runLog?: RunLog | null;
}): Promise<{
  predicate: Predicate;
  hash: PredicateHash;
}> {
  const runLog = opts.runLog ?? null;
  const pKey = plannerKey({
    prompt: opts.prompt,
    videoHash: opts.sceneSummary.video_hash,
  });
  const cacheHit = _plannerCache.get(pKey);
  if (cacheHit) {
    touchPlannerKey(pKey);
    runLog?.write({
      kind: "planner_cache_hit",
      prompt: opts.prompt,
      video_hash: opts.sceneSummary.video_hash,
      predicate: compact(cacheHit.predicate),
      predicate_hash: cacheHit.hash,
    });
    return cacheHit;
  }

  const model = agenticLanguageModel();
  const system = [
    "You convert a user's redaction instruction into a structured predicate describing EXACTLY what to redact.",
    "You are shown sampled video frames plus OCR anchors from across the video.",
    "",
    "HARD RULES — follow these without exception:",
    "",
    "1. NEVER ask clarification questions. Never request more information. Make the most literal, best-informed interpretation of the user's prompt using the scene context and proceed. The user has already said what they want; your job is to execute it.",
    "",
    "2. Take the prompt LITERALLY. If the user says \"redact X inside Y\", redact only X (the contents inside Y), NOT Y itself and NOT Y's surrounding chrome.",
    "   - \"inside a cell of the spreadsheet\" = only the text characters that sit inside spreadsheet data cells. It does NOT include column/row headers (A, B, 1, 2, …), the formula bar, sheet tabs, the toolbar, menu bar, filenames, browser chrome, dock icons, or any app UI surrounding the sheet.",
    "   - \"in the document body\" = only body text, not titles/metadata/menus.",
    "   - If the user says \"redact the email addresses\", redact only values that look like email addresses — not field labels like \"Email:\" next to them.",
    "",
    "3. NEVER emit a predicate that would redact an entire frame or broadly match UI chrome. If you can't find a more specific grounding, prefer a `semantic` leaf describing what the USER WROTE (verbatim), not a `region` leaf with a vague description. A `region` leaf MUST describe something visually precise enough that a localizer can draw tight bounding boxes for each instance (e.g. \"data cells of the visible spreadsheet grid, excluding the header row and header column\"). Never use bare region descriptions like \"spreadsheet\", \"page\", \"document\", or \"UI\".",
    "",
    "4. Choose the leaf kind that matches the user's intent:",
    "   - `text` for exact/fuzzy literal strings the user named.",
    "   - `region` for tightly specified repeating visual zones (cells, rows, cards). When using `region`, set `all_instances: true` for repeating zones and put a precise, exclusion-aware description in `description`. Populate `anchors` with any visible labels that help the localizer find the zone (e.g. column letters, table headers).",
    "   - `semantic` for content categories (emails, phone numbers, dollar amounts, names, etc.). Prefer `semantic` over `region` when the user is describing what kind of content to hide rather than where it sits.",
    "",
    "5. Use `and`/`or`/`not` only when the prompt explicitly combines constraints.",
    "",
    "Return JSON of the form { \"predicate\": <Predicate> }. Do not include any other fields.",
  ].join("\n");

  const summaryText = {
    prompt: opts.prompt,
    notes: opts.sceneSummary.notes,
    anchors_per_frame: opts.sceneSummary.anchors_per_frame,
    global_anchors: opts.sceneSummary.global_anchors,
  };
  const userText =
    "Return JSON with { predicate }. Do not ask for clarification.\n" +
    JSON.stringify(summaryText, null, 2);

  runLog?.write({
    kind: "planner_request",
    model: agenticModelSlug(),
    thinking_level: agenticThinkingLevel(),
    prompt: opts.prompt,
    video_hash: opts.sceneSummary.video_hash,
    sample_frame_indices: opts.sceneSummary.sample_frame_indices,
    sample_frame_jpeg_bytes: opts.sceneSummary.sample_frames_b64.map((f) => ({
      frame_index: f.frame_index,
      // jpeg_b64 is base64; byte length ≈ 3/4 of that.
      jpeg_bytes: Math.floor((f.jpeg_b64.length * 3) / 4),
    })),
    system_prompt: system,
    user_text: userText,
  });

  const t0 = Date.now();
  let predicate: Predicate;
  let usedFallback = false;
  try {
    const response = await generateObject({
      model,
      schema: PlannerOutputSchema,
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
            ...opts.sceneSummary.sample_frames_b64.map((f) => ({
              type: "image" as const,
              image: Buffer.from(f.jpeg_b64, "base64"),
              mediaType: "image/jpeg" as const,
            })),
          ],
        },
      ],
    });

    const parsed = response.object;
    const rawPredicate = parsed.predicate;
    if (rawPredicate && typeof rawPredicate === "object") {
      try {
        predicate = PredicateSchema.parse(rawPredicate);
      } catch (e) {
        runLog?.write({
          kind: "planner_schema_error",
          error: e instanceof Error ? e.message : String(e),
          raw_predicate: compact(rawPredicate),
        });
        predicate = fallbackPredicate(opts.prompt);
        usedFallback = true;
      }
    } else {
      predicate = fallbackPredicate(opts.prompt);
      usedFallback = true;
    }

    predicate = assignBranchIds(predicate);

    runLog?.write({
      kind: "planner_response",
      elapsed_ms: Date.now() - t0,
      finish_reason: response.finishReason,
      warnings: response.warnings,
      usage: response.usage,
      used_fallback: usedFallback,
      predicate: compact(predicate),
      raw_parsed: compact(parsed),
    });
  } catch (e) {
    predicate = assignBranchIds(fallbackPredicate(opts.prompt));
    usedFallback = true;
    runLog?.write({
      kind: "planner_error",
      elapsed_ms: Date.now() - t0,
      error: e instanceof Error ? e.message : String(e),
      fallback_predicate: compact(predicate),
    });
  }

  const hash = hashPredicate(predicate);
  const out: PlannerCacheEntry = { predicate, hash };
  putPlannerCache(pKey, out);
  return out;
}

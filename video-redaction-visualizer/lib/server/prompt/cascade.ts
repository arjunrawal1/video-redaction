import { agenticCascadeMaxAgents, agenticCascadeMaxDepth } from "@/lib/server/openrouter";
import { alog } from "@/lib/server/agentic-log";
import type { NavFrameState, NavHit, NavigatorEvent } from "@/lib/server/agentic-navigator";
import { compact, type RunLog } from "@/lib/server/run-log";
import { filterOcr } from "./tools";
import { collectLeafPredicates, type Predicate, type PredicateLeaf, type RegionBbox, type ResolvedCandidate } from "./types";

function parseIntEnv(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : fallback;
}

const PROMPT_CASCADE_MAX_AGENTS = parseIntEnv(
  "PROMPT_CASCADE_MAX_AGENTS",
  Math.max(1, agenticCascadeMaxAgents() * 4),
);
const PROMPT_CASCADE_MAX_DEPTH = parseIntEnv(
  "PROMPT_CASCADE_MAX_DEPTH",
  agenticCascadeMaxDepth(),
);

const HIT_PREFIX = "P";
let _hitCounter = 0;

function mintHitId(): string {
  _hitCounter = (_hitCounter + 1) % 1_000_000;
  return `${HIT_PREFIX}${Date.now().toString(36)}${_hitCounter.toString(36)}`;
}

type PromptSeed = {
  instance_id: string;
  direction: "backward" | "forward";
  first_focus: number;
  source_frame: number;
  source_hit: NavHit;
};

export function findPromptChainSeeds(frames: NavFrameState[]): PromptSeed[] {
  const seeds: PromptSeed[] = [];
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1];
    const cur = frames[i];
    const prevById = new Map(
      prev.hits
        .filter((h) => h.instance_id)
        .map((h) => [h.instance_id as string, h]),
    );
    const curById = new Map(
      cur.hits
        .filter((h) => h.instance_id)
        .map((h) => [h.instance_id as string, h]),
    );

    for (const [id, hit] of curById.entries()) {
      if (prevById.has(id)) continue;
      seeds.push({
        instance_id: id,
        direction: "backward",
        first_focus: cur.index - 1,
        source_frame: cur.index,
        source_hit: hit,
      });
    }

    for (const [id, hit] of prevById.entries()) {
      if (curById.has(id)) continue;
      seeds.push({
        instance_id: id,
        direction: "forward",
        first_focus: cur.index,
        source_frame: prev.index,
        source_hit: hit,
      });
    }
  }
  return seeds;
}

function branchFromInstance(instanceId: string): string {
  const idx = instanceId.indexOf(":");
  return idx > 0 ? instanceId.slice(0, idx) : instanceId;
}

function subIdFromInstance(instanceId: string): string {
  const idx = instanceId.indexOf(":");
  return idx > 0 ? instanceId.slice(idx + 1) : "0";
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function similarity(a: string, b: string): number {
  if (a === b) return 1;
  if (!a.length || !b.length) return 0;
  let shared = 0;
  for (const ch of new Set(a)) {
    if (b.includes(ch)) shared += 1;
  }
  return shared / Math.max(a.length, b.length);
}

async function attemptRecover(args: {
  frame: NavFrameState;
  seed: PromptSeed;
  branchLeaf: PredicateLeaf | null;
  predicate: Predicate;
  regions: RegionBbox[];
  runLog?: RunLog | null;
}): Promise<NavHit[]> {
  const out: NavHit[] = [];
  const branch = branchFromInstance(args.seed.instance_id);
  const subId = subIdFromInstance(args.seed.instance_id);

  // Text recovery: substring/character overlap search on raw OCR.
  if (args.branchLeaf?.kind === "text") {
    const needle = args.seed.source_hit.text;
    const items = await filterOcr({
      ctx: {
        videoHash: "",
        frameIndex: args.frame.index,
        frameBlob: args.frame.blob,
        width: args.frame.width,
        height: args.frame.height,
        rawOcr: args.frame.ocrRaw,
        predicate: args.predicate,
        regions: args.regions,
        textCandidates: [] as ResolvedCandidate[],
        runLog: args.runLog ?? null,
      },
      textSubstring: needle.slice(0, Math.max(1, Math.min(needle.length, 6))),
      maxItems: 40,
    });
    for (const item of items) {
      if (similarity(normalize(item.text), normalize(needle)) < 0.2) continue;
      out.push({
        x: Math.round((item.bbox_2d[1] / 1000) * args.frame.width),
        y: Math.round((item.bbox_2d[0] / 1000) * args.frame.height),
        w: Math.max(1, Math.round(((item.bbox_2d[3] - item.bbox_2d[1]) / 1000) * args.frame.width)),
        h: Math.max(1, Math.round(((item.bbox_2d[2] - item.bbox_2d[0]) / 1000) * args.frame.height)),
        text: item.text,
        score: item.confidence / 100,
        origin: "fix",
        hit_id: mintHitId(),
        instance_id: args.seed.instance_id,
        branch,
        category: args.seed.source_hit.category ?? null,
        track_id: args.seed.instance_id,
      });
    }
    return out;
  }

  // Region recovery: find region with same branch/sub_id and redact OCR inside it.
  if (args.branchLeaf?.kind === "region") {
    const targetRegions = args.regions.filter(
      (r) => r.branch === branch && r.sub_id === subId,
    );
    for (const region of targetRegions) {
      const items = await filterOcr({
        ctx: {
          videoHash: "",
          frameIndex: args.frame.index,
          frameBlob: args.frame.blob,
          width: args.frame.width,
          height: args.frame.height,
          rawOcr: args.frame.ocrRaw,
          predicate: args.predicate,
          regions: args.regions,
          textCandidates: [] as ResolvedCandidate[],
          runLog: args.runLog ?? null,
        },
        regionBbox: region.bbox_2d,
        maxItems: 300,
      });
      for (const item of items) {
        out.push({
          x: Math.round((item.bbox_2d[1] / 1000) * args.frame.width),
          y: Math.round((item.bbox_2d[0] / 1000) * args.frame.height),
          w: Math.max(1, Math.round(((item.bbox_2d[3] - item.bbox_2d[1]) / 1000) * args.frame.width)),
          h: Math.max(1, Math.round(((item.bbox_2d[2] - item.bbox_2d[0]) / 1000) * args.frame.height)),
          text: item.text,
          score: item.confidence / 100,
          origin: "fix",
          hit_id: mintHitId(),
          instance_id: args.seed.instance_id,
          branch,
          category: args.seed.source_hit.category ?? null,
          track_id: args.seed.instance_id,
        });
      }
    }
    return out;
  }

  // Semantic recovery: query semantic category and keep items similar to tracked value.
  if (args.branchLeaf?.kind === "semantic") {
    const items = await filterOcr({
      ctx: {
        videoHash: "",
        frameIndex: args.frame.index,
        frameBlob: args.frame.blob,
        width: args.frame.width,
        height: args.frame.height,
        rawOcr: args.frame.ocrRaw,
        predicate: args.predicate,
        regions: args.regions,
        textCandidates: [] as ResolvedCandidate[],
        runLog: args.runLog ?? null,
      },
      semantic: { description: args.branchLeaf.description, min_confidence: 0.6 },
      maxItems: 120,
    });
    for (const item of items) {
      if (similarity(normalize(item.text), normalize(args.seed.source_hit.text)) < 0.2) {
        continue;
      }
      out.push({
        x: Math.round((item.bbox_2d[1] / 1000) * args.frame.width),
        y: Math.round((item.bbox_2d[0] / 1000) * args.frame.height),
        w: Math.max(1, Math.round(((item.bbox_2d[3] - item.bbox_2d[1]) / 1000) * args.frame.width)),
        h: Math.max(1, Math.round(((item.bbox_2d[2] - item.bbox_2d[0]) / 1000) * args.frame.height)),
        text: item.text,
        score: item.semantic_score ?? item.confidence / 100,
        origin: "fix",
        hit_id: mintHitId(),
        instance_id: args.seed.instance_id,
        branch,
        category: args.branchLeaf.category,
        track_id: args.seed.instance_id,
      });
    }
  }

  return out;
}

export async function runPromptCascade(args: {
  predicate: Predicate;
  frames: NavFrameState[];
  onEvent: (ev: NavigatorEvent) => void;
  runLog?: RunLog | null;
}): Promise<{
  added: number;
  removed: number;
  totalSteps: number;
  finishSummary: string | null;
}> {
  const runLog = args.runLog ?? null;
  const frameByIndex = new Map(args.frames.map((f) => [f.index, f]));
  const leaves = collectLeafPredicates(args.predicate);
  const branchToLeaf = new Map<string, PredicateLeaf>();
  for (const leaf of leaves) {
    if (leaf.branch) branchToLeaf.set(leaf.branch, leaf);
  }

  const seeds = findPromptChainSeeds(args.frames);
  runLog?.write({
    kind: "cascade_plan",
    frame_count: args.frames.length,
    frame_indices: args.frames.map((f) => f.index),
    seed_count: seeds.length,
    seeds: compact(
      seeds.map((s) => ({
        instance_id: s.instance_id,
        direction: s.direction,
        first_focus: s.first_focus,
        source_frame: s.source_frame,
        source_text: s.source_hit.text,
      })),
    ),
    max_agents: PROMPT_CASCADE_MAX_AGENTS,
    max_depth: PROMPT_CASCADE_MAX_DEPTH,
    predicate: compact(args.predicate),
  });
  alog("prompt cascade plan", {
    frame_count: args.frames.length,
    seed_count: seeds.length,
    max_agents: PROMPT_CASCADE_MAX_AGENTS,
    max_depth: PROMPT_CASCADE_MAX_DEPTH,
  });
  let added = 0;
  const removed = 0;
  let totalSteps = 0;
  let agents = 0;

  type ClaimKey = string;
  const claimed = new Set<ClaimKey>();

  for (const seed of seeds) {
    if (agents >= PROMPT_CASCADE_MAX_AGENTS) {
      runLog?.write({
        kind: "cascade_max_agents_reached",
        agents,
        max_agents: PROMPT_CASCADE_MAX_AGENTS,
      });
      alog("prompt cascade max agents reached", {
        agents,
        max_agents: PROMPT_CASCADE_MAX_AGENTS,
      });
      break;
    }
    const agentId = `prompt-${seed.direction}-${seed.instance_id}-${seed.first_focus}`;
    agents += 1;

    runLog?.write({
      kind: "cascade_agent_start",
      agent_id: agentId,
      instance_id: seed.instance_id,
      direction: seed.direction,
      first_focus: seed.first_focus,
      source_frame: seed.source_frame,
      source_text: seed.source_hit.text,
    });
    alog(`[${agentId}] start`, {
      instance_id: seed.instance_id,
      direction: seed.direction,
      first_focus: seed.first_focus,
      source_frame: seed.source_frame,
      source_text: seed.source_hit.text,
    });
    args.onEvent({
      type: "agent_start",
      agent_id: agentId,
      focus_frame: seed.first_focus,
      source: "cascade",
      parent_agent_id: null,
      reason: `instance transition for ${seed.instance_id}`,
    });

    let current = seed.first_focus;
    let depth = 0;
    let stepAdded = 0;

    while (depth < PROMPT_CASCADE_MAX_DEPTH) {
      const frame = frameByIndex.get(current);
      if (!frame) break;
      const key: ClaimKey = `${seed.direction}|${seed.instance_id}|${current}`;
      if (claimed.has(key)) break;
      claimed.add(key);

      totalSteps += 1;
      const existing = frame.hits.filter((h) => h.instance_id === seed.instance_id);
      let stillVisible = existing.length > 0;

      const branch = branchFromInstance(seed.instance_id);
      const branchLeaf = branchToLeaf.get(branch) ?? null;
      if (!stillVisible) {
        const regions = (frame as NavFrameState & { regions?: RegionBbox[] }).regions ?? [];
        const recovered = await attemptRecover({
          frame,
          seed,
          branchLeaf,
          predicate: args.predicate,
          regions,
          runLog,
        });
        runLog?.write({
          kind: "cascade_recover_attempt",
          agent_id: agentId,
          frame_index: current,
          branch_leaf_kind: branchLeaf?.kind ?? null,
          recovered_count: recovered.length,
          recovered: compact(recovered.map((r) => ({ text: r.text, bbox: [r.x, r.y, r.w, r.h] }))),
        });
        for (const hit of recovered) {
          if (
            frame.hits.some(
              (h) =>
                h.instance_id === hit.instance_id &&
                Math.abs(h.x - hit.x) <= 1 &&
                Math.abs(h.y - hit.y) <= 1 &&
                Math.abs(h.w - hit.w) <= 1 &&
                Math.abs(h.h - hit.h) <= 1,
            )
          ) {
            continue;
          }
          frame.hits.push(hit);
          added += 1;
          stepAdded += 1;
          stillVisible = true;
          args.onEvent({
            type: "frame_update",
            index: frame.index,
            action: "add",
            hit,
            reason: "prompt cascade recovered instance",
            agent_id: agentId,
          });
        }
      }

      if (!stillVisible) break;
      current += seed.direction === "backward" ? -1 : 1;
      depth += 1;
    }

    runLog?.write({
      kind: "cascade_agent_end",
      agent_id: agentId,
      instance_id: seed.instance_id,
      direction: seed.direction,
      added: stepAdded,
      total_steps_in_chain: Math.max(1, depth),
    });
    alog(`[${agentId}] end`, {
      instance_id: seed.instance_id,
      direction: seed.direction,
      added: stepAdded,
      total_steps_in_chain: Math.max(1, depth),
    });
    args.onEvent({
      type: "agent_end",
      agent_id: agentId,
      focus_frame: seed.first_focus,
      added: stepAdded,
      removed: 0,
      total_steps: Math.max(1, depth),
      finish_summary: stepAdded > 0 ? "Recovered missing prompt instances." : "No additional partials found.",
      cost_usd: 0,
    });
  }

  const finishSummary =
    added > 0
      ? `Prompt cascade added ${added} box${added === 1 ? "" : "es"} across instance chains.`
      : "Prompt cascade found no extra boxes.";

  runLog?.write({
    kind: "cascade_finish",
    added,
    removed,
    total_steps: totalSteps,
    agents,
    summary: finishSummary,
  });
  alog("prompt cascade finish", {
    added,
    removed,
    total_steps: totalSteps,
    agents,
    summary: finishSummary,
  });
  args.onEvent({
    type: "finish",
    summary: finishSummary,
    total_steps: totalSteps,
  });

  return {
    added,
    removed,
    totalSteps,
    finishSummary,
  };
}

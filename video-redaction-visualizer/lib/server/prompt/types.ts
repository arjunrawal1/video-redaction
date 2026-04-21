import { createHash } from "node:crypto";
import { z } from "zod";

export type BranchId = string;

export type PredicateLeaf =
  | {
      kind: "text";
      text: string;
      fuzzy: boolean;
      branch?: BranchId;
      category?: string | null;
    }
  | {
      kind: "region";
      description: string;
      all_instances: boolean;
      anchors?: string[];
      branch?: BranchId;
      category?: string | null;
    }
  | {
      kind: "semantic";
      category: string;
      description: string;
      examples?: string[];
      branch?: BranchId;
    };

export type Predicate =
  | PredicateLeaf
  | { kind: "and" | "or"; children: Predicate[] }
  | { kind: "not"; child: Predicate };

export type Instance = {
  id: string;
  branch: BranchId;
  descriptor: string;
  category: string | null;
};

export type ResolvedCandidate = {
  ocr_index: number;
  branch: BranchId;
  bbox: [number, number, number, number];
  text: string;
  prior_score: number;
};

export type RegionBbox = {
  branch: BranchId;
  sub_id: string;
  bbox_2d: [number, number, number, number];
  confidence: number;
  reason: string;
};

export type PlanSession = {
  session_id: string;
  prompt: string;
  predicate: Predicate;
  hash?: PredicateHash;
  sample_frame_indices?: number[];
};

export type SceneSummary = {
  video_hash: string;
  total_dedup_frames: number;
  sample_frame_indices: number[];
  anchors_per_frame: { frame_index: number; anchors: string[] }[];
  global_anchors: { text: string; seen_on: number[] }[];
  sample_frames_b64: { frame_index: number; jpeg_b64: string }[];
  notes: string[];
};

export type PredicateHash = string;

const TextLeafSchema = z.object({
  kind: z.literal("text"),
  text: z.string().min(1),
  fuzzy: z.boolean(),
  branch: z.string().optional(),
  category: z.string().nullable().optional(),
});

const RegionLeafSchema = z.object({
  kind: z.literal("region"),
  description: z.string().min(1),
  all_instances: z.boolean(),
  anchors: z.array(z.string()).optional(),
  branch: z.string().optional(),
  category: z.string().nullable().optional(),
});

const SemanticLeafSchema = z.object({
  kind: z.literal("semantic"),
  category: z.string().min(1),
  description: z.string().min(1),
  examples: z.array(z.string()).optional(),
  branch: z.string().optional(),
});

export const PredicateSchema: z.ZodType<Predicate> = z.lazy(() =>
  z.union([
    TextLeafSchema,
    RegionLeafSchema,
    SemanticLeafSchema,
    z.object({
      kind: z.literal("and"),
      children: z.array(PredicateSchema).min(1),
    }),
    z.object({
      kind: z.literal("or"),
      children: z.array(PredicateSchema).min(1),
    }),
    z.object({
      kind: z.literal("not"),
      child: PredicateSchema,
    }),
  ]),
);

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortValue);
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const out: Record<string, unknown> = {};
    for (const k of keys) {
      out[k] = sortValue(obj[k]);
    }
    return out;
  }
  return value;
}

export function stableStringify(value: unknown): string {
  return JSON.stringify(sortValue(value));
}

export function canonicalize(predicate: Predicate): string {
  return stableStringify(predicate);
}

export function hashPredicate(predicate: Predicate): PredicateHash {
  return createHash("sha256").update(canonicalize(predicate)).digest("hex");
}

export function parsePredicateJson(raw: string): Predicate {
  const parsed = JSON.parse(raw) as unknown;
  return PredicateSchema.parse(parsed);
}

function assignBranchesInner(
  predicate: Predicate,
  counter: { value: number },
): Predicate {
  if (
    predicate.kind === "text" ||
    predicate.kind === "region" ||
    predicate.kind === "semantic"
  ) {
    const branch = `L${counter.value++}`;
    return { ...predicate, branch };
  }
  if (predicate.kind === "not") {
    return { ...predicate, child: assignBranchesInner(predicate.child, counter) };
  }
  return {
    ...predicate,
    children: predicate.children.map((c) => assignBranchesInner(c, counter)),
  };
}

export function assignBranchIds(predicate: Predicate): Predicate {
  return assignBranchesInner(predicate, { value: 0 });
}

export function collectLeafPredicates(predicate: Predicate): PredicateLeaf[] {
  if (
    predicate.kind === "text" ||
    predicate.kind === "region" ||
    predicate.kind === "semantic"
  ) {
    return [predicate];
  }
  if (predicate.kind === "not") {
    return collectLeafPredicates(predicate.child);
  }
  const out: PredicateLeaf[] = [];
  for (const child of predicate.children) {
    out.push(...collectLeafPredicates(child));
  }
  return out;
}

export function hasUnresolvedPredicate(predicate: Predicate): boolean {
  for (const leaf of collectLeafPredicates(predicate)) {
    if (leaf.kind === "region" && /<UNRESOLVED>/i.test(leaf.description)) {
      return true;
    }
  }
  return false;
}

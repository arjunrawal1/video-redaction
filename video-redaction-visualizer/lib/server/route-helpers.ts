// Shared helpers for the three Gemini route handlers.
import {
  hashPredicate,
  parsePredicateJson,
  type Predicate,
} from "./prompt/types";

export type FormInputs = {
  file: File;
  query: string;
  frameFrom: number;
  frameTo: number | null;
  fps: number | null;
  dedupThreshold: number;
  maxGap: number;
};

export type PromptFormInputs = {
  file: File;
  prompt: string;
  predicate: Predicate | null;
  predicateHash: string | null;
  frameFrom: number;
  frameTo: number | null;
  fps: number | null;
  dedupThreshold: number;
  maxGap: number;
};

export async function readFormInputs(
  req: Request,
): Promise<FormInputs | { error: string }> {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return { error: "Expected multipart/form-data." };
  }
  const file = form.get("file");
  if (!(file instanceof File)) return { error: "Missing file upload." };
  const query = String(form.get("query") ?? "").trim();
  if (!query) return { error: "Missing query." };
  const frameFromRaw = form.get("frame_from");
  const frameToRaw = form.get("frame_to");
  const fpsRaw = form.get("fps");
  const dedupRaw = form.get("dedup_threshold");
  const maxGapRaw = form.get("max_gap");

  const asNum = (v: FormDataEntryValue | null): number | null => {
    if (v == null) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  const frameFrom = Math.max(1, Math.floor(asNum(frameFromRaw) ?? 1));
  const frameTo = asNum(frameToRaw) == null ? null : Math.floor(asNum(frameToRaw)!);
  const fps = asNum(fpsRaw) == null ? null : (asNum(fpsRaw) as number);
  // Keep in sync with `_DEFAULT_DEDUP_THRESHOLD` in
  // backend/app/frame_service.py — this default is part of the cache key
  // for both the Python frame cache and the Gemini/teamwork caches, so
  // drift here would silently invalidate everything.
  const dedupThreshold = Math.floor(asNum(dedupRaw) ?? 2);
  // Keep in sync with `_DEFAULT_MAX_GAP` in backend/app/frame_service.py.
  // Also part of the cache key in Python frame_cache, ocr_cache, and the
  // TS gemini-cache — the kept-frame sequence differs across values of
  // this knob, so we must not alias runs with different max_gap.
  const maxGap = Math.max(0, Math.floor(asNum(maxGapRaw) ?? 1));

  return {
    file,
    query,
    frameFrom,
    frameTo,
    fps,
    dedupThreshold,
    maxGap,
  };
}

export async function readPromptFormInputs(
  req: Request,
): Promise<PromptFormInputs | { error: string }> {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return { error: "Expected multipart/form-data." };
  }
  const file = form.get("file");
  if (!(file instanceof File)) return { error: "Missing file upload." };
  const prompt = String(form.get("prompt") ?? "").trim();
  const predicateJson = form.get("predicate_json");
  const statedHash = String(form.get("predicate_hash") ?? "").trim();

  const frameFromRaw = form.get("frame_from");
  const frameToRaw = form.get("frame_to");
  const fpsRaw = form.get("fps");
  const dedupRaw = form.get("dedup_threshold");
  const maxGapRaw = form.get("max_gap");

  const asNum = (v: FormDataEntryValue | null): number | null => {
    if (v == null) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  const frameFrom = Math.max(1, Math.floor(asNum(frameFromRaw) ?? 1));
  const frameTo = asNum(frameToRaw) == null ? null : Math.floor(asNum(frameToRaw)!);
  const fps = asNum(fpsRaw) == null ? null : (asNum(fpsRaw) as number);
  const dedupThreshold = Math.floor(asNum(dedupRaw) ?? 2);
  const maxGap = Math.max(0, Math.floor(asNum(maxGapRaw) ?? 1));

  let predicate: Predicate | null = null;
  let predicateHash: string | null = null;
  if (typeof predicateJson === "string" && predicateJson.trim()) {
    try {
      predicate = parsePredicateJson(predicateJson);
      predicateHash = hashPredicate(predicate);
    } catch (e) {
      return {
        error:
          e instanceof Error
            ? `Invalid predicate_json: ${e.message}`
            : "Invalid predicate_json.",
      };
    }
    if (statedHash && predicateHash !== statedHash) {
      return {
        error: `predicate_hash mismatch (expected ${predicateHash}, got ${statedHash}).`,
      };
    }
  }

  return {
    file,
    prompt,
    predicate,
    predicateHash,
    frameFrom,
    frameTo,
    fps,
    dedupThreshold,
    maxGap,
  };
}

export function normalizeQuery(q: string): string {
  return q.toLowerCase().replace(/\s+/g, " ").trim();
}

export function ndjsonLine(obj: unknown): string {
  return JSON.stringify(obj) + "\n";
}

const NDJSON_HEADERS = {
  "Content-Type": "application/x-ndjson",
  "Cache-Control": "no-cache, no-transform",
  "X-Accel-Buffering": "no",
};

/**
 * Wrap an async generator that yields NDJSON strings into a Response with the
 * right headers. Abort signal: when the client disconnects, Node will cancel
 * the stream and we swallow the resulting AbortError.
 */
export function ndjsonStreamResponse(
  produce: (ctrl: {
    emit: (obj: unknown) => void;
    error: (message: string) => void;
    done: () => void;
  }) => Promise<void>,
): Response {
  const stream = new ReadableStream({
    async start(ctrl) {
      const enc = new TextEncoder();
      const emit = (obj: unknown) => {
        try {
          ctrl.enqueue(enc.encode(ndjsonLine(obj)));
        } catch {
          /* client disconnected */
        }
      };
      const errorOnce = (message: string) => {
        emit({ type: "error", message });
      };
      try {
        await produce({ emit, error: errorOnce, done: () => {} });
      } catch (e) {
        errorOnce(e instanceof Error ? e.message : String(e));
      } finally {
        try {
          ctrl.close();
        } catch {
          /* already closed */
        }
      }
    },
  });
  return new Response(stream, { headers: NDJSON_HEADERS });
}

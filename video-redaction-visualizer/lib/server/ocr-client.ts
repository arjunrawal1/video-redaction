// Server-to-server client for the Python OCR endpoints. Used by teamwork
// routes to fetch OCR's phase-1 results (boxes + raw Textract response per
// frame) without re-implementing Textract in TypeScript.

import { pythonApiBaseUrl } from "./frames";
import type { ServerBox } from "./openrouter";

export type OcrFrameResult = {
  index: number;
  width: number;
  height: number;
  boxes: ServerBox[];
  raw: unknown;
};

function buildForm(
  file: File,
  query: string,
  opts: {
    frameFrom?: number;
    frameTo?: number;
    fps?: number | null;
    dedupThreshold?: number;
    maxGap?: number;
  },
): FormData {
  const form = new FormData();
  form.append("file", file);
  form.append("query", query);
  if (opts.frameFrom != null) form.append("frame_from", String(opts.frameFrom));
  if (opts.frameTo != null) form.append("frame_to", String(opts.frameTo));
  if (opts.fps != null && opts.fps > 0) {
    form.append("fps", String(opts.fps));
  }
  if (opts.dedupThreshold != null) {
    form.append("dedup_threshold", String(opts.dedupThreshold));
  }
  if (opts.maxGap != null) {
    form.append("max_gap", String(opts.maxGap));
  }
  return form;
}

/**
 * POST to the Python OCR detect stream and collect all per-frame results.
 *
 * Streaming isn't exposed to callers for simplicity: we consume the whole
 * NDJSON stream and return once done. For typical 8-frame ranges the
 * Python side is parallelized with concurrency=8 so wall time is ~12 s.
 */
export async function fetchOcrDetect(
  file: File,
  query: string,
  opts: {
    frameFrom?: number;
    frameTo?: number;
    fps?: number | null;
    dedupThreshold?: number;
    maxGap?: number;
  },
): Promise<{ videoHash: string; frames: OcrFrameResult[] }> {
  const base = pythonApiBaseUrl();
  const res = await fetch(`${base}/api/ocr/detect/stream`, {
    method: "POST",
    body: buildForm(file, query, opts),
  });
  if (!res.ok || !res.body) {
    const text = res.body ? await res.text().catch(() => "") : "";
    throw new Error(
      `Python OCR detect failed: ${res.status}${text ? ` ${text}` : ""}`,
    );
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let videoHash = "";
  const frames: OcrFrameResult[] = [];

  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(trimmed);
    } catch {
      return;
    }
    if (ev.type === "start" && typeof ev.video_hash === "string") {
      videoHash = ev.video_hash;
      return;
    }
    if (ev.type === "frame") {
      const boxes = Array.isArray(ev.boxes)
        ? (ev.boxes as Array<Record<string, number | string>>).map(
            (b) =>
              ({
                x: Number(b.x),
                y: Number(b.y),
                w: Number(b.w),
                h: Number(b.h),
                text: String(b.text ?? ""),
                score: Number(b.score ?? 1),
              }) as ServerBox,
          )
        : [];
      frames.push({
        index: Number(ev.index),
        width: Number(ev.width),
        height: Number(ev.height),
        boxes,
        raw: ev.raw ?? null,
      });
      return;
    }
    if (ev.type === "error") {
      throw new Error(`Python OCR detect error: ${String(ev.message ?? "?")}`);
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl = buf.indexOf("\n");
      while (nl !== -1) {
        handleLine(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
        nl = buf.indexOf("\n");
      }
    }
    buf += decoder.decode();
    if (buf.length > 0) handleLine(buf);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }

  frames.sort((a, b) => a.index - b.index);
  return { videoHash, frames };
}

/**
 * Run Textract on a single JPEG blob and return the matching boxes plus
 * the raw response. Used by the post-detection gap-filler: inserted
 * frames aren't part of any kept-frame sequence, so they can't go
 * through the range-based /api/ocr/detect/stream endpoint — this
 * per-frame call is the escape hatch.
 *
 * The caller supplies the raw JPEG bytes (not the full video) and the
 * same ``query`` that the rest of the pipeline is using. Output shape
 * matches a single entry of ``fetchOcrDetect.frames``, but without an
 * ``index`` (gap-filler assigns its own).
 */
export async function fetchOcrDetectSingle(
  jpeg: Uint8Array,
  query: string,
): Promise<{
  width: number;
  height: number;
  boxes: ServerBox[];
  raw: unknown;
}> {
  const base = pythonApiBaseUrl();
  const form = new FormData();
  // Include a filename so multer/FastAPI classify the upload as a file
  // rather than a form field. JPEG MIME chosen to match our extraction
  // pipeline; Textract accepts JPEG/PNG interchangeably either way.
  const blob = new Blob([jpeg as Uint8Array<ArrayBuffer>], {
    type: "image/jpeg",
  });
  form.append("file", blob, "frame.jpg");
  form.append("query", query);

  const res = await fetch(`${base}/api/ocr/detect/single`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Python OCR single-frame detect failed: ${res.status}${text ? ` ${text}` : ""}`,
    );
  }
  const body = (await res.json()) as {
    width: number;
    height: number;
    boxes: Array<Record<string, number | string>>;
    raw: unknown;
  };
  const boxes: ServerBox[] = (body.boxes ?? []).map(
    (b) =>
      ({
        x: Number(b.x),
        y: Number(b.y),
        w: Number(b.w),
        h: Number(b.h),
        text: String(b.text ?? ""),
        score: Number(b.score ?? 1),
      }) as ServerBox,
  );
  return {
    width: Number(body.width ?? 0),
    height: Number(body.height ?? 0),
    boxes,
    raw: body.raw ?? null,
  };
}

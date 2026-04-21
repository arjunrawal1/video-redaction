// Server-to-server helper: ask the Python backend for deduplicated frames
// and return them as decoded Uint8Array JPEGs with dimensions.
//
// We call Python's /api/frames/deduplicated which caches by SHA256(video),
// so repeat calls from multiple phases of the same run cost just the HTTP
// round-trip + base64 encode/decode — no re-extraction.

export type ServerFrame = {
  blob: Uint8Array;
  width: number;
  height: number;
};

export type FetchFramesResult = {
  videoHash: string;
  deduplicatedCount: number;
  frames: ServerFrame[];
  /**
   * 0-based index into the original ffmpeg-emitted sequence for each
   * kept frame. ``keptSourceIndices[k]`` is the raw source index of the
   * k-th kept frame. Used by the gap-filler to bisect kept-frame pairs
   * in source-index space and request specific raw frames from
   * ``/api/frames/by_source_index``. May be empty on older backends
   * that didn't return this field.
   */
  keptSourceIndices: number[];
  /**
   * Effective frame rate at which ffmpeg emitted the raw sequence. When
   * the caller passed `fps`, this is that value; otherwise ffprobe's
   * probe of the source. Required when resolving `keptSourceIndices`
   * back to timestamps in the source video (exporter, gap-filler).
   * `null` when probing failed.
   */
  sourceFps: number | null;
};

export function pythonApiBaseUrl(): string {
  return (
    process.env.PYTHON_API_BASE_URL?.replace(/\/$/, "") ||
    "http://localhost:8000"
  );
}

function base64ToUint8Array(b64: string): Uint8Array {
  if (typeof Buffer !== "undefined") {
    const buf = Buffer.from(b64, "base64");
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/**
 * Read JPEG dimensions from the marker chunks without fully decoding the
 * image. Good enough for Textract-style downstream coord math.
 */
function jpegDimensions(bytes: Uint8Array): { width: number; height: number } {
  // Step over SOI (0xFFD8) then scan markers. SOFn markers (0xC0..0xCF
  // excluding 0xC4, 0xC8, 0xCC) carry height/width.
  let i = 2;
  while (i < bytes.length) {
    if (bytes[i] !== 0xff) {
      i++;
      continue;
    }
    while (bytes[i] === 0xff && i < bytes.length) i++;
    const marker = bytes[i];
    i++;
    const isSOF =
      marker >= 0xc0 &&
      marker <= 0xcf &&
      marker !== 0xc4 &&
      marker !== 0xc8 &&
      marker !== 0xcc;
    if (marker === 0xd8 || marker === 0xd9) {
      // SOI / EOI — no payload.
      continue;
    }
    if (i + 1 >= bytes.length) break;
    const segLen = (bytes[i] << 8) | bytes[i + 1];
    if (isSOF && i + 6 < bytes.length) {
      const height = (bytes[i + 3] << 8) | bytes[i + 4];
      const width = (bytes[i + 5] << 8) | bytes[i + 6];
      return { width, height };
    }
    i += segLen;
  }
  return { width: 0, height: 0 };
}

export async function fetchDeduplicatedFramesServer(opts: {
  file: File;
  fps?: number | null;
  dedupThreshold?: number;
  maxGap?: number;
}): Promise<FetchFramesResult> {
  const form = new FormData();
  form.append("file", opts.file);
  const qs = new URLSearchParams();
  if (opts.fps != null && opts.fps > 0) qs.set("fps", String(opts.fps));
  if (opts.dedupThreshold != null) {
    qs.set("dedup_threshold", String(opts.dedupThreshold));
  }
  if (opts.maxGap != null) {
    qs.set("max_gap", String(opts.maxGap));
  }
  const suffix = qs.size ? `?${qs.toString()}` : "";

  const res = await fetch(
    `${pythonApiBaseUrl()}/api/frames/deduplicated${suffix}`,
    { method: "POST", body: form },
  );
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Python frame extraction failed: ${res.status} ${text}`);
  }
  const body = (await res.json()) as {
    video_hash?: string;
    deduplicated_count: number;
    kept_source_indices?: number[];
    source_fps?: number | null;
    frames: { mime: string; data_base64: string }[];
  };

  const frames: ServerFrame[] = body.frames.map((f) => {
    const blob = base64ToUint8Array(f.data_base64);
    const { width, height } = jpegDimensions(blob);
    return { blob, width, height };
  });

  return {
    videoHash: body.video_hash ?? "",
    deduplicatedCount: body.deduplicated_count,
    frames,
    keptSourceIndices: Array.isArray(body.kept_source_indices)
      ? body.kept_source_indices
      : [],
    sourceFps:
      typeof body.source_fps === "number" && Number.isFinite(body.source_fps)
        ? body.source_fps
        : null,
  };
}

/**
 * Fetch specific raw frames by their 0-based source index — used by the
 * post-detection gap-filler to pull frames that initial dedup discarded
 * but that now need detection because they sit inside a kept-frame pair
 * whose motion/resize exceeds the smoothness-invariant thresholds.
 *
 * The backend re-runs ffmpeg with a ``select='eq(n,i)+…'`` filter, so
 * this is a per-call operation that does not hit the frame cache. Keep
 * the caller side deduplicated (a Set of indices you actually need) to
 * avoid redundant re-extraction costs.
 *
 * ``fps`` must match the value originally passed to
 * ``fetchDeduplicatedFramesServer`` — indices are interpreted against
 * the same decoded-frame stream, so a mismatch would resolve to
 * different source frames.
 */
export async function fetchFramesBySourceIndexServer(opts: {
  file: File;
  sourceIndices: number[];
  fps?: number | null;
}): Promise<{
  frames: Array<ServerFrame & { sourceIndex: number }>;
}> {
  const uniq = Array.from(
    new Set(opts.sourceIndices.filter((i) => Number.isInteger(i) && i >= 0)),
  ).sort((a, b) => a - b);
  if (uniq.length === 0) return { frames: [] };

  const form = new FormData();
  form.append("file", opts.file);
  form.append("source_indices", JSON.stringify(uniq));
  if (opts.fps != null && opts.fps > 0) {
    form.append("fps", String(opts.fps));
  }

  const res = await fetch(
    `${pythonApiBaseUrl()}/api/frames/by_source_index`,
    { method: "POST", body: form },
  );
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Python by_source_index extraction failed: ${res.status} ${text}`,
    );
  }
  const body = (await res.json()) as {
    frames: Array<{
      source_index: number;
      mime: string;
      data_base64: string;
    }>;
  };
  const frames = body.frames.map((f) => {
    const blob = base64ToUint8Array(f.data_base64);
    const { width, height } = jpegDimensions(blob);
    return { sourceIndex: f.source_index, blob, width, height };
  });
  return { frames };
}

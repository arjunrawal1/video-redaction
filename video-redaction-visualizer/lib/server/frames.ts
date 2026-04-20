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
}): Promise<FetchFramesResult> {
  const form = new FormData();
  form.append("file", opts.file);
  const qs = new URLSearchParams();
  if (opts.fps != null && opts.fps > 0) qs.set("fps", String(opts.fps));
  if (opts.dedupThreshold != null) {
    qs.set("dedup_threshold", String(opts.dedupThreshold));
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
  };
}

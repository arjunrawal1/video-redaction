const DEFAULT_API_BASE = "http://localhost:8000";

export function getFramesApiBase(): string {
  return (
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
    DEFAULT_API_BASE
  );
}

export type DeduplicatedFrame = {
  mime: string;
  data_base64: string;
};

export type DeduplicatedFramesResponse = {
  filename: string | null;
  fps: number | null;
  dedup_threshold: number;
  raw_frame_count: number;
  deduplicated_count: number;
  frames: DeduplicatedFrame[];
};

function parseErrorBody(text: string): string {
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail
        .map((d) =>
          typeof d === "object" && d && "msg" in d
            ? String((d as { msg: string }).msg)
            : JSON.stringify(d),
        )
        .join("; ");
    }
  } catch {
    /* ignore */
  }
  return text || "Request failed";
}

export async function fetchDeduplicatedFrames(
  file: File,
  signal?: AbortSignal,
): Promise<DeduplicatedFramesResponse> {
  const base = getFramesApiBase();
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${base}/api/frames/deduplicated`, {
    method: "POST",
    body: form,
    signal,
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(parseErrorBody(text) || `HTTP ${res.status}`);
  }
  return JSON.parse(text) as DeduplicatedFramesResponse;
}

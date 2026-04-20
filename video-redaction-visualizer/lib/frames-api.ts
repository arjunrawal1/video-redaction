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
  video_hash?: string;
  fps: number | null;
  dedup_threshold: number;
  raw_frame_count: number;
  deduplicated_count: number;
  frames: DeduplicatedFrame[];
};

export type DetectionBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  text: string;
  score: number;
  // Set by the second/third passes; first-pass detect hits have this unset.
  origin?: "backtrack" | "forward";
  // Excel-style cross-frame identity label. Populated by engines that
  // produce labels server-side (Gemini). OCR leaves this undefined and the
  // client falls back to the deterministic assignLabels pass.
  label?: string;
};

export type DetectionFrame = {
  width: number;
  height: number;
  boxes: DetectionBox[];
};

export type DetectStartEvent = {
  type: "start";
  video_hash: string;
  query: string;
  deduplicated_count: number;
  frame_from: number;
  frame_to: number;
  frame_indices: number[];
  total: number;
};

export type DetectFrameEvent = {
  type: "frame";
  index: number;
  width: number;
  height: number;
  boxes: DetectionBox[];
  // Full Textract DetectDocumentText response for this frame, included
  // verbatim so the UI can surface it via a "Copy OCR debug" button.
  raw?: unknown;
};

export type DetectDoneEvent = {
  type: "done";
  matched_frames: number;
  total_boxes: number;
};

export type DetectErrorEvent = {
  type: "error";
  message: string;
};

export type DetectEvent =
  | DetectStartEvent
  | DetectFrameEvent
  | DetectDoneEvent
  | DetectErrorEvent;

export type StreamDetectOptions = {
  frameFrom?: number;
  frameTo?: number;
  fps?: number;
  dedupThreshold?: number;
  // "ocr"   → Python backend /api/ocr/*    (default)
  // "gemini"→ Next.js route   /api/gemini/* (same-origin, Vercel AI SDK)
  engine?: "ocr" | "gemini";
};

export type BacktrackStartEvent = {
  type: "start";
  video_hash: string;
  query: string;
  frame_from: number;
  frame_to: number;
  total_frames: number;
};

export type BacktrackFrameEvent = {
  type: "frame";
  index: number;
  width: number;
  height: number;
  box: DetectionBox;
  origin: "backtrack";
};

export type ForwardStartEvent = BacktrackStartEvent;

export type ForwardFrameEvent = {
  type: "frame";
  index: number;
  width: number;
  height: number;
  box: DetectionBox;
  origin: "forward";
};

export type ForwardDoneEvent = BacktrackDoneEvent;
export type ForwardErrorEvent = BacktrackErrorEvent;

export type ForwardEvent =
  | ForwardStartEvent
  | ForwardFrameEvent
  | ForwardDoneEvent
  | ForwardErrorEvent;

export type BacktrackDoneEvent = {
  type: "done";
  added_frames: number;
  added_boxes: number;
};

export type BacktrackErrorEvent = {
  type: "error";
  message: string;
};

export type BacktrackEvent =
  | BacktrackStartEvent
  | BacktrackFrameEvent
  | BacktrackDoneEvent
  | BacktrackErrorEvent;

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

async function streamNdjson<E>(
  url: string,
  form: FormData,
  onEvent: (event: E) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(url, { method: "POST", body: form, signal });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(parseErrorBody(text) || `HTTP ${res.status}`);
  }
  if (!res.body) {
    throw new Error("Streaming not supported by this browser.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  const flushLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let parsed: E;
    try {
      parsed = JSON.parse(trimmed) as E;
    } catch {
      return;
    }
    onEvent(parsed);
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl = buf.indexOf("\n");
      while (nl !== -1) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        flushLine(line);
        nl = buf.indexOf("\n");
      }
    }
    buf += decoder.decode();
    if (buf.length > 0) flushLine(buf);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}

function buildDetectForm(
  file: File,
  query: string,
  opts: StreamDetectOptions,
): FormData {
  const form = new FormData();
  form.append("file", file);
  form.append("query", query);
  if (opts.frameFrom != null) form.append("frame_from", String(opts.frameFrom));
  if (opts.frameTo != null) form.append("frame_to", String(opts.frameTo));
  if (opts.fps != null) form.append("fps", String(opts.fps));
  if (opts.dedupThreshold != null) {
    form.append("dedup_threshold", String(opts.dedupThreshold));
  }
  return form;
}

type PassName = "detect/stream" | "backtrack" | "forward";

function passUrl(pass: PassName, engine: "ocr" | "gemini" | undefined): string {
  if (engine === "gemini") {
    // Same-origin Next.js route handler.
    return `/api/gemini/${pass}`;
  }
  return `${getFramesApiBase()}/api/ocr/${pass}`;
}

export async function streamDetect(
  file: File,
  query: string,
  opts: StreamDetectOptions,
  onEvent: (event: DetectEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const form = buildDetectForm(file, query, opts);
  return streamNdjson<DetectEvent>(
    passUrl("detect/stream", opts.engine),
    form,
    onEvent,
    signal,
  );
}

export async function streamBacktrack(
  file: File,
  query: string,
  opts: StreamDetectOptions,
  onEvent: (event: BacktrackEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const form = buildDetectForm(file, query, opts);
  return streamNdjson<BacktrackEvent>(
    passUrl("backtrack", opts.engine),
    form,
    onEvent,
    signal,
  );
}

export async function streamForward(
  file: File,
  query: string,
  opts: StreamDetectOptions,
  onEvent: (event: ForwardEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const form = buildDetectForm(file, query, opts);
  return streamNdjson<ForwardEvent>(
    passUrl("forward", opts.engine),
    form,
    onEvent,
    signal,
  );
}

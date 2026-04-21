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
  // Reference pixel dimensions shared by every kept frame. Detection
  // boxes live in this space; the exporter uses it to scale boxes up to
  // the source video's native resolution.
  frame_width: number;
  frame_height: number;
  frames: DeduplicatedFrame[];
};

export type DetectionBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  text: string;
  score: number;
  // Set by later passes; first-pass detect hits leave this unset.
  //  - "backtrack" / "forward" come from the chain-extension passes.
  //  - "fix" comes from teamwork's phase-4 final Gemini recheck: boxes
  //    added because Gemini still saw the query after every prior
  //    detection was painted over.
  origin?: "backtrack" | "forward" | "fix";
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
  // Teamwork-only: set when Gemini still saw query text on the
  // OCR-filled frame. UI renders a warning badge.
  flagged?: boolean;
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
  // "ocr"     → Python backend /api/ocr/*      (default)
  // "gemini"  → Next.js route   /api/gemini/*   (same-origin, Vercel AI SDK)
  // "teamwork"→ Next.js route   /api/teamwork/* (OCR + Gemini cooperation)
  engine?: "ocr" | "gemini" | "teamwork";
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

type PassName = "detect/stream" | "backtrack" | "forward" | "navigate";

function passUrl(
  pass: PassName,
  engine: "ocr" | "gemini" | "teamwork" | undefined,
): string {
  if (engine === "gemini") {
    return `/api/gemini/${pass}`;
  }
  if (engine === "teamwork") {
    return `/api/teamwork/${pass}`;
  }
  // OCR doesn't have a navigate phase; callers should only request it
  // for the teamwork engine. Fall back to the OCR base if somehow reached
  // so the 404 is obvious and not a silent wrong-URL.
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

// --- Agentic ("teamwork") navigate pass ----------------------------------
// Streams the tool-calling Gemini agent that revises phase-1 curator
// output by freely moving between frames.

export type NavigateStartEvent = {
  type: "start";
  video_hash: string;
  query: string;
  frame_from: number;
  frame_to: number;
  total_frames: number;
};

export type NavigateToolCallEvent = {
  type: "tool_call";
  step: number;
  name: string;
  input: unknown;
};

export type NavigateToolResultEvent = {
  type: "tool_result";
  step: number;
  name: string;
  summary: string;
};

export type NavigateModelTextEvent = {
  type: "model_text";
  step: number;
  text: string;
};

export type NavigateFrameUpdateEvent = {
  type: "frame_update";
  index: number;
  action: "add" | "remove";
  box: DetectionBox & { hit_id: string };
};

export type NavigateFinishEvent = {
  type: "finish";
  summary: string;
  total_steps: number;
};

export type NavigateDoneEvent = {
  type: "done";
  added_boxes: number;
  removed_boxes: number;
  total_steps: number;
};

export type NavigateErrorEvent = {
  type: "error";
  message: string;
};

export type NavigateEvent =
  | NavigateStartEvent
  | NavigateToolCallEvent
  | NavigateToolResultEvent
  | NavigateModelTextEvent
  | NavigateFrameUpdateEvent
  | NavigateFinishEvent
  | NavigateDoneEvent
  | NavigateErrorEvent;

export async function streamNavigate(
  file: File,
  query: string,
  opts: StreamDetectOptions,
  onEvent: (event: NavigateEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const form = buildDetectForm(file, query, opts);
  return streamNdjson<NavigateEvent>(
    passUrl("navigate", opts.engine ?? "teamwork"),
    form,
    onEvent,
    signal,
  );
}

// --- Full-video export ---------------------------------------------------
// Renders an MP4 with detection boxes painted onto the source video. The
// server reconstructs each kept frame's time window from the same frame
// cache that detection used, so exporting the same (file, fps, dedup)
// combination is cheap.

export type ExportBoxRect = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export type ExportStyle = {
  color?: string;
  padding_px?: number;
};

export type ExportRedactedVideoOptions = {
  fps?: number | null;
  dedupThreshold?: number;
  frameWidth: number;
  frameHeight: number;
  boxesByFrame: Record<number, ExportBoxRect[]>;
  style?: ExportStyle;
  signal?: AbortSignal;
};

export async function exportRedactedVideo(
  file: File,
  opts: ExportRedactedVideoOptions,
): Promise<Blob> {
  const base = getFramesApiBase();
  const payload = {
    fps: opts.fps ?? null,
    // Keep in sync with `_DEFAULT_DEDUP_THRESHOLD` in
    // backend/app/frame_service.py. In practice the caller forwards the
    // value from `framesResult.dedup_threshold`, but we keep a sane
    // fallback so the cache key still hits the server default.
    dedup_threshold: opts.dedupThreshold ?? 2,
    frame_width: opts.frameWidth,
    frame_height: opts.frameHeight,
    style: {
      color: opts.style?.color ?? "black",
      padding_px: opts.style?.padding_px ?? 4,
    },
    boxes_by_frame: opts.boxesByFrame,
  };
  const form = new FormData();
  form.append("file", file);
  form.append("payload", JSON.stringify(payload));

  const res = await fetch(`${base}/api/video/export`, {
    method: "POST",
    body: form,
    signal: opts.signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(parseErrorBody(text) || `HTTP ${res.status}`);
  }
  return res.blob();
}

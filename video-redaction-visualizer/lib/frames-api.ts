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
  /**
   * Upper bound on the run of consecutive raw frames skipped between
   * kept frames. Set to 1 by default so the tween between adjacent kept
   * frames never spans more than ~2 raw-frame steps. Part of the cache
   * key for both Python and TS caches — callers forward it verbatim
   * from here to detect / export / backtrack routes.
   */
  max_gap?: number;
  /**
   * Effective frame rate at which ffmpeg emitted the raw sequence. Used
   * by the gap-filler and exporter to resolve `kept_source_indices`
   * back to timestamps in the source video. `null` when the source fps
   * couldn't be probed and no explicit `fps` was passed.
   */
  source_fps?: number | null;
  raw_frame_count: number;
  deduplicated_count: number;
  /**
   * 0-based index into the original ffmpeg-emitted sequence for each
   * kept frame. Parallel to `frames` (index `k` of `kept_source_indices`
   * describes `frames[k]`). The gap-filler bisects these values to
   * request specific raw frames via /api/frames/by_source_index.
   */
  kept_source_indices?: number[];
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
  // Cross-frame identity assigned by the teamwork phase-1.5 linker
  // (lib/server/agentic-linker.ts). Boxes that share a track_id refer
  // to the same real-world redaction across adjacent scanned frames,
  // so the UI can safely interpolate (tween) their coordinates for
  // smooth rendering between keyframes. Undefined on OCR / Gemini
  // engines and on teamwork boxes that arrived before the linker
  // event did; the existing assignLabels fallback still produces
  // letter labels in that case.
  track_id?: string;
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
  /** Absolute path of the per-run NDJSON log file (empty if disabled). */
  run_log_path?: string;
};

/**
 * Streamed incrementally after every priced event (OCR pages known up
 * front, each curator call, each linker call, each focused-agent call).
 * The client can render a live "Running: $X.XXXX" counter by listening
 * for these and using `running_usd` as the authoritative cumulative.
 */
export type CostUpdateEvent = {
  type: "cost_update";
  /** Which pipeline stage produced this update. */
  phase: string;
  /** Optional identifiers for the specific call. */
  frame_index?: number;
  frame_pair?: [number, number];
  /** USD spent by this single call (delta). */
  call_usd: number;
  /** Cumulative USD across the entire route invocation so far. */
  running_usd: number;
  /** Coarse breakdown of where the running total came from. */
  breakdown: Record<string, number | string>;
};

/**
 * Emitted once at the end of a detect / navigate invocation with the
 * authoritative final total. Clients should replace their running
 * estimate with this number when it arrives.
 */
export type CostFinalEvent = {
  type: "cost_final";
  /** "detect" | "navigate" — which route produced this summary. */
  phase: string;
  total_usd: number;
  breakdown: Record<string, number | string>;
  elapsed_ms?: number;
  /** Absolute path to the per-run NDJSON log for this route invocation. */
  run_log_path?: string;
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

/**
 * Teamwork phase-1.5: per-frame linker output, emitted after every
 * curator `frame` event in the stream has landed. One event per
 * scanned frame (including frames with zero boxes, which emit an empty
 * `track_ids`). The array is aligned with the `boxes` array from the
 * earlier `frame` event for the same `index` — position i in
 * `track_ids` is the identity of `boxes[i]`.
 *
 * `links` is a debug-friendly parallel array whose i'th entry says
 * which previous-frame box index (or null for a newly-appearing track)
 * this box inherited its identity from. Safe to ignore on the client
 * unless you're rendering a link-decisions debug panel.
 */
export type DetectLinkEvent = {
  type: "link";
  /** 1-indexed frame number in the original deduplicated sequence. */
  index: number;
  /**
   * Identity assigned to each box in the frame, aligned with the order
   * emitted in the prior `frame` event. Minted server-side as `t{n}`.
   * Empty when the frame has no boxes.
   */
  track_ids: string[];
  /**
   * Per-box decision narrative. `prev_b_index` identifies which box in
   * the immediately-preceding scanned frame this track was inherited
   * from; null means "new track that first appears here".
   */
  links: Array<{
    prev_index: number | null;
    prev_b_index: number | null;
    reason: string | null;
    /** True if this frame's decision used the deterministic fallback. */
    fallback?: boolean;
  }>;
};

/**
 * Post-linker gap-filler insertion. Emitted after the normal
 * curator/linker stream has landed but before ``cost_final``/``done``,
 * one event per frame that the gap-filler recursively added between
 * two adjacent kept frames. The frame sits at source index ``source_index``
 * (between the two kept frames' source indices) and carries its own
 * boxes + track_ids, already linked into the outer track graph — no
 * separate ``frame`` / ``link`` events follow.
 *
 * Indices are minted as fresh integers past the original kept count
 * (``kept_count + 1``, ``+2``, …). Clients should splice the new
 * frame into their kept-frame array using the ``between`` pair, not
 * the raw ``virtual_index`` ordering; two inserts may share the same
 * ``between`` pair at different recursion depths.
 */
export type DetectInsertedFrameEvent = {
  type: "inserted_frame";
  /** The two kept-frame indices this insert sits between, in the
   *  original dedup sequence. */
  between: [number, number];
  /** Recursion depth (0 at the first bisection of an original gap). */
  depth: number;
  /** Human-readable trigger that caused the bisection; useful for
   *  post-hoc threshold analysis. */
  trigger_reason: string;
  /** 0-based index into the original ffmpeg-emitted sequence. */
  source_index: number;
  frame: {
    width: number;
    height: number;
    /** Base64-encoded JPEG of the inserted frame. Decode client-side
     *  and render into the same frame list as the original kept
     *  frames, positioned by ``between`` + ``source_index``. */
    jpeg_base64: string;
    boxes: DetectionBox[];
    track_ids: string[];
  };
};

export type DetectEvent =
  | DetectStartEvent
  | DetectFrameEvent
  | DetectLinkEvent
  | DetectInsertedFrameEvent
  | DetectDoneEvent
  | DetectErrorEvent
  | CostUpdateEvent
  | CostFinalEvent;

export type StreamDetectOptions = {
  frameFrom?: number;
  frameTo?: number;
  fps?: number;
  dedupThreshold?: number;
  /**
   * Upper bound on consecutive raw frames skipped between kept frames.
   * Plumbed to the Python backend so the kept-frame sequence (and
   * therefore all dedup-keyed caches) match the value used at extract
   * time. Defaults to 1 on the backend.
   */
  maxGap?: number;
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
  if (opts.maxGap != null) {
    form.append("max_gap", String(opts.maxGap));
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
  run_log_path?: string;
};

/** Cascade-only: fires when a focused agent is about to start. */
export type NavigateAgentStartEvent = {
  type: "agent_start";
  agent_id: string;
  focus_frame: number;
  source: "transition" | "cascade";
  parent_agent_id: string | null;
  reason: string;
};

/** Cascade-only: fires when a focused agent finishes. Carries the USD
 *  cost for the single generateText call that ran this agent. */
export type NavigateAgentEndEvent = {
  type: "agent_end";
  agent_id: string;
  focus_frame: number;
  added: number;
  removed: number;
  total_steps: number;
  finish_summary: string | null;
  cost_usd?: number;
};

export type NavigateToolCallEvent = {
  type: "tool_call";
  step: number;
  name: string;
  input: unknown;
  agent_id?: string | null;
};

export type NavigateToolResultEvent = {
  type: "tool_result";
  step: number;
  name: string;
  summary: string;
  agent_id?: string | null;
};

export type NavigateModelTextEvent = {
  type: "model_text";
  step: number;
  text: string;
  agent_id?: string | null;
};

export type NavigateFrameUpdateEvent = {
  type: "frame_update";
  index: number;
  action: "add" | "remove";
  box: DetectionBox & { hit_id: string };
  reason?: string | null;
  agent_id?: string | null;
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
  | NavigateAgentStartEvent
  | NavigateAgentEndEvent
  | NavigateToolCallEvent
  | NavigateToolResultEvent
  | NavigateModelTextEvent
  | NavigateFrameUpdateEvent
  | NavigateFinishEvent
  | NavigateDoneEvent
  | NavigateErrorEvent
  | CostUpdateEvent
  | CostFinalEvent;

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
  /**
   * Cross-frame identity from the teamwork phase-1.5 linker. When two
   * boxes in consecutive kept frames share the same `track_id`, the
   * exporter interpolates the drawbox coords linearly across the
   * time window between them so the redaction tweens smoothly instead
   * of jumping at each keyframe. Undefined → the exporter treats the
   * box as static across its keyframe window (legacy behavior).
   */
  track_id?: string;
};

export type ExportStyle = {
  color?: string;
  padding_px?: number;
};

export type ExportRedactedVideoOptions = {
  fps?: number | null;
  dedupThreshold?: number;
  /**
   * Max-gap used at detect time. Part of the backend frame-cache key;
   * must match the value from `framesResult.max_gap` or the server will
   * re-extract with a different kept-frame set and box coordinates
   * won't line up with the kept-frame sequence the client painted on.
   */
  maxGap?: number;
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
    // Keep in sync with `_DEFAULT_MAX_GAP` in backend/app/frame_service.py.
    max_gap: opts.maxGap ?? 1,
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

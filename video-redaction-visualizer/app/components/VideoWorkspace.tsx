"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import {
  exportRedactedVideo,
  fetchDeduplicatedFrames,
  getFramesApiBase,
  streamBacktrack,
  streamDetect,
  streamForward,
  streamNavigate,
  type DeduplicatedFramesResponse,
  type DetectionFrame,
  type ExportBoxRect,
} from "@/lib/frames-api";
import { assignLabels, type FrameLabelMap } from "@/lib/labeling";

type DetectionEntry = {
  detection: DetectionFrame;
  // Color for first-pass boxes on this frame. Baked when the phase-1 event
  // arrives so later picker changes don't retroactively recolor old runs.
  color: string;
  // Color for backtrack boxes on this frame. Baked when the phase-2 event
  // arrives. Same philosophy as `color`.
  backtrackColor?: string;
  // Color for forward-pass boxes. Baked when the phase-3 event arrives.
  forwardColor?: string;
  raw?: unknown;
  // Teamwork phase-1 flag: Gemini still saw query text on the
  // OCR-filled frame. Rendered as a small warning badge on the thumbnail.
  flagged?: boolean;
};
type DetectionMap = Record<number, DetectionEntry>;

const DEFAULT_FIRST_PASS_COLOR = "#dc2626";
const DEFAULT_BACKWARD_PASS_COLOR = "#f59e0b";
const DEFAULT_FORWARD_PASS_COLOR = "#3b82f6";
// Fallback if somehow no baked color is present; matches the globals.css var.
const BACKTRACK_COLOR = "var(--backtrack)";
import {
  clearUploadedVideo,
  loadUploadedVideo,
  saveUploadedVideo,
} from "@/lib/video-storage";

type VideoClip = {
  file: File;
  url: string;
};

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatToolCall(name: string, input: unknown): string {
  if (!input || typeof input !== "object") return `${name}(?)`;
  const i = input as Record<string, unknown>;
  if (name === "get_frame") return `get_frame(#${i.frame_index})`;
  if (name === "add_box") {
    return `add_box(#${i.frame_index}, "${String(i.text ?? "").slice(0, 24)}")`;
  }
  if (name === "remove_box") {
    return `remove_box(#${i.frame_index}, ${String(i.hit_id ?? "?")})`;
  }
  if (name === "adopt_ocr_box") {
    return `adopt_ocr_box(#${i.frame_index}, ocr:${i.ocr_index})`;
  }
  if (name === "finish") return `finish()`;
  return `${name}(${JSON.stringify(input).slice(0, 80)})`;
}

function DetectionOverlay({
  entry,
  labels,
  className,
}: {
  entry: DetectionEntry | undefined;
  labels?: string[];
  className?: string;
}) {
  if (
    !entry ||
    !entry.detection.width ||
    !entry.detection.height ||
    entry.detection.boxes.length === 0
  ) {
    return null;
  }
  const { width, height, boxes } = entry.detection;
  return (
    <div
      aria-hidden="true"
      className={["pointer-events-none absolute inset-0", className ?? ""].join(" ")}
    >
      {boxes.map((b, i) => {
        const color =
          b.origin === "backtrack"
            ? entry.backtrackColor ?? BACKTRACK_COLOR
            : b.origin === "forward"
              ? entry.forwardColor ?? BACKTRACK_COLOR
              : b.origin === "fix"
                ? // Teamwork phase-4 fix boxes reuse the backward color so
                // they visually stand apart from first-pass hits without
                // needing another picker. See onDetect verify step.
                entry.backtrackColor ?? BACKTRACK_COLOR
                : entry.color;
        const boxStyle = {
          left: `${(b.x / width) * 100}%`,
          top: `${(b.y / height) * 100}%`,
          width: `${(b.w / width) * 100}%`,
          height: `${(b.h / height) * 100}%`,
          borderColor: color,
          boxShadow: `0 0 0 1px rgba(0,0,0,0.4)`,
        } as const;
        // Prefer a per-box label coming from the server (Gemini path).
        // Fall back to the client-side assignLabels result for OCR.
        const label = b.label ?? labels?.[i];
        return (
          <div key={`b-${i}`}>
            <div
              className="absolute rounded-[2px] border-2"
              style={boxStyle}
            />
            {label && (
              <span
                className="absolute -translate-y-full rounded px-1 py-px font-mono text-[10px] font-semibold leading-none text-white shadow-sm"
                style={{
                  left: `${(b.x / width) * 100}%`,
                  top: `${(b.y / height) * 100}%`,
                  backgroundColor: color,
                }}
              >
                {label}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function FrameLightbox({
  frames,
  detections,
  frameLabels,
  index,
  onClose,
  onNavigate,
  onCopyOcr,
  copiedFrameNumber,
}: {
  frames: DeduplicatedFramesResponse["frames"];
  detections: DetectionMap;
  frameLabels: FrameLabelMap;
  index: number;
  onClose: () => void;
  onNavigate: (next: number) => void;
  onCopyOcr: (frameNumber: number) => void;
  copiedFrameNumber: number | null;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight")
        onNavigate(Math.min(frames.length - 1, index + 1));
      else if (e.key === "ArrowLeft") onNavigate(Math.max(0, index - 1));
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [frames.length, index, onClose, onNavigate]);

  const frame = frames[index];
  if (!frame) return null;

  const hasPrev = index > 0;
  const hasNext = index < frames.length - 1;
  const frameNumber = index + 1;
  const entry = detections[frameNumber];
  const hasRaw = entry?.raw !== undefined;
  const isCopied = copiedFrameNumber === frameNumber;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Frame ${index + 1} of ${frames.length}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div className="absolute right-4 top-4 flex items-center gap-2">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onCopyOcr(frameNumber);
          }}
          disabled={!hasRaw}
          title={
            hasRaw
              ? "Copy full Textract response for this frame"
              : "Run Detect on this frame to enable"
          }
          className="rounded-full bg-white/10 px-3 py-1.5 text-sm text-white transition-colors hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isCopied ? "Copied" : "Copy OCR debug"}
        </button>
        <button
          type="button"
          aria-label="Close"
          onClick={onClose}
          className="rounded-full bg-white/10 px-3 py-1.5 text-sm text-white transition-colors hover:bg-white/20"
        >
          Close
        </button>
      </div>

      <p
        className="absolute left-1/2 top-4 -translate-x-1/2 rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-white"
        aria-live="polite"
      >
        {index + 1} / {frames.length}
      </p>

      {hasPrev && (
        <button
          type="button"
          aria-label="Previous frame"
          onClick={(e) => {
            e.stopPropagation();
            onNavigate(index - 1);
          }}
          className="absolute left-4 top-1/2 -translate-y-1/2 rounded-full bg-white/10 p-3 text-white transition-colors hover:bg-white/20"
        >
          <span aria-hidden="true">‹</span>
        </button>
      )}
      {hasNext && (
        <button
          type="button"
          aria-label="Next frame"
          onClick={(e) => {
            e.stopPropagation();
            onNavigate(index + 1);
          }}
          className="absolute right-4 top-1/2 -translate-y-1/2 rounded-full bg-white/10 p-3 text-white transition-colors hover:bg-white/20"
        >
          <span aria-hidden="true">›</span>
        </button>
      )}

      <div
        className="relative inline-block max-h-[90vh] max-w-[95vw]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- data URLs from API */}
        <img
          src={`data:${frame.mime};base64,${frame.data_base64}`}
          alt={`Frame ${index + 1}`}
          className="block max-h-[90vh] max-w-[95vw]"
        />
        <DetectionOverlay
          entry={detections[index + 1]}
          labels={frameLabels[index + 1]}
        />
      </div>
    </div>
  );
}

function DeduplicatedFramesPanel({ file }: { file: File }) {
  const queryInputId = useId();
  const fromInputId = useId();
  const toInputId = useId();
  const firstColorInputId = useId();
  const backwardColorInputId = useId();
  const forwardColorInputId = useId();

  const [framesResult, setFramesResult] = useState<DeduplicatedFramesResponse | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const [query, setQuery] = useState("");
  const [frameFrom, setFrameFrom] = useState<number>(1);
  const [frameTo, setFrameTo] = useState<number>(1);
  // "ocr"      → Python backend (Textract).
  // "gemini"   → Next.js routes (OpenRouter + Vercel AI SDK).
  // "teamwork" → Agentic: parallel per-frame curator + tool-calling
  //              navigator. Gemini 3 Pro is the ultimate decision maker.
  const [engine, setEngine] = useState<"ocr" | "gemini" | "teamwork">("ocr");
  const [firstPassColor, setFirstPassColor] = useState<string>(
    DEFAULT_FIRST_PASS_COLOR,
  );
  const [backwardPassColor, setBackwardPassColor] = useState<string>(
    DEFAULT_BACKWARD_PASS_COLOR,
  );
  // Forward-pass color is UI-only for now; forward pass itself is a later
  // feature. Captured here so the picker can be wired without a refactor.
  const [forwardPassColor, setForwardPassColor] = useState<string>(
    DEFAULT_FORWARD_PASS_COLOR,
  );

  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);
  // Sparse map: 1-based frame index -> {detection, color}. Only populated
  // for frames that have been OCR'd at least once.
  const [detections, setDetections] = useState<DetectionMap>({});
  const [lastQuery, setLastQuery] = useState<string>("");
  const [runProgress, setRunProgress] = useState<{
    processed: number;
    total: number;
    from: number;
    to: number;
  } | null>(null);

  const [backtracking, setBacktracking] = useState(false);
  const [backtrackStats, setBacktrackStats] = useState<{
    addedFrames: number;
    addedBoxes: number;
  } | null>(null);
  // Ordered list of frame numbers that phase-2 actually added a box to in
  // the most recent run. Rendered under the input so users can see which
  // specific frames were touched by the backtrack.
  const [backtrackTouched, setBacktrackTouched] = useState<number[]>([]);

  const [forwarding, setForwarding] = useState(false);
  const [forwardStats, setForwardStats] = useState<{
    addedFrames: number;
    addedBoxes: number;
  } | null>(null);
  const [forwardTouched, setForwardTouched] = useState<number[]>([]);

  // Agentic navigator state (teamwork engine only). Replaces the old
  // backtrack / forward / verify chain with a single tool-using Gemini
  // session that decides what to fix and moves between frames freely.
  const [navigating, setNavigating] = useState(false);
  const [navStats, setNavStats] = useState<{
    steps: number;
    added: number;
    removed: number;
    finishSummary: string | null;
  } | null>(null);
  const [navTouched, setNavTouched] = useState<number[]>([]);
  // Rolling log of tool calls + model text for live UI feedback.
  const [navEvents, setNavEvents] = useState<
    Array<{ id: number; step: number; label: string }>
  >([]);

  // Full-video export. The blob URL stays alive as the `href` of the
  // download link; we revoke it on replace/unmount to avoid leaking.
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [exportFilename, setExportFilename] = useState<string | null>(null);

  const detectAbortRef = useRef<AbortController | null>(null);
  const backtrackAbortRef = useRef<AbortController | null>(null);
  const forwardAbortRef = useRef<AbortController | null>(null);
  const navigateAbortRef = useRef<AbortController | null>(null);
  const exportAbortRef = useRef<AbortController | null>(null);

  const totalFrames = framesResult?.deduplicated_count ?? 0;

  useEffect(() => {
    const ac = new AbortController();
    let cancelled = false;

    (async () => {
      try {
        const data = await fetchDeduplicatedFrames(file, ac.signal);
        if (!cancelled) setFramesResult(data);
      } catch (e) {
        if (cancelled || ac.signal.aborted) return;
        const msg =
          e instanceof Error ? e.message : "Could not load deduplicated frames.";
        setError(msg);
      } finally {
        if (!cancelled && !ac.signal.aborted) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [file]);

  useEffect(() => {
    setDetections({});
    setLastQuery("");
    setDetectError(null);
    setQuery("");
    setRunProgress(null);
    setFirstPassColor(DEFAULT_FIRST_PASS_COLOR);
    setBackwardPassColor(DEFAULT_BACKWARD_PASS_COLOR);
    setForwardPassColor(DEFAULT_FORWARD_PASS_COLOR);
    setBacktracking(false);
    setBacktrackStats(null);
    setBacktrackTouched([]);
    setForwarding(false);
    setForwardStats(null);
    setForwardTouched([]);
    setNavigating(false);
    setNavStats(null);
    setNavTouched([]);
    setNavEvents([]);
    setExporting(false);
    setExportError(null);
    setExportUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setExportFilename(null);
    exportAbortRef.current?.abort();
    detectAbortRef.current?.abort();
    backtrackAbortRef.current?.abort();
    forwardAbortRef.current?.abort();
    navigateAbortRef.current?.abort();
  }, [file]);

  // When frames arrive (or count changes), default the range to the full span.
  useEffect(() => {
    if (totalFrames > 0) {
      setFrameFrom(1);
      setFrameTo(totalFrames);
    }
  }, [totalFrames]);

  const frameLabels: FrameLabelMap = useMemo(() => {
    if (totalFrames <= 0) return {};
    const flat: Record<number, { width: number; height: number; boxes: typeof detections[number]["detection"]["boxes"] }> = {};
    for (const [k, v] of Object.entries(detections)) {
      flat[Number(k)] = {
        width: v.detection.width,
        height: v.detection.height,
        boxes: v.detection.boxes,
      };
    }
    return assignLabels(flat, totalFrames);
  }, [detections, totalFrames]);

  const stats = useMemo(() => {
    let matched = 0;
    let primaryBoxes = 0;
    let backtrackBoxes = 0;
    let forwardBoxes = 0;
    let fixBoxes = 0;
    let covered = 0;
    for (const entry of Object.values(detections)) {
      covered += 1;
      if (entry.detection.boxes.length > 0) matched += 1;
      for (const b of entry.detection.boxes) {
        if (b.origin === "backtrack") backtrackBoxes += 1;
        else if (b.origin === "forward") forwardBoxes += 1;
        else if (b.origin === "fix") fixBoxes += 1;
        else primaryBoxes += 1;
      }
    }
    return {
      matched,
      boxes: primaryBoxes + backtrackBoxes + forwardBoxes + fixBoxes,
      primaryBoxes,
      backtrackBoxes,
      forwardBoxes,
      fixBoxes,
      covered,
    };
  }, [detections]);

  const onDetect = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const q = query.trim();
      if (!q || !totalFrames) return;

      const from = Math.max(1, Math.min(totalFrames, Math.floor(frameFrom)));
      const to = Math.max(from, Math.min(totalFrames, Math.floor(frameTo)));

      detectAbortRef.current?.abort();
      backtrackAbortRef.current?.abort();
      forwardAbortRef.current?.abort();
      exportAbortRef.current?.abort();
      const ac = new AbortController();
      detectAbortRef.current = ac;
      setDetecting(true);
      setDetectError(null);
      setLastQuery(q);
      setRunProgress({ processed: 0, total: to - from + 1, from, to });
      setBacktrackStats(null);
      setBacktrackTouched([]);
      setForwardStats(null);
      setForwardTouched([]);
      // A new detection run invalidates any previously exported video.
      setExporting(false);
      setExportError(null);
      setExportUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
      setExportFilename(null);

      // Snapshot the pickers + engine at run start so later changes don't
      // retroactively recolor this run's boxes (consistent with the existing
      // per-run color semantics for the first pass).
      const currentColor = firstPassColor;
      const currentBackwardColor = backwardPassColor;
      const currentForwardColor = forwardPassColor;
      const currentEngine = engine;
      // Clear prior entries inside the range only; preserve the rest.
      setDetections((prev) => {
        const next = { ...prev };
        for (let i = from; i <= to; i++) delete next[i];
        return next;
      });

      // --- Phase 1: detect --------------------------------------------------
      let phase1Ok = false;
      try {
        await streamDetect(
          file,
          q,
          { frameFrom: from, frameTo: to, engine: currentEngine },
          (event) => {
            if (event.type === "start") {
              setRunProgress({
                processed: 0,
                total: event.total,
                from: event.frame_from,
                to: event.frame_to,
              });
            } else if (event.type === "frame") {
              setDetections((prev) => ({
                ...prev,
                [event.index]: {
                  detection: {
                    width: event.width,
                    height: event.height,
                    boxes: event.boxes,
                  },
                  color: currentColor,
                  raw: event.raw,
                  flagged: Boolean(event.flagged),
                },
              }));
              setRunProgress((p) =>
                p ? { ...p, processed: p.processed + 1 } : p,
              );
            } else if (event.type === "error") {
              setDetectError(event.message);
            }
          },
          ac.signal,
        );
        phase1Ok = true;
      } catch (err) {
        if (ac.signal.aborted) return;
        setDetectError(
          err instanceof Error ? err.message : "Detection failed.",
        );
      } finally {
        if (!ac.signal.aborted) setDetecting(false);
      }

      if (!phase1Ok || ac.signal.aborted) return;

      // --- Teamwork branch: agentic navigator replaces backtrack/forward ---
      if (currentEngine === "teamwork") {
        const nac = new AbortController();
        navigateAbortRef.current = nac;
        setNavigating(true);
        let nextEventId = 0;
        try {
          await streamNavigate(
            file,
            q,
            { frameFrom: from, frameTo: to, engine: currentEngine },
            (event) => {
              if (event.type === "frame_update") {
                setDetections((prev) => {
                  const existing = prev[event.index];
                  if (event.action === "add") {
                    const incoming = {
                      ...event.box,
                      origin: event.box.origin ?? ("fix" as const),
                    };
                    if (existing) {
                      return {
                        ...prev,
                        [event.index]: {
                          ...existing,
                          backtrackColor: currentBackwardColor,
                          detection: {
                            ...existing.detection,
                            boxes: existing.detection.boxes.some(
                              (b) =>
                                b.x === incoming.x &&
                                b.y === incoming.y &&
                                b.w === incoming.w &&
                                b.h === incoming.h &&
                                b.origin === incoming.origin,
                            )
                              ? existing.detection.boxes
                              : [...existing.detection.boxes, incoming],
                          },
                        },
                      };
                    }
                    return {
                      ...prev,
                      [event.index]: {
                        detection: {
                          width: 0,
                          height: 0,
                          boxes: [incoming],
                        },
                        color: currentColor,
                        backtrackColor: currentBackwardColor,
                      },
                    };
                  }
                  if (!existing) return prev;
                  return {
                    ...prev,
                    [event.index]: {
                      ...existing,
                      detection: {
                        ...existing.detection,
                        boxes: existing.detection.boxes.filter(
                          (b) =>
                            !(
                              b.x === event.box.x &&
                              b.y === event.box.y &&
                              b.w === event.box.w &&
                              b.h === event.box.h
                            ),
                        ),
                      },
                    },
                  };
                });
                setNavTouched((prev) =>
                  prev.includes(event.index) ? prev : [...prev, event.index],
                );
              } else if (event.type === "tool_call") {
                const label = formatToolCall(event.name, event.input);
                const id = nextEventId++;
                setNavEvents((prev) =>
                  [...prev, { id, step: event.step, label }].slice(-30),
                );
              } else if (event.type === "tool_result") {
                const id = nextEventId++;
                setNavEvents((prev) =>
                  [
                    ...prev,
                    {
                      id,
                      step: event.step,
                      label: `↳ ${event.summary}`,
                    },
                  ].slice(-30),
                );
              } else if (event.type === "model_text") {
                const id = nextEventId++;
                setNavEvents((prev) =>
                  [
                    ...prev,
                    {
                      id,
                      step: event.step,
                      label: `💭 ${event.text.slice(0, 140)}${event.text.length > 140 ? "…" : ""}`,
                    },
                  ].slice(-30),
                );
              } else if (event.type === "finish") {
                setNavStats((prev) => ({
                  steps: event.total_steps,
                  added: prev?.added ?? 0,
                  removed: prev?.removed ?? 0,
                  finishSummary: event.summary,
                }));
              } else if (event.type === "done") {
                setNavStats((prev) => ({
                  steps: event.total_steps,
                  added: event.added_boxes,
                  removed: event.removed_boxes,
                  finishSummary: prev?.finishSummary ?? null,
                }));
              } else if (event.type === "error") {
                setDetectError(event.message);
              }
            },
            nac.signal,
          );
        } catch (err) {
          if (nac.signal.aborted) return;
          setDetectError(
            err instanceof Error ? err.message : "Navigator failed.",
          );
        } finally {
          if (!nac.signal.aborted) setNavigating(false);
        }
        return;
      }

      // --- Phase 2: backtrack (ocr + gemini engines) ------------------------
      const bac = new AbortController();
      backtrackAbortRef.current = bac;
      setBacktracking(true);
      let phase2Ok = false;
      try {
        await streamBacktrack(
          file,
          q,
          { frameFrom: from, frameTo: to, engine: currentEngine },
          (event) => {
            if (event.type === "frame") {
              setDetections((prev) => {
                const existing = prev[event.index];
                const taggedBox = { ...event.box, origin: "backtrack" as const };
                if (existing) {
                  return {
                    ...prev,
                    [event.index]: {
                      ...existing,
                      backtrackColor: currentBackwardColor,
                      detection: {
                        ...existing.detection,
                        // Guard against duplicate appends if the backend
                        // somehow re-emits the same (x,y,w,h).
                        boxes: existing.detection.boxes.some(
                          (b) =>
                            b.x === taggedBox.x &&
                            b.y === taggedBox.y &&
                            b.w === taggedBox.w &&
                            b.h === taggedBox.h &&
                            b.origin === "backtrack",
                        )
                          ? existing.detection.boxes
                          : [...existing.detection.boxes, taggedBox],
                      },
                    },
                  };
                }
                return {
                  ...prev,
                  [event.index]: {
                    detection: {
                      width: event.width,
                      height: event.height,
                      boxes: [taggedBox],
                    },
                    color: currentColor,
                    backtrackColor: currentBackwardColor,
                  },
                };
              });
              setBacktrackTouched((prev) =>
                prev.includes(event.index) ? prev : [...prev, event.index],
              );
            } else if (event.type === "done") {
              setBacktrackStats({
                addedFrames: event.added_frames,
                addedBoxes: event.added_boxes,
              });
            } else if (event.type === "error") {
              setDetectError(event.message);
            }
          },
          bac.signal,
        );
        phase2Ok = true;
      } catch (err) {
        if (bac.signal.aborted) return;
        setDetectError(
          err instanceof Error ? err.message : "Backtrack failed.",
        );
      } finally {
        if (!bac.signal.aborted) setBacktracking(false);
      }

      if (!phase2Ok || bac.signal.aborted) return;

      // --- Phase 3: forward --------------------------------------------------
      const fac = new AbortController();
      forwardAbortRef.current = fac;
      setForwarding(true);
      let phase3Ok = false;
      try {
        await streamForward(
          file,
          q,
          { frameFrom: from, frameTo: to, engine: currentEngine },
          (event) => {
            if (event.type === "frame") {
              setDetections((prev) => {
                const existing = prev[event.index];
                const taggedBox = { ...event.box, origin: "forward" as const };
                if (existing) {
                  return {
                    ...prev,
                    [event.index]: {
                      ...existing,
                      forwardColor: currentForwardColor,
                      detection: {
                        ...existing.detection,
                        boxes: existing.detection.boxes.some(
                          (b) =>
                            b.x === taggedBox.x &&
                            b.y === taggedBox.y &&
                            b.w === taggedBox.w &&
                            b.h === taggedBox.h &&
                            b.origin === "forward",
                        )
                          ? existing.detection.boxes
                          : [...existing.detection.boxes, taggedBox],
                      },
                    },
                  };
                }
                return {
                  ...prev,
                  [event.index]: {
                    detection: {
                      width: event.width,
                      height: event.height,
                      boxes: [taggedBox],
                    },
                    color: currentColor,
                    forwardColor: currentForwardColor,
                  },
                };
              });
              setForwardTouched((prev) =>
                prev.includes(event.index) ? prev : [...prev, event.index],
              );
            } else if (event.type === "done") {
              setForwardStats({
                addedFrames: event.added_frames,
                addedBoxes: event.added_boxes,
              });
            } else if (event.type === "error") {
              setDetectError(event.message);
            }
          },
          fac.signal,
        );
        phase3Ok = true;
      } catch (err) {
        if (fac.signal.aborted) return;
        setDetectError(
          err instanceof Error ? err.message : "Forward pass failed.",
        );
      } finally {
        if (!fac.signal.aborted) setForwarding(false);
      }

      if (!phase3Ok || fac.signal.aborted) return;
    },
    [
      file,
      query,
      frameFrom,
      frameTo,
      firstPassColor,
      backwardPassColor,
      forwardPassColor,
      engine,
      totalFrames,
    ],
  );

  const onCancel = useCallback(() => {
    detectAbortRef.current?.abort();
    backtrackAbortRef.current?.abort();
    forwardAbortRef.current?.abort();
    navigateAbortRef.current?.abort();
    setDetecting(false);
    setBacktracking(false);
    setForwarding(false);
    setNavigating(false);
  }, []);

  const onClearDetection = useCallback(() => {
    detectAbortRef.current?.abort();
    backtrackAbortRef.current?.abort();
    forwardAbortRef.current?.abort();
    navigateAbortRef.current?.abort();
    exportAbortRef.current?.abort();
    setDetections({});
    setLastQuery("");
    setDetectError(null);
    setDetecting(false);
    setBacktracking(false);
    setForwarding(false);
    setNavigating(false);
    setBacktrackStats(null);
    setBacktrackTouched([]);
    setForwardStats(null);
    setForwardTouched([]);
    setNavStats(null);
    setNavTouched([]);
    setNavEvents([]);
    setRunProgress(null);
    setExporting(false);
    setExportError(null);
    setExportUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setExportFilename(null);
  }, []);

  const onExport = useCallback(async () => {
    if (!framesResult) return;
    const refWidth = framesResult.frame_width;
    const refHeight = framesResult.frame_height;
    if (!refWidth || !refHeight) {
      setExportError(
        "Missing reference frame dimensions from the API response.",
      );
      return;
    }

    // Flatten the current detections map into the { frame: boxes } shape
    // the export endpoint expects. Zero-box frames are skipped so the
    // server doesn't emit no-op drawbox instances.
    const boxesByFrame: Record<number, ExportBoxRect[]> = {};
    for (const [k, entry] of Object.entries(detections)) {
      const frameNumber = Number(k);
      if (!Number.isFinite(frameNumber) || entry.detection.boxes.length === 0) {
        continue;
      }
      boxesByFrame[frameNumber] = entry.detection.boxes.map((b) => ({
        x: b.x,
        y: b.y,
        w: b.w,
        h: b.h,
      }));
    }

    if (Object.keys(boxesByFrame).length === 0) {
      setExportError("No detections to export yet. Run Detect first.");
      return;
    }

    exportAbortRef.current?.abort();
    const ac = new AbortController();
    exportAbortRef.current = ac;

    setExportError(null);
    setExporting(true);
    setExportUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setExportFilename(null);

    try {
      const blob = await exportRedactedVideo(file, {
        fps: framesResult.fps,
        dedupThreshold: framesResult.dedup_threshold,
        frameWidth: refWidth,
        frameHeight: refHeight,
        boxesByFrame,
        signal: ac.signal,
      });
      if (ac.signal.aborted) return;
      const url = URL.createObjectURL(blob);
      const base = (file.name || "video").replace(/\.[^.]+$/, "") || "video";
      setExportUrl(url);
      setExportFilename(`${base}.redacted.mp4`);
    } catch (err) {
      if (ac.signal.aborted) return;
      setExportError(err instanceof Error ? err.message : "Export failed.");
    } finally {
      if (!ac.signal.aborted) setExporting(false);
    }
  }, [detections, file, framesResult]);

  useEffect(() => {
    // Revoke any pending blob URL when this panel unmounts (e.g. on clip
    // replacement). The reset effects above handle the replace case; this
    // catches teardown.
    return () => {
      if (exportUrl) URL.revokeObjectURL(exportUrl);
    };
  }, [exportUrl]);

  const selectFullRange = useCallback(() => {
    if (totalFrames > 0) {
      setFrameFrom(1);
      setFrameTo(totalFrames);
    }
  }, [totalFrames]);

  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copyResetRef.current) clearTimeout(copyResetRef.current);
    };
  }, []);

  const copyOcrDebug = useCallback(
    async (frameNumber: number) => {
      const entry = detections[frameNumber];
      if (!entry || entry.raw === undefined) return;
      const payload = {
        frame: frameNumber,
        query: lastQuery,
        width: entry.detection.width,
        height: entry.detection.height,
        matched_boxes: entry.detection.boxes,
        textract_response: entry.raw,
      };
      const text = JSON.stringify(payload, null, 2);
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        // Fallback: legacy execCommand copy via a temp textarea.
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand("copy");
        } finally {
          document.body.removeChild(ta);
        }
      }
      setCopiedIndex(frameNumber);
      if (copyResetRef.current) clearTimeout(copyResetRef.current);
      copyResetRef.current = setTimeout(() => setCopiedIndex(null), 1500);
    },
    [detections, lastQuery],
  );

  return (
    <section
      aria-label="Deduplicated frames"
      className="flex flex-col gap-3 rounded-xl border border-zinc-200 bg-zinc-50/80 p-4 dark:border-zinc-700 dark:bg-zinc-900/50"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold text-foreground">
          Deduplicated frames
        </h2>
        {framesResult && (
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {framesResult.deduplicated_count} kept ·{" "}
            {framesResult.raw_frame_count} sampled
            {framesResult.fps != null ? ` at ${framesResult.fps}/s` : ""}
          </p>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        Source:{" "}
        <span className="font-mono text-zinc-600 dark:text-zinc-400">
          {getFramesApiBase()}
        </span>
      </p>

      <form
        onSubmit={onDetect}
        className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-background/60 p-2"
        aria-label="Redaction text search"
      >
        <label htmlFor={queryInputId} className="sr-only">
          Text to redact
        </label>
        <input
          id={queryInputId}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Text to find and redact (e.g. an email, a name)"
          disabled={detecting || backtracking || forwarding || navigating || loading || !framesResult}
          className="min-w-56 flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-foreground outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-sky-500 disabled:opacity-60"
        />

        <div className="flex items-center gap-1 rounded-md border border-border bg-background px-2 py-1 text-xs text-muted-foreground">
          <label htmlFor={fromInputId} className="sr-only">
            Frame from
          </label>
          <span aria-hidden="true">#</span>
          <input
            id={fromInputId}
            type="number"
            inputMode="numeric"
            min={1}
            max={Math.max(1, totalFrames)}
            value={frameFrom}
            onChange={(e) => {
              const v = Number(e.target.value);
              setFrameFrom(Number.isFinite(v) ? v : 1);
            }}
            disabled={detecting || backtracking || forwarding || navigating || loading || !framesResult}
            className="w-14 bg-transparent text-foreground outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          />
          <span aria-hidden="true" className="text-muted-foreground">
            –
          </span>
          <label htmlFor={toInputId} className="sr-only">
            Frame to
          </label>
          <input
            id={toInputId}
            type="number"
            inputMode="numeric"
            min={1}
            max={Math.max(1, totalFrames)}
            value={frameTo}
            onChange={(e) => {
              const v = Number(e.target.value);
              setFrameTo(Number.isFinite(v) ? v : 1);
            }}
            disabled={detecting || backtracking || forwarding || navigating || loading || !framesResult}
            className="w-14 bg-transparent text-foreground outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          />
          <span className="ml-1 text-[10px] text-muted-foreground">
            of {totalFrames || "—"}
          </span>
          <button
            type="button"
            onClick={selectFullRange}
            disabled={detecting || backtracking || forwarding || navigating || loading || !totalFrames}
            className="ml-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-40"
            title="Set range to all frames"
          >
            all
          </button>
        </div>

        <div
          className="flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1 text-xs text-muted-foreground"
          title="Box colors for each pass"
        >
          <label
            htmlFor={firstColorInputId}
            className="flex items-center gap-1"
            title="First pass (detect)"
          >
            <span>1st</span>
            <input
              id={firstColorInputId}
              type="color"
              value={firstPassColor}
              onChange={(e) => setFirstPassColor(e.target.value)}
              disabled={detecting || backtracking || forwarding || navigating}
              className="h-5 w-6 cursor-pointer appearance-none rounded border border-border bg-transparent p-0 disabled:opacity-50"
            />
          </label>
          <label
            htmlFor={backwardColorInputId}
            className="flex items-center gap-1"
            title="Backward pass (retro-fill partials from earlier frames)"
          >
            <span>Back</span>
            <input
              id={backwardColorInputId}
              type="color"
              value={backwardPassColor}
              onChange={(e) => setBackwardPassColor(e.target.value)}
              disabled={detecting || backtracking || forwarding || navigating}
              className="h-5 w-6 cursor-pointer appearance-none rounded border border-border bg-transparent p-0 disabled:opacity-50"
            />
          </label>
          <label
            htmlFor={forwardColorInputId}
            className="flex items-center gap-1"
            title="Forward pass (retro-fill partials into later frames)"
          >
            <span>Fwd</span>
            <input
              id={forwardColorInputId}
              type="color"
              value={forwardPassColor}
              onChange={(e) => setForwardPassColor(e.target.value)}
              disabled={detecting || backtracking || forwarding || navigating}
              className="h-5 w-6 cursor-pointer appearance-none rounded border border-border bg-transparent p-0 disabled:opacity-50"
            />
          </label>
        </div>

        <div
          role="group"
          aria-label="Detection engine"
          className="inline-flex overflow-hidden rounded-md border border-border text-xs"
        >
          {(["ocr", "gemini", "teamwork"] as const).map((k) => {
            const active = engine === k;
            const label =
              k === "ocr" ? "OCR" : k === "gemini" ? "Gemini" : "Agentic";
            const title =
              k === "ocr"
                ? "Textract OCR via Python backend"
                : k === "gemini"
                  ? "OpenRouter (Gemini 2.5) via Vercel AI SDK"
                  : "Agentic: Gemini 3 Flash (direct) — per-frame curator + bi-directional cascade navigator with code-execution tool for image zoom";
            return (
              <button
                key={k}
                type="button"
                onClick={() => setEngine(k)}
                aria-pressed={active}
                disabled={
                  detecting || backtracking || forwarding || navigating
                }
                title={title}
                className={[
                  "px-2 py-1 font-medium transition-colors disabled:opacity-50",
                  active
                    ? "bg-foreground text-background"
                    : "bg-background text-muted-foreground hover:bg-muted hover:text-foreground",
                ].join(" ")}
              >
                {label}
              </button>
            );
          })}
        </div>

        {detecting || backtracking || forwarding || navigating ? (
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-border bg-transparent px-3 py-1.5 text-sm text-foreground transition-colors hover:bg-muted"
          >
            Cancel
          </button>
        ) : (
          <button
            type="submit"
            disabled={loading || !framesResult || !query.trim()}
            className="rounded-md bg-foreground px-3 py-1.5 text-sm font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            {stats.covered > 0 ? "Re-run range" : "Detect"}
          </button>
        )}

        {stats.covered > 0 && !detecting && !backtracking && !forwarding && !navigating && (
          <button
            type="button"
            onClick={onClearDetection}
            className="rounded-md border border-border bg-transparent px-3 py-1.5 text-sm text-foreground transition-colors hover:bg-muted"
          >
            Clear all
          </button>
        )}
      </form>

      {(stats.covered > 0 || lastQuery) && !detectError && (
        <p className="text-xs text-muted-foreground">
          {lastQuery && (
            <>
              <span className="font-medium text-foreground">
                &ldquo;{lastQuery}&rdquo;
              </span>{" "}
            </>
          )}
          found in {stats.matched}/{stats.covered} scanned frames ·{" "}
          <span style={{ color: firstPassColor }}>
            {stats.primaryBoxes} primary box
            {stats.primaryBoxes === 1 ? "" : "es"}
          </span>
          {stats.backtrackBoxes > 0 && (
            <>
              {" "}
              ·{" "}
              <span style={{ color: backwardPassColor }}>
                {stats.backtrackBoxes} backtracked
              </span>
            </>
          )}
          {stats.forwardBoxes > 0 && (
            <>
              {" "}
              ·{" "}
              <span style={{ color: forwardPassColor }}>
                {stats.forwardBoxes} forwarded
              </span>
            </>
          )}
          {stats.covered < totalFrames && (
            <>
              {" "}
              <span className="text-muted-foreground">
                ({totalFrames - stats.covered} not scanned)
              </span>
            </>
          )}
        </p>
      )}

      {stats.boxes > 0 && (
        <div
          className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-background/60 p-2"
          aria-label="Export redacted video"
        >
          <button
            type="button"
            onClick={onExport}
            disabled={
              exporting ||
              detecting ||
              backtracking ||
              forwarding ||
              navigating
            }
            className="rounded-md bg-foreground px-3 py-1.5 text-sm font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            {exporting
              ? "Rendering…"
              : exportUrl
                ? "Re-export"
                : "Export redacted video"}
          </button>

          {exportUrl && exportFilename && !exporting && (
            <a
              href={exportUrl}
              download={exportFilename}
              className="rounded-md border border-border bg-background px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
            >
              Download {exportFilename}
            </a>
          )}

          <span className="text-[11px] text-muted-foreground">
            Paints {stats.boxes} box{stats.boxes === 1 ? "" : "es"} across{" "}
            {stats.matched} frame{stats.matched === 1 ? "" : "s"} onto the
            source video at native resolution. Audio is copied.
          </span>

          {exportError && (
            <span
              role="status"
              className="w-full rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1 text-xs text-destructive"
            >
              {exportError}
            </span>
          )}
        </div>
      )}

      {detecting && runProgress && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          Streaming OCR · frames {runProgress.from}–{runProgress.to} ·{" "}
          {runProgress.processed}/{runProgress.total} processed
        </p>
      )}

      {backtracking && !detecting && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          Backtracking partials across earlier frames…
        </p>
      )}

      {forwarding && !detecting && !backtracking && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          Forwarding partials into later frames…
        </p>
      )}

      {navigating && !detecting && !backtracking && !forwarding && (
        <div
          className="flex flex-col gap-1 rounded-lg border border-border bg-background/60 p-2"
          aria-live="polite"
        >
          <p className="text-xs text-muted-foreground">
            Navigator working…{" "}
            {navStats?.steps != null && (
              <span className="font-mono text-foreground">
                {navStats.steps} steps
              </span>
            )}{" "}
            <span style={{ color: backwardPassColor }}>
              +{navStats?.added ?? 0}
            </span>
            {" / "}
            <span className="text-muted-foreground">
              −{navStats?.removed ?? 0}
            </span>
          </p>
          {navEvents.length > 0 && (
            <ul className="max-h-40 overflow-y-auto rounded border border-border/60 bg-background px-2 py-1 font-mono text-[10px] text-muted-foreground">
              {navEvents.slice(-12).map((e) => (
                <li key={e.id} className="truncate">
                  <span className="text-foreground">[{e.step}]</span>{" "}
                  {e.label}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {!detecting && !backtracking && !forwarding && !navigating && backtrackStats && (
        <p className="text-[11px] text-muted-foreground">
          <span style={{ color: backwardPassColor }}>Backtrack</span> added{" "}
          {backtrackStats.addedBoxes} box
          {backtrackStats.addedBoxes === 1 ? "" : "es"} across{" "}
          {backtrackStats.addedFrames} frame
          {backtrackStats.addedFrames === 1 ? "" : "s"}
          {backtrackTouched.length > 0 && (
            <>
              {": "}
              {backtrackTouched
                .slice()
                .sort((a, b) => a - b)
                .map((n, i, arr) => (
                  <span key={`bt-${n}`}>
                    <span
                      className="font-mono text-foreground"
                      style={{ color: backwardPassColor }}
                    >
                      #{n}
                    </span>
                    {i < arr.length - 1 ? ", " : ""}
                  </span>
                ))}
            </>
          )}
          .
        </p>
      )}

      {!detecting && !backtracking && !forwarding && !navigating && forwardStats && (
        <p className="text-[11px] text-muted-foreground">
          <span style={{ color: forwardPassColor }}>Forward</span> added{" "}
          {forwardStats.addedBoxes} box
          {forwardStats.addedBoxes === 1 ? "" : "es"} across{" "}
          {forwardStats.addedFrames} frame
          {forwardStats.addedFrames === 1 ? "" : "s"}
          {forwardTouched.length > 0 && (
            <>
              {": "}
              {forwardTouched
                .slice()
                .sort((a, b) => a - b)
                .map((n, i, arr) => (
                  <span key={`fw-${n}`}>
                    <span
                      className="font-mono text-foreground"
                      style={{ color: forwardPassColor }}
                    >
                      #{n}
                    </span>
                    {i < arr.length - 1 ? ", " : ""}
                  </span>
                ))}
            </>
          )}
          .
        </p>
      )}

      {!detecting && !backtracking && !forwarding && !navigating && navStats && (
        <p className="text-[11px] text-muted-foreground">
          <span style={{ color: backwardPassColor }}>Agent</span> ran{" "}
          <span className="font-mono text-foreground">{navStats.steps}</span>{" "}
          step{navStats.steps === 1 ? "" : "s"} ·{" "}
          <span style={{ color: backwardPassColor }}>+{navStats.added}</span>
          {" / "}
          <span>−{navStats.removed}</span>
          {navTouched.length > 0 && (
            <>
              {" · touched: "}
              {navTouched
                .slice()
                .sort((a, b) => a - b)
                .map((n, i, arr) => (
                  <span key={`nav-${n}`}>
                    <span
                      className="font-mono text-foreground"
                      style={{ color: backwardPassColor }}
                    >
                      #{n}
                    </span>
                    {i < arr.length - 1 ? ", " : ""}
                  </span>
                ))}
            </>
          )}
          {navStats.finishSummary && (
            <span className="ml-2 italic">“{navStats.finishSummary}”</span>
          )}
          .
        </p>
      )}

      {detectError && (
        <p
          role="status"
          className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
        >
          {detectError}
        </p>
      )}

      {loading && (
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Extracting and deduplicating frames…
        </p>
      )}

      {error && (
        <p
          role="status"
          className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-900 dark:border-red-900 dark:bg-red-950/50 dark:text-red-100"
        >
          {error}
        </p>
      )}

      {!loading && !error && framesResult && framesResult.frames.length === 0 && (
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          No frames returned after deduplication.
        </p>
      )}

      {framesResult && framesResult.frames.length > 0 && (
        <ul className="grid grid-cols-2 gap-3">
          {framesResult.frames.map((f, i) => {
            const frameNumber = i + 1;
            const entry = detections[frameNumber];
            const labelsForFrame = frameLabels[frameNumber];
            const hasRaw = entry?.raw !== undefined;
            const isCopied = copiedIndex === frameNumber;
            return (
              <li key={`frame-${i}`} className="flex flex-col gap-1">
                <button
                  type="button"
                  onClick={() => setOpenIndex(i)}
                  aria-label={`Open frame ${frameNumber} in full screen`}
                  className="group relative block w-full overflow-hidden rounded-lg border border-zinc-200 bg-zinc-100 transition-colors hover:border-zinc-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 dark:border-zinc-600 dark:bg-zinc-800 dark:hover:border-zinc-400"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element -- data URLs from API */}
                  <img
                    src={`data:${f.mime};base64,${f.data_base64}`}
                    alt={`Frame ${frameNumber}`}
                    className="block w-full"
                    loading="lazy"
                  />
                  <DetectionOverlay entry={entry} labels={labelsForFrame} />
                  <span className="pointer-events-none absolute bottom-1.5 left-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[10px] font-medium text-white">
                    #{frameNumber}
                  </span>
                  {entry?.flagged && (
                    <span
                      className="pointer-events-none absolute left-1.5 top-1.5 rounded bg-amber-500 px-1 py-0.5 text-[10px] font-semibold text-white shadow-sm"
                      title="Gemini still saw the query text on the OCR-redacted frame; may be an OCR miss"
                    >
                      !
                    </span>
                  )}
                  {entry?.detection.boxes.length ? (() => {
                    const primary = entry.detection.boxes.filter(
                      (b) => b.origin == null,
                    ).length;
                    const back = entry.detection.boxes.filter(
                      (b) => b.origin === "backtrack",
                    ).length;
                    // Pick the badge color from whichever origin has any
                    // representation on this frame, preferring primary.
                    const badgeBg =
                      primary > 0
                        ? entry.color
                        : back > 0
                          ? entry.backtrackColor ?? BACKTRACK_COLOR
                          : entry.forwardColor ?? BACKTRACK_COLOR;
                    // Letter list, deduped and in first-encountered order.
                    // Prefer per-box label (Gemini); fall back to the
                    // client-side assignLabels result (OCR).
                    const uniqueLabels: string[] = [];
                    for (let i = 0; i < entry.detection.boxes.length; i++) {
                      const boxLabel =
                        entry.detection.boxes[i].label ??
                        labelsForFrame?.[i];
                      if (boxLabel && !uniqueLabels.includes(boxLabel)) {
                        uniqueLabels.push(boxLabel);
                      }
                    }
                    const letterLabel =
                      uniqueLabels.length > 0
                        ? uniqueLabels.join("\u00b7")
                        : primary > 0 && back > 0
                          ? `${primary}+${back}`
                          : String(primary || back);
                    return (
                      <span
                        className="pointer-events-none absolute bottom-1.5 right-1.5 rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold text-white"
                        style={{ backgroundColor: badgeBg }}
                        title={
                          back > 0
                            ? `${primary} primary + ${back} backtracked`
                            : `${primary} hit${primary === 1 ? "" : "s"}`
                        }
                      >
                        {letterLabel}
                      </span>
                    );
                  })() : null}
                </button>
                <button
                  type="button"
                  onClick={() => copyOcrDebug(frameNumber)}
                  disabled={!hasRaw}
                  title={
                    hasRaw
                      ? "Copy full Textract response for this frame"
                      : "Run Detect on this frame to enable"
                  }
                  className="self-start rounded border border-border bg-background px-2 py-0.5 text-[10px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isCopied ? "Copied" : "Copy OCR debug"}
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {framesResult && openIndex !== null && (
        <FrameLightbox
          frames={framesResult.frames}
          detections={detections}
          frameLabels={frameLabels}
          index={openIndex}
          onClose={() => setOpenIndex(null)}
          onNavigate={setOpenIndex}
          onCopyOcr={copyOcrDebug}
          copiedFrameNumber={copiedIndex}
        />
      )}
    </section>
  );
}

export function VideoWorkspace() {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const clipRef = useRef<VideoClip | null>(null);
  const [clip, setClip] = useState<VideoClip | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const replaceClip = useCallback((next: File | null) => {
    setClip((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      const resolved: VideoClip | null = next
        ? { file: next, url: URL.createObjectURL(next) }
        : null;
      clipRef.current = resolved;
      return resolved;
    });
  }, []);

  useEffect(() => {
    return () => {
      const current = clipRef.current;
      if (current) {
        URL.revokeObjectURL(current.url);
        clipRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stored = await loadUploadedVideo();
        if (!cancelled && stored) replaceClip(stored);
      } catch {
        if (!cancelled) {
          setStorageError("Could not restore video from browser storage.");
        }
      } finally {
        if (!cancelled) setHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [replaceClip]);

  const applyFile = useCallback(
    async (next: File) => {
      setStorageError(null);
      replaceClip(next);
      try {
        await saveUploadedVideo(next);
      } catch {
        setStorageError(
          "Video plays in this session but was not saved to storage.",
        );
      }
    },
    [replaceClip],
  );

  const onInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      e.target.value = "";
      if (!f) return;
      await applyFile(f);
    },
    [applyFile],
  );

  const onDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (!f || !f.type.startsWith("video/")) return;
      await applyFile(f);
    },
    [applyFile],
  );

  const onClear = useCallback(async () => {
    setStorageError(null);
    setIsDragging(false);
    replaceClip(null);
    try {
      await clearUploadedVideo();
    } catch {
      setStorageError("Removed from view; storage may still contain old data.");
    }
  }, [replaceClip]);

  return (
    <div className="mx-auto flex w-full flex-col gap-8">
      <header className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-foreground">
          Video Redaction Visualizer
        </h1>
        <p className="mt-2 text-base text-zinc-600 dark:text-zinc-400">
          Upload a video to preview it locally. Deduplicated frames are built by
          the backend API (every decoded frame, near-duplicates removed).
        </p>
      </header>

      {!clip && (
        <section
          aria-label="Video upload"
          onDragEnter={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) {
              setIsDragging(false);
            }
          }}
          onDrop={onDrop}
          className={[
            "rounded-xl border-2 border-dashed p-8 transition-colors",
            isDragging
              ? "border-sky-500 bg-sky-50 dark:border-sky-400 dark:bg-sky-950/40"
              : "border-zinc-300 bg-zinc-50/80 dark:border-zinc-600 dark:bg-zinc-900/40",
          ].join(" ")}
        >
          <div className="flex flex-col items-center gap-4 text-center">
            <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Drop a video here, or choose a file
            </p>
            <input
              ref={inputRef}
              id={inputId}
              type="file"
              accept="video/*"
              className="sr-only"
              onChange={onInputChange}
            />
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="rounded-lg bg-foreground px-4 py-2.5 text-sm font-medium text-background transition-opacity hover:opacity-90"
            >
              Choose video
            </button>
            <p className="text-xs text-zinc-500 dark:text-zinc-500">
              Preview and IndexedDB storage are local; frames are fetched from
              your API after you load a video below.
            </p>
          </div>
        </section>
      )}

      {storageError && (
        <p
          role="status"
          className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100"
        >
          {storageError}
        </p>
      )}

      {!hydrated && (
        <p className="text-center text-sm text-zinc-500">Loading…</p>
      )}

      {hydrated && clip && (
        <section aria-label="Video preview" className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-foreground">
                {clip.file.name}
              </p>
              <p className="text-xs text-zinc-500 dark:text-zinc-500">
                {formatBytes(clip.file.size)}
              </p>
            </div>
            <button
              type="button"
              onClick={onClear}
              className="shrink-0 rounded-lg border border-zinc-300 bg-transparent px-3 py-1.5 text-sm text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              Remove
            </button>
          </div>
          <div className="overflow-hidden rounded-xl border border-zinc-200 bg-black shadow-sm dark:border-zinc-700">
            <video
              key={clip.url}
              className="max-h-[70vh] w-full object-contain"
              src={clip.url}
              controls
              playsInline
              preload="metadata"
            />
          </div>

          <DeduplicatedFramesPanel key={clip.url} file={clip.file} />
        </section>
      )}

      {hydrated && !clip && (
        <p className="text-center text-sm text-zinc-500 dark:text-zinc-500">
          No video loaded yet.
        </p>
      )}
    </div>
  );
}

"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  fetchDeduplicatedFrames,
  getFramesApiBase,
  type DeduplicatedFramesResponse,
} from "@/lib/frames-api";
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

function FrameLightbox({
  frames,
  index,
  onClose,
  onNavigate,
}: {
  frames: DeduplicatedFramesResponse["frames"];
  index: number;
  onClose: () => void;
  onNavigate: (next: number) => void;
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

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Frame ${index + 1} of ${frames.length}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="absolute right-4 top-4 rounded-full bg-white/10 px-3 py-1.5 text-sm text-white transition-colors hover:bg-white/20"
      >
        Close
      </button>

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

      {/* eslint-disable-next-line @next/next/no-img-element -- data URLs from API */}
      <img
        src={`data:${frame.mime};base64,${frame.data_base64}`}
        alt={`Frame ${index + 1}`}
        className="max-h-full max-w-full object-contain"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

function DeduplicatedFramesPanel({ file }: { file: File }) {
  const [framesResult, setFramesResult] = useState<DeduplicatedFramesResponse | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openIndex, setOpenIndex] = useState<number | null>(null);

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
      <p className="text-xs text-zinc-500 dark:text-zinc-500">
        Source:{" "}
        <span className="font-mono text-zinc-600 dark:text-zinc-400">
          {getFramesApiBase()}
        </span>
      </p>

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
          {framesResult.frames.map((f, i) => (
            <li key={`frame-${i}`}>
              <button
                type="button"
                onClick={() => setOpenIndex(i)}
                aria-label={`Open frame ${i + 1} in full screen`}
                className="group relative block w-full overflow-hidden rounded-lg border border-zinc-200 bg-zinc-100 transition-colors hover:border-zinc-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 dark:border-zinc-600 dark:bg-zinc-800 dark:hover:border-zinc-400"
              >
                {/* eslint-disable-next-line @next/next/no-img-element -- data URLs from API */}
                <img
                  src={`data:${f.mime};base64,${f.data_base64}`}
                  alt={`Frame ${i + 1}`}
                  className="aspect-video w-full object-cover"
                  loading="lazy"
                />
                <span className="pointer-events-none absolute bottom-1.5 left-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[10px] font-medium text-white">
                  #{i + 1}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {framesResult && openIndex !== null && (
        <FrameLightbox
          frames={framesResult.frames}
          index={openIndex}
          onClose={() => setOpenIndex(null)}
          onNavigate={setOpenIndex}
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

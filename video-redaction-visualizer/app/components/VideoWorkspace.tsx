"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
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
    replaceClip(null);
    try {
      await clearUploadedVideo();
    } catch {
      setStorageError("Removed from view; storage may still contain old data.");
    }
  }, [replaceClip]);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8">
      <header className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-foreground">
          Video Redaction Visualizer
        </h1>
        <p className="mt-2 text-base text-zinc-600 dark:text-zinc-400">
          Upload a video to preview it. The file is kept in memory and in
          browser storage for this site.
        </p>
      </header>

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
            Stored locally in your browser (IndexedDB), not sent to a server.
          </p>
        </div>
      </section>

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

// Per-run append-only log file.
//
// Every detect / navigate route invocation opens one of these. The file
// is a newline-delimited JSON stream: one structured event per line,
// capturing every Gemini request / response / tool call plus a per-call
// cost stamp. Meant for post-run inspection ("show me everything the
// third agent did") and for cross-run cost accounting.
//
// Files live under ./logs/ by default (override via AGENTIC_LOG_DIR).
// Filenames encode timestamp, run kind (detect|navigate), and a sanitized
// query slug so you can find a specific run without `ls -t`.
//
// Behavioural contract:
//   - write() is fire-and-forget (returns void). No awaits in hot paths.
//   - close() flushes and awaits the underlying stream's finish event.
//   - open() succeeds even if the logs dir is unwritable (falls back to
//     a no-op logger) so logging never breaks production traffic.

import {
  type WriteStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

export type RunLogEvent = Record<string, unknown>;

export type RunLog = {
  /** Opaque run id (filename stem, also echoed in every event). */
  id: string;
  /** Absolute path to the .jsonl file — surface to the client so the
   *  user can open it. Will be "" on a no-op logger. */
  path: string;
  /**
   * Absolute path to the sibling directory this run writes annotated
   * frames into (`<runId>-frames/`). Lazily created the first time a
   * frame is written. Empty string on a no-op logger or when frame
   * logging is disabled via ``AGENTIC_LOG_FRAMES=off``.
   */
  framesDir: string;
  /** True iff write() does anything. */
  enabled: boolean;
  write: (event: RunLogEvent) => void;
  /**
   * Write a raw binary artifact (typically an annotated JPEG) into the
   * frames directory. ``name`` is the filename; callers should pass
   * the full basename including extension (e.g. ``"frame-042.jpg"``).
   * Fire-and-forget: errors are logged to console but never thrown.
   */
  writeFrame: (name: string, bytes: Uint8Array) => void;
  close: () => Promise<void>;
};

function logsDir(): string {
  return process.env.AGENTIC_LOG_DIR || join(process.cwd(), "logs");
}

function slug(raw: string): string {
  const s = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return s || "noquery";
}

function timestamp(): string {
  // 2026-04-21T15-28-00-123Z — filesystem-safe form of ISO-8601.
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function randomSuffix(): string {
  return Math.random().toString(36).slice(2, 6);
}

/**
 * Open a new run log. `kind` tags the run ("detect" | "navigate"); extra
 * `meta` is written into a `run_start` event at the top of the file.
 */
export function openRunLog(
  kind: string,
  meta: Record<string, unknown> = {},
): RunLog {
  const noop: RunLog = {
    id: "",
    path: "",
    framesDir: "",
    enabled: false,
    write: () => {},
    writeFrame: () => {},
    close: async () => {},
  };

  if (process.env.AGENTIC_RUN_LOG === "off") return noop;

  let dir: string;
  try {
    dir = logsDir();
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  } catch (e) {
    console.warn(
      "[run-log] could not create logs dir, disabling run log:",
      (e as Error).message,
    );
    return noop;
  }

  const querySlug = typeof meta.query === "string" ? slug(meta.query) : "run";
  const id = `${timestamp()}-${kind}-${querySlug}-${randomSuffix()}`;
  const path = join(dir, `${id}.jsonl`);

  let stream: WriteStream;
  try {
    stream = createWriteStream(path, { flags: "w" });
  } catch (e) {
    console.warn(
      "[run-log] could not open log file, disabling run log:",
      (e as Error).message,
    );
    return noop;
  }

  stream.on("error", (e) => {
    console.warn("[run-log] write stream error:", e.message);
  });

  let closed = false;

  const write = (event: RunLogEvent) => {
    if (closed) return;
    const payload = { ts: new Date().toISOString(), run_id: id, ...event };
    try {
      stream.write(JSON.stringify(payload) + "\n");
    } catch (e) {
      console.warn("[run-log] failed to write event:", (e as Error).message);
    }
  };

  write({ kind: "run_start", run_kind: kind, ...meta });

  // Frame-dump support. Disabled at the env level (``AGENTIC_LOG_FRAMES=off``)
  // or when we can't create the directory. The directory is created
  // lazily on the first ``writeFrame`` call so a run that never produces
  // annotated frames leaves zero empty directories behind.
  const framesEnabled = process.env.AGENTIC_LOG_FRAMES !== "off";
  const framesDir = framesEnabled ? join(dir, `${id}-frames`) : "";
  let framesDirReady = false;
  const writeFrame = (name: string, bytes: Uint8Array): void => {
    if (!framesEnabled || !framesDir) return;
    if (!framesDirReady) {
      try {
        if (!existsSync(framesDir)) mkdirSync(framesDir, { recursive: true });
        framesDirReady = true;
      } catch (e) {
        console.warn(
          "[run-log] could not create frames dir, disabling frame dump:",
          (e as Error).message,
        );
        return;
      }
    }
    try {
      writeFileSync(join(framesDir, name), bytes);
    } catch (e) {
      console.warn(
        `[run-log] failed to write frame ${name}:`,
        (e as Error).message,
      );
    }
  };

  const close = (): Promise<void> =>
    new Promise<void>((resolve) => {
      if (closed) return resolve();
      closed = true;
      stream.end(() => resolve());
    });

  return { id, path, framesDir, enabled: true, write, writeFrame, close };
}

/**
 * Convenience: truncate long strings / arrays for inline inspection
 * without destroying the structure.
 */
export function compact(
  value: unknown,
  maxStringLen = 4000,
  maxArrayLen = 64,
  depth = 0,
): unknown {
  if (value == null) return value;
  if (typeof value === "string") {
    return value.length > maxStringLen
      ? value.slice(0, maxStringLen) + `…[+${value.length - maxStringLen} chars]`
      : value;
  }
  if (typeof value !== "object") return value;
  if (depth > 6) return "[truncated-depth]";
  if (Array.isArray(value)) {
    const arr = value.slice(0, maxArrayLen).map((v) =>
      compact(v, maxStringLen, maxArrayLen, depth + 1),
    );
    if (value.length > maxArrayLen) arr.push(`…[+${value.length - maxArrayLen} items]`);
    return arr;
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    out[k] = compact(v, maxStringLen, maxArrayLen, depth + 1);
  }
  return out;
}

// Observability helper for the agentic pipeline.
//
// Nothing here affects behavior. Everything writes to `console.log` /
// `console.error` with a clear `[agentic]` prefix so it interleaves
// visibly with Next.js' own dev-server log stream.
//
// Toggle off by setting AGENTIC_LOG=off (default is on). Image bytes and
// other huge blobs are truncated so a single log line stays readable.

function enabled(): boolean {
  const raw = (process.env.AGENTIC_LOG || "on").trim().toLowerCase();
  return raw !== "off" && raw !== "0" && raw !== "false";
}

function ts(): string {
  return new Date().toISOString().slice(11, 23); // HH:MM:SS.mmm
}

/**
 * Truncate huge strings (mostly base64 images) and trim arrays that get
 * spammy in the log. Preserves structure so json is still readable.
 */
export function sanitize(
  value: unknown,
  maxStringLen = 240,
  maxArrayLen = 24,
  depth = 0,
): unknown {
  if (depth > 6) return "[depth>6]";
  if (value == null) return value;
  if (typeof value === "string") {
    if (value.length <= maxStringLen) return value;
    return `${value.slice(0, maxStringLen)}… [+${value.length - maxStringLen} chars]`;
  }
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (value instanceof Uint8Array || value instanceof ArrayBuffer) {
    const n = value instanceof Uint8Array ? value.byteLength : value.byteLength;
    return `[binary ${n}B]`;
  }
  if (Array.isArray(value)) {
    const out = value
      .slice(0, maxArrayLen)
      .map((v) => sanitize(v, maxStringLen, maxArrayLen, depth + 1));
    if (value.length > maxArrayLen) out.push(`… +${value.length - maxArrayLen} more`);
    return out;
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(obj)) {
      // Specifically squash known-large fields aggressively.
      if (
        k === "image_base64" ||
        k === "data" && typeof obj[k] === "string" && (obj[k] as string).length > 256
      ) {
        out[k] = `[base64 ${(obj[k] as string).length}B]`;
        continue;
      }
      out[k] = sanitize(obj[k], maxStringLen, maxArrayLen, depth + 1);
    }
    return out;
  }
  return String(value);
}

function pretty(obj: unknown): string {
  try {
    return JSON.stringify(sanitize(obj), null, 2);
  } catch {
    return String(obj);
  }
}

export function alog(label: string, payload?: unknown): void {
  if (!enabled()) return;
  if (payload === undefined) {
    console.log(`[agentic] ${ts()} ${label}`);
  } else {
    console.log(`[agentic] ${ts()} ${label} ${pretty(payload)}`);
  }
}

export function aerr(label: string, err: unknown): void {
  if (!enabled()) return;
  const msg = err instanceof Error ? err.message : String(err);
  const stack = err instanceof Error ? err.stack : undefined;
  console.error(`[agentic] ${ts()} ERR ${label} :: ${msg}${stack ? `\n${stack}` : ""}`);
}

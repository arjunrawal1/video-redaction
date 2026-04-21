// Box-shrink post-processing pass.
//
// Every pipeline that emits a redaction box (agentic curator, cascade
// navigator, prompt-mode tools) asks Gemini — in the same model call
// that places the box — to label `text_color_hex` and
// `background_color_hex` for that box. Those two colors let us run a
// deterministic, pixel-based tightening pass after all Gemini work has
// settled: for every side of every box whose TWO corner pixels both
// fall on the background (i.e. non-text) color, pull that side inward
// one pixel at a time until at least one corner on that side reads as
// non-background.
//
// Why "both corners on the same side on background" and not "any
// corner on background":
//   - A legitimate box tight to the text glyphs usually has at least
//     one corner (especially on a thin sans-serif line) falling on the
//     background between two letters. That is fine — the GOAL is to
//     redact the rectangular hull of the text, not every ink stroke.
//   - A box that bleeds into margin on one side, however, will have
//     BOTH of that side's corners on background (there is no text
//     anywhere along that row/column). That is the exact visual
//     signature of a too-wide / too-tall box and is safe to trim.
//
// Corner sampling uses a small NxN window (default 3x3) rather than a
// single pixel to absorb JPEG chroma noise and 1-pixel anti-aliasing
// halos. Classification is simple nearest-neighbor by Euclidean
// distance in RGB space — with only two reference colors and small
// sample sizes, more expensive perceptual metrics (Lab, ΔE) don't
// change outcomes.
//
// Safety:
//   - If either color is missing (deterministic fallback paths), the
//     box passes through unchanged.
//   - If the box is already too small (w <= 2 or h <= 2), no shrinking
//     is attempted.
//   - A generous per-side cap (`BOX_SHRINK_MAX_FRACTION`, default 0.5)
//     prevents runaway shrinkage on frames where the chosen colors
//     happen to misclassify the whole rectangle. Degenerate output is
//     never produced.

import sharp from "sharp";
import type { ServerBox } from "./openrouter";

/** RGB triple, each channel 0..255. */
type RGB = readonly [number, number, number];

export type ShrinkOptions = {
  /**
   * Half-width of the NxN window sampled at each corner. Default 1
   * (3x3). 0 means single-pixel sample. Window is clamped to image
   * bounds.
   */
  sampleRadius?: number;
  /**
   * Max fraction of a side's original length we're willing to trim.
   * Default 0.5 — a runaway misclassification can't eat more than half
   * the box. Set higher (up to 1.0) to allow more aggressive shrinking
   * or lower (e.g. 0.25) to be conservative.
   */
  maxShrinkFraction?: number;
};

export type ShrinkResult = {
  /** Original box object (unchanged reference). */
  box: ServerBox;
  /** Post-shrink geometry. Always a valid rectangle inside the frame. */
  shrunk: { x: number; y: number; w: number; h: number };
  /** True when geometry changed from `box`. */
  changed: boolean;
  /**
   * Why the box was not shrunk further, or an empty string when the
   * algorithm ran to natural completion on a changed box. "no_colors"
   * / "too_small" / "no_violation" / "cap_reached" / "degenerate".
   */
  reason: string;
  /** How many pixels were trimmed from each side. */
  trimmed: { left: number; right: number; top: number; bottom: number };
};

const HEX6 = /^#?([0-9a-f]{6})$/i;
const HEX3 = /^#?([0-9a-f]{3})$/i;

/**
 * Accept `#RRGGBB`, `#RGB`, or the same without the leading `#`.
 * Returns a canonical `#rrggbb` lowercase string, or undefined on
 * anything else (named colors, rgb() literals, whitespace-only,
 * missing). Undefined signals "no color label" to
 * `shrinkBoxesOnFrame`, which will then pass the box through
 * unchanged. Used by every path that attaches Gemini-emitted color
 * strings to a `ServerBox`.
 */
export function normalizeHexColor(
  value: string | null | undefined,
): string | undefined {
  if (!value) return undefined;
  const trimmed = value.trim().toLowerCase();
  const m6 = /^#?([0-9a-f]{6})$/.exec(trimmed);
  if (m6) return `#${m6[1]}`;
  const m3 = /^#?([0-9a-f]{3})$/.exec(trimmed);
  if (m3) {
    const c = m3[1];
    return `#${c[0]}${c[0]}${c[1]}${c[1]}${c[2]}${c[2]}`;
  }
  return undefined;
}

/**
 * Parse `#RRGGBB` or `#RGB` (with or without the leading `#`) into an
 * RGB triple. Returns null on garbage so callers can skip the shrink
 * path rather than crash.
 */
export function hexToRgb(hex: string | undefined | null): RGB | null {
  if (!hex) return null;
  const s = hex.trim();
  const m6 = HEX6.exec(s);
  if (m6) {
    const n = parseInt(m6[1], 16);
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff] as const;
  }
  const m3 = HEX3.exec(s);
  if (m3) {
    const chars = m3[1];
    return [
      parseInt(chars[0] + chars[0], 16),
      parseInt(chars[1] + chars[1], 16),
      parseInt(chars[2] + chars[2], 16),
    ] as const;
  }
  return null;
}

function dist2(a: RGB, b: RGB): number {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return dr * dr + dg * dg + db * db;
}

type RawImage = {
  data: Buffer;
  width: number;
  height: number;
  channels: number;
};

async function decodeToRaw(jpeg: Uint8Array): Promise<RawImage> {
  // Force 3-channel RGB so downstream pixel math doesn't have to
  // special-case alpha. sharp handles JPEG/PNG/WebP uniformly.
  const { data, info } = await sharp(jpeg)
    .removeAlpha()
    .toColorspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    data,
    width: info.width,
    height: info.height,
    channels: info.channels,
  };
}

/**
 * Average RGB over an (2r+1)x(2r+1) window centered at (cx, cy),
 * clipped to image bounds.
 */
function sampleAvg(img: RawImage, cx: number, cy: number, r: number): RGB {
  const { data, width, height, channels } = img;
  const x0 = Math.max(0, Math.min(width - 1, cx - r));
  const x1 = Math.max(0, Math.min(width - 1, cx + r));
  const y0 = Math.max(0, Math.min(height - 1, cy - r));
  const y1 = Math.max(0, Math.min(height - 1, cy + r));
  let sr = 0;
  let sg = 0;
  let sb = 0;
  let n = 0;
  for (let y = y0; y <= y1; y++) {
    const row = y * width * channels;
    for (let x = x0; x <= x1; x++) {
      const i = row + x * channels;
      sr += data[i];
      sg += data[i + 1];
      sb += data[i + 2];
      n += 1;
    }
  }
  if (n === 0) return [0, 0, 0] as const;
  return [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)] as const;
}

type Classification = "text" | "background";

function classifyCorner(sample: RGB, text: RGB, bg: RGB): Classification {
  // Ties (equidistant, rare with integer pixels) resolve to "text" so
  // we err toward NOT shrinking — over-shrinking is a correctness
  // risk, over-keeping is just a few wasted pixels of redaction.
  return dist2(sample, bg) < dist2(sample, text) ? "background" : "text";
}

function parseEnvFraction(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  if (!Number.isFinite(raw)) return fallback;
  return Math.min(1, Math.max(0, raw));
}

function parseEnvInt(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  if (!Number.isFinite(raw) || raw < 0) return fallback;
  return Math.floor(raw);
}

/**
 * Apply the corner-constraint shrink pass to every box on a single
 * frame. Returns one `ShrinkResult` per input box (same order). The
 * frame JPEG is decoded once, shared across all boxes.
 */
export async function shrinkBoxesOnFrame(
  jpeg: Uint8Array,
  boxes: ServerBox[],
  opts: ShrinkOptions = {},
): Promise<ShrinkResult[]> {
  if (boxes.length === 0) return [];

  const sampleRadius =
    opts.sampleRadius ?? parseEnvInt("BOX_SHRINK_SAMPLE_RADIUS", 1);
  const maxFraction =
    opts.maxShrinkFraction ?? parseEnvFraction("BOX_SHRINK_MAX_FRACTION", 0.5);

  let img: RawImage;
  try {
    img = await decodeToRaw(jpeg);
  } catch {
    // Decode failed (corrupt JPEG or sharp missing on platform). The
    // caller gets every box back unchanged — shrinking is a best-
    // effort pass, never critical path.
    return boxes.map((box) => ({
      box,
      shrunk: { x: box.x, y: box.y, w: box.w, h: box.h },
      changed: false,
      reason: "decode_failed",
      trimmed: { left: 0, right: 0, top: 0, bottom: 0 },
    }));
  }

  return boxes.map((box) => shrinkSingleBox(img, box, sampleRadius, maxFraction));
}

function shrinkSingleBox(
  img: RawImage,
  box: ServerBox,
  sampleRadius: number,
  maxFraction: number,
): ShrinkResult {
  const orig = { x: box.x, y: box.y, w: box.w, h: box.h };
  const text = hexToRgb(box.text_color_hex ?? null);
  const bg = hexToRgb(box.background_color_hex ?? null);
  if (!text || !bg) {
    return {
      box,
      shrunk: orig,
      changed: false,
      reason: "no_colors",
      trimmed: { left: 0, right: 0, top: 0, bottom: 0 },
    };
  }
  // Near-identical text & background colors mean the classifier can't
  // distinguish corners — bail rather than shrink based on noise.
  if (dist2(text, bg) < 8 * 8 * 3) {
    return {
      box,
      shrunk: orig,
      changed: false,
      reason: "colors_too_close",
      trimmed: { left: 0, right: 0, top: 0, bottom: 0 },
    };
  }
  if (orig.w <= 2 || orig.h <= 2) {
    return {
      box,
      shrunk: orig,
      changed: false,
      reason: "too_small",
      trimmed: { left: 0, right: 0, top: 0, bottom: 0 },
    };
  }

  let x = orig.x;
  let y = orig.y;
  let w = orig.w;
  let h = orig.h;
  const maxLeft = Math.floor(orig.w * maxFraction);
  const maxRight = Math.floor(orig.w * maxFraction);
  const maxTop = Math.floor(orig.h * maxFraction);
  const maxBottom = Math.floor(orig.h * maxFraction);
  let cutLeft = 0;
  let cutRight = 0;
  let cutTop = 0;
  let cutBottom = 0;

  // Hard iteration bound: even in the degenerate case of a huge box
  // where we trim 1 px per iteration from each side, (w+h) iterations
  // saturate all four caps.
  const maxIters = Math.max(4, orig.w + orig.h);
  let reason = "no_violation";

  for (let iter = 0; iter < maxIters; iter++) {
    if (w <= 2 || h <= 2) {
      reason = "degenerate";
      break;
    }
    const tl = classifyCorner(sampleAvg(img, x, y, sampleRadius), text, bg);
    const tr = classifyCorner(
      sampleAvg(img, x + w - 1, y, sampleRadius),
      text,
      bg,
    );
    const bl = classifyCorner(
      sampleAvg(img, x, y + h - 1, sampleRadius),
      text,
      bg,
    );
    const br = classifyCorner(
      sampleAvg(img, x + w - 1, y + h - 1, sampleRadius),
      text,
      bg,
    );

    // Side-priority: shrink horizontally before vertically, because
    // redaction boxes are almost always set in horizontal lines where
    // left/right over-reach is the dominant failure mode. This
    // ordering also means a "hollow" box (all 4 corners bg) gets
    // chewed inward from all sides roughly symmetrically.
    if (tl === "background" && bl === "background" && cutLeft < maxLeft) {
      x += 1;
      w -= 1;
      cutLeft += 1;
      continue;
    }
    if (tr === "background" && br === "background" && cutRight < maxRight) {
      w -= 1;
      cutRight += 1;
      continue;
    }
    if (tl === "background" && tr === "background" && cutTop < maxTop) {
      y += 1;
      h -= 1;
      cutTop += 1;
      continue;
    }
    if (bl === "background" && br === "background" && cutBottom < maxBottom) {
      h -= 1;
      cutBottom += 1;
      continue;
    }

    // Either every side passes, or every violating side has hit its
    // per-side cap. Either way we're done.
    const anyViolatingSideCapped =
      (tl === "background" && bl === "background" && cutLeft >= maxLeft) ||
      (tr === "background" && br === "background" && cutRight >= maxRight) ||
      (tl === "background" && tr === "background" && cutTop >= maxTop) ||
      (bl === "background" && br === "background" && cutBottom >= maxBottom);
    reason = anyViolatingSideCapped ? "cap_reached" : "no_violation";
    break;
  }

  const changed =
    x !== orig.x || y !== orig.y || w !== orig.w || h !== orig.h;
  return {
    box,
    shrunk: { x, y, w, h },
    changed,
    reason: changed ? reason : "no_violation",
    trimmed: {
      left: cutLeft,
      right: cutRight,
      top: cutTop,
      bottom: cutBottom,
    },
  };
}

/**
 * Apply shrink results back onto the original box objects, mutating
 * geometry in place. Returns the count of boxes that changed, for
 * logging. Use this when you want the cache/state to be updated with
 * the tightened coords.
 */
export function applyShrinkInPlace(results: ShrinkResult[]): number {
  let changed = 0;
  for (const r of results) {
    if (!r.changed) continue;
    r.box.x = r.shrunk.x;
    r.box.y = r.shrunk.y;
    r.box.w = r.shrunk.w;
    r.box.h = r.shrunk.h;
    changed += 1;
  }
  return changed;
}

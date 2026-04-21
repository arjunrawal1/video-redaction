// Server-side JPEG compositor for teamwork mode.
//
// Given a JPEG and a list of pixel rectangles, return a new JPEG with each
// rectangle painted solid black. Used to "redact" OCR-found boxes before
// asking Gemini whether anything query-matching is still visible.
//
// We use sharp.composite with synthesized solid-color tiles rather than an
// SVG overlay because sharp's SVG rasterizer is optional at install time;
// raw `create` tiles are always available.

import sharp from "sharp";

export type FillRect = { x: number; y: number; w: number; h: number };

/**
 * Paint each rectangle in `rects` solid black on top of `jpeg`. Returns a
 * fresh JPEG buffer. If `rects` is empty we still re-encode once so the
 * caller gets a consistent bytestream (small cost, ~2-5ms).
 */
export async function fillRectangles(
  jpeg: Uint8Array,
  rects: FillRect[],
  opts: { color?: { r: number; g: number; b: number; alpha?: number } } = {},
): Promise<Uint8Array> {
  const color = opts.color ?? { r: 0, g: 0, b: 0, alpha: 1 };
  const pipeline = sharp(jpeg);

  // We need image dims to clamp boxes inside the frame so sharp doesn't
  // throw on out-of-bounds composites.
  const { width = 0, height = 0 } = await pipeline.metadata();

  const clamped = rects
    .map((r) => {
      const x = Math.max(0, Math.min(width, Math.floor(r.x)));
      const y = Math.max(0, Math.min(height, Math.floor(r.y)));
      const w = Math.max(0, Math.min(width - x, Math.floor(r.w)));
      const h = Math.max(0, Math.min(height - y, Math.floor(r.h)));
      return { left: x, top: y, width: w, height: h };
    })
    .filter((r) => r.width > 0 && r.height > 0);

  if (clamped.length === 0) {
    const buf = await sharp(jpeg).jpeg({ quality: 92 }).toBuffer();
    return new Uint8Array(buf);
  }

  const composites = clamped.map((r) => ({
    input: {
      create: {
        width: r.width,
        height: r.height,
        channels: 4 as const,
        background: color,
      },
    },
    left: r.left,
    top: r.top,
  }));

  const out = await sharp(jpeg)
    .composite(composites)
    .jpeg({ quality: 92 })
    .toBuffer();
  return new Uint8Array(out);
}

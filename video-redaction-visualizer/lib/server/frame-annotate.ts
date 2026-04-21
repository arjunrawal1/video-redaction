// Render a JPEG with redaction boxes drawn on top for human review.
//
// Used by the run-log machinery to dump visually-annotated frames into
// the logs folder so we can eyeball "did the curator/linker/gap-filler
// land the box on the right pixels?" without replaying the stream in
// the browser.
//
// We use sharp + an SVG overlay. sharp ships librsvg on mainstream
// prebuilts (macOS + glibc linux), so rendering <rect> + <text> is
// usually fine. On platforms where SVG rasterization is unavailable,
// sharp throws and the caller logs + skips — annotation is a debug
// artifact, never critical path.
//
// Style choices: hollow red rectangles with a 1 px black inner outline
// for contrast against bright backgrounds, and a tiny red label above
// each box (box index + short text). Everything is rendered on top of
// the original frame — the JPEG underneath is unchanged pixels, not
// redacted.

import sharp from "sharp";

export type AnnotateBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  /** Short label rendered above the box. Optional. */
  label?: string | null;
  /**
   * Origin tag — used to pick a stroke color so you can tell at a glance
   * which stage added the box. Falls back to red when unset.
   */
  origin?: "ocr" | "fix" | "nav-add" | "gap-fill" | string | null;
};

const COLOR_BY_ORIGIN: Record<string, string> = {
  ocr: "#2563eb", // blue — Textract seed
  fix: "#dc2626", // red — curator addition
  "nav-add": "#f59e0b", // amber — navigator addition
  "gap-fill": "#10b981", // green — gap-filler insert
};

function colorFor(origin: string | null | undefined): string {
  if (!origin) return "#dc2626";
  return COLOR_BY_ORIGIN[origin] ?? "#dc2626";
}

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function buildSvg(
  width: number,
  height: number,
  boxes: AnnotateBox[],
): string {
  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
  );
  // Small font-size that stays readable on 1600-wide frames without
  // dominating the image.
  const fontSize = Math.max(10, Math.round(Math.min(width, height) / 80));
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    const color = colorFor(b.origin);
    const x = Math.max(0, Math.round(b.x));
    const y = Math.max(0, Math.round(b.y));
    const w = Math.max(1, Math.round(b.w));
    const h = Math.max(1, Math.round(b.h));
    parts.push(
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="2" />`,
    );
    // Inner black hairline so the red box doesn't vanish over red UI chrome.
    parts.push(
      `<rect x="${x + 1}" y="${y + 1}" width="${Math.max(1, w - 2)}" height="${Math.max(1, h - 2)}" fill="none" stroke="#000000" stroke-opacity="0.5" stroke-width="1" />`,
    );
    const rawLabel = `#${i}${b.label ? ` ${b.label}` : ""}`;
    const label = escapeXml(rawLabel.slice(0, 48));
    // Label sits just above the box, or just inside the top edge if the
    // box starts at y=0.
    const labelY = y > fontSize + 4 ? y - 4 : y + fontSize + 2;
    parts.push(
      `<text x="${x + 2}" y="${labelY}" font-family="sans-serif" font-size="${fontSize}" font-weight="bold" fill="${color}" stroke="#000000" stroke-width="0.5" paint-order="stroke">${label}</text>`,
    );
  }
  parts.push(`</svg>`);
  return parts.join("");
}

/**
 * Draw `boxes` on top of `jpeg` and return a fresh JPEG buffer.
 *
 * The base image isn't modified — we just overlay an SVG with hollow
 * rectangles + labels. If sharp can't rasterize SVG on this platform
 * the call throws; the caller is expected to catch and log.
 */
export async function annotateFrame(
  jpeg: Uint8Array,
  boxes: AnnotateBox[],
): Promise<Uint8Array> {
  const meta = await sharp(jpeg).metadata();
  const width = meta.width ?? 0;
  const height = meta.height ?? 0;
  if (width === 0 || height === 0) {
    // Unreadable dims — just re-encode unchanged so the caller still
    // gets a JPEG for the log, better than nothing.
    const out = await sharp(jpeg).jpeg({ quality: 90 }).toBuffer();
    return new Uint8Array(out);
  }

  if (boxes.length === 0) {
    const out = await sharp(jpeg).jpeg({ quality: 90 }).toBuffer();
    return new Uint8Array(out);
  }

  const svg = buildSvg(width, height, boxes);
  const out = await sharp(jpeg)
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .jpeg({ quality: 90 })
    .toBuffer();
  return new Uint8Array(out);
}

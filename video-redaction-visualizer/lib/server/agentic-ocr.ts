// Shared helpers for turning a raw Textract response into something a
// language model can consume: (1) flat list of WORD/LINE blocks in
// Gemini's native box_2d format, and (2) relevance ranking against a
// query so we can prune/rank before stuffing into a prompt.
//
// Originally lived inline in agentic-curator.ts. Extracted so the
// focused-agent (cascade) path can inline the same raw-OCR dump into
// its initial user message and skip the first 2–4 round-trips it used
// to spend fetching the focus frame's OCR via `get_ocr_text`.

export type TextractBlock = {
  BlockType?: string;
  Text?: string;
  Confidence?: number;
  Geometry?: {
    BoundingBox?: {
      Left?: number;
      Top?: number;
      Width?: number;
      Height?: number;
    };
  };
};

export type RawOcrItem = {
  type: "word" | "line";
  text: string;
  confidence: number;
  /** Gemini native box_2d: [y_min, x_min, y_max, x_max] in integer 0..1000. */
  bbox: [number, number, number, number];
};

/**
 * Pull every WORD and LINE Textract block off the raw response and
 * convert to Gemini's native box_2d contract.
 */
export function extractRawOcrItems(raw: unknown): RawOcrItem[] {
  if (!raw || typeof raw !== "object") return [];
  const blocks: TextractBlock[] =
    (raw as { Blocks?: TextractBlock[] } | null)?.Blocks ?? [];
  const out: RawOcrItem[] = [];
  for (const b of blocks) {
    const t = b.BlockType;
    if (t !== "WORD" && t !== "LINE") continue;
    const bbox = b.Geometry?.BoundingBox;
    if (!bbox) continue;
    const left = Number(bbox.Left ?? 0);
    const top = Number(bbox.Top ?? 0);
    const w = Number(bbox.Width ?? 0);
    const h = Number(bbox.Height ?? 0);
    out.push({
      type: t === "WORD" ? "word" : "line",
      text: String(b.Text ?? ""),
      confidence: Math.round(Number(b.Confidence ?? 0) * 10) / 10,
      bbox: [
        Math.round(top * 1000),
        Math.round(left * 1000),
        Math.round((top + h) * 1000),
        Math.round((left + w) * 1000),
      ],
    });
  }
  return out;
}

/**
 * Rank raw OCR items by plausible relevance to a query and split them
 * into a "relevant" head (likely matches / substrings / fuzzy variants)
 * and a "context" tail (other LINE blocks, capped for disambiguation).
 *
 * Used by both the curator (per-frame OCR review) and the focused
 * cascade agent (inline focus-frame dump in the initial prompt).
 */
export function filterAndRankOcrItems(
  items: RawOcrItem[],
  query: string,
  maxItems = 120,
): { relevant: RawOcrItem[]; context: RawOcrItem[] } {
  const q = query.toLowerCase();
  const qWords = q.split(/\s+/).filter((w) => w.length >= 2);

  const sharedLen = (a: string, b: string): number => {
    if (!a || !b) return 0;
    const lo = a.toLowerCase();
    const rhs = b.toLowerCase();
    let best = 0;
    for (let i = 0; i < lo.length; i++) {
      for (let j = i + 1; j <= lo.length; j++) {
        const slice = lo.slice(i, j);
        if (slice.length <= best) continue;
        if (rhs.includes(slice)) best = slice.length;
      }
    }
    return best;
  };

  const scored: Array<{ item: RawOcrItem; score: number }> = items.map((it) => {
    const t = it.text.toLowerCase();
    const viaQ = qWords.reduce(
      (m, w) => Math.max(m, t.includes(w) ? w.length : 0),
      0,
    );
    const viaShared = sharedLen(t, q);
    return { item: it, score: Math.max(viaQ, viaShared) };
  });

  const relevantScored = scored.filter((s) => s.score >= 2);
  relevantScored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    // Prefer LINE blocks when scores tie — they're more useful for
    // copying bboxes of multi-token matches.
    if (a.item.type !== b.item.type) return a.item.type === "line" ? -1 : 1;
    return b.item.confidence - a.item.confidence;
  });

  const relevant = relevantScored
    .map((s) => s.item)
    .slice(0, Math.min(maxItems, 60));
  const remaining = maxItems - relevant.length;
  const context = items
    .filter((it) => it.type === "line")
    .filter((it) => !relevant.includes(it))
    .slice(0, Math.max(0, remaining));
  return { relevant, context };
}

export function formatOcrItemsForPrompt(items: RawOcrItem[]): string {
  if (items.length === 0) return "(none)";
  return items
    .map(
      (it, i) =>
        `${it.type.toUpperCase()}[${i}] text=${JSON.stringify(it.text)} conf=${it.confidence} bbox=[${it.bbox.join(", ")}]`,
    )
    .join("\n");
}

"""Second-pass backtracking: retro-detect partials of the query that the
first pass missed.

Algorithm (see plan: "Backtracking second pass"):

- Walk frames in reverse (`frame_to` down to `frame_from + 1`).
- For each hit `H` in frame `n`, check whether it has a predecessor in
  frame `n-1` (same text, roughly same spot). If so, H is the same text
  that just moved -- nothing to do.
- If no predecessor, H is a "genuinely new" hit. Use H's box as a lens on
  frame `n-1`'s Textract output and look for any block whose text is a
  contiguous substring of the query (or fuzzy-close to one), within the
  region surrounding H.
- If a partial is found, record it as a backtracked hit on `n-1`. Because
  we mutate the per-frame list in place, when the outer loop reaches `n-1`
  it sees this new hit as a "current-frame" hit and checks against `n-2`,
  which is what gives us the recursive behavior through iteration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from app.ocr_cache import FramePhase1, OcrEntry
from app.ocr_service import (
    Box,
    _bbox_to_rect,
    _fuzzy_contains,
    _line_child_words,
    _normalize,
)

log = logging.getLogger(__name__)


_SPATIAL_MARGIN = float(os.getenv("BACKTRACK_SPATIAL_MARGIN", "0.5"))
_MIN_PARTIAL_CHARS = int(os.getenv("BACKTRACK_MIN_PARTIAL_CHARS", "2"))
_LINK_CENTER_DIST = float(os.getenv("BACKTRACK_LINK_CENTER_DIST", "0.08"))
_FUZZY_THRESHOLD = float(os.getenv("BACKTRACK_FUZZY_THRESHOLD", "0.82"))


@dataclass(frozen=True)
class AddedHit:
    """A hit produced by the backtrack pass. Includes the frame it landed on
    so the streaming driver can emit one NDJSON event per added hit."""

    frame_idx_1: int
    box: Box


# ---------------------------------------------------------------------------
# Correspondence between a current-frame hit and candidate predecessors.
# ---------------------------------------------------------------------------


def _pixel_center(box: Box, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Normalized (0..1) center of a pixel-space Box."""
    cx = (box.x + box.w / 2.0) / max(frame_w, 1)
    cy = (box.y + box.h / 2.0) / max(frame_h, 1)
    return cx, cy


def _text_matches(a: str, b: str) -> bool:
    """Rough text equality: exact substring either way or fuzzy at 0.82."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    # Same threshold as the detect pass; either side could be the "haystack".
    return _fuzzy_contains(na, nb, _FUZZY_THRESHOLD) or _fuzzy_contains(
        nb, na, _FUZZY_THRESHOLD
    )


def _has_correspondence(
    current: Box,
    current_frame: FramePhase1,
    prev_frame: FramePhase1,
) -> bool:
    """Is there a hit in `prev_frame` that is plausibly the same piece of
    text (same wording, close position) as `current`?"""
    cw, ch = current_frame.width, current_frame.height
    pw, ph = prev_frame.width, prev_frame.height
    cx, cy = _pixel_center(current, cw, ch)
    for prev in prev_frame.matched:
        if not _text_matches(current.text, prev.text):
            continue
        px, py = _pixel_center(prev, pw, ph)
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if dist <= _LINK_CENTER_DIST:
            return True
    return False


# ---------------------------------------------------------------------------
# Finding a partial inside frame n-1 near the region of hit H.
# ---------------------------------------------------------------------------


def _box_to_normalized_rect(
    box: Box, frame_w: int, frame_h: int
) -> tuple[float, float, float, float]:
    """Convert a pixel Box to (left, top, right, bottom) in 0..1 space."""
    left = box.x / max(frame_w, 1)
    top = box.y / max(frame_h, 1)
    right = (box.x + box.w) / max(frame_w, 1)
    bottom = (box.y + box.h) / max(frame_h, 1)
    return left, top, right, bottom


def _expand(
    rect: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    """Expand a normalized rect by `margin` of its current size on each side,
    clamped to the unit square."""
    left, top, right, bottom = rect
    w = right - left
    h = bottom - top
    pad_x = w * margin
    pad_y = h * margin
    return (
        max(0.0, left - pad_x),
        max(0.0, top - pad_y),
        min(1.0, right + pad_x),
        min(1.0, bottom + pad_y),
    )


def _bbox_overlaps(block_bbox: dict, region: tuple[float, float, float, float]) -> bool:
    """Does a Textract BoundingBox dict (normalized) overlap the region?"""
    if not block_bbox:
        return False
    bleft = float(block_bbox.get("Left", 0.0))
    btop = float(block_bbox.get("Top", 0.0))
    bright = bleft + float(block_bbox.get("Width", 0.0))
    bbottom = btop + float(block_bbox.get("Height", 0.0))
    rleft, rtop, rright, rbottom = region
    return not (
        bright <= rleft
        or bleft >= rright
        or bbottom <= rtop
        or btop >= rbottom
    )


def _substring_score(t: str, query: str) -> float:
    """Score how well `t` looks like a contiguous substring of `query`.

    Returns 0.0 if no plausible match, else a ratio in (0, 1]. Exact
    substring wins outright at 1.0; otherwise we scan windows of `query` of
    length `len(t) ± 2` and take the best SequenceMatcher ratio.
    """
    if not t or not query:
        return 0.0
    if t in query:
        return 1.0
    n = len(t)
    best = 0.0
    for delta in (0, -1, 1, -2, 2):
        wlen = n + delta
        if wlen < 2 or wlen > len(query):
            continue
        for i in range(0, len(query) - wlen + 1):
            window = query[i : i + wlen]
            ratio = SequenceMatcher(None, t, window).ratio()
            if ratio > best:
                best = ratio
    return best


def _union_words_rect(
    words: list[dict], frame_w: int, frame_h: int
) -> tuple[int, int, int, int] | None:
    """Union pixel rect across a list of WORD blocks."""
    xs, ys, xe, ye = 1.0, 1.0, 0.0, 0.0
    any_box = False
    for w in words:
        bbox = (w.get("Geometry") or {}).get("BoundingBox")
        if not bbox:
            continue
        left = float(bbox.get("Left", 0.0))
        top = float(bbox.get("Top", 0.0))
        width = float(bbox.get("Width", 0.0))
        height = float(bbox.get("Height", 0.0))
        xs = min(xs, left)
        ys = min(ys, top)
        xe = max(xe, left + width)
        ye = max(ye, top + height)
        any_box = True
    if not any_box or xe <= xs or ye <= ys:
        return None
    return _bbox_to_rect(
        {"Left": xs, "Top": ys, "Width": xe - xs, "Height": ye - ys},
        frame_w,
        frame_h,
    )


def _tighten_line_to_partial(
    line_block: dict,
    block_by_id: dict,
    query_norm: str,
    frame_w: int,
    frame_h: int,
) -> tuple[tuple[int, int, int, int], str, float] | None:
    """For a LINE block, find the tightest contiguous run of child words whose
    joined text is a substring (or close to one) of `query_norm`."""
    words = _line_child_words(line_block, block_by_id)
    if not words:
        return None
    norms = [_normalize(str(w.get("Text", "") or "")) for w in words]
    n = len(norms)

    best_score = 0.0
    best_span: tuple[int, int] | None = None
    best_text = ""
    for i in range(n):
        for j in range(i, n):
            joined = " ".join(norms[i : j + 1])
            if not joined or len(joined) < _MIN_PARTIAL_CHARS:
                continue
            score = _substring_score(joined, query_norm)
            if score <= 0.0:
                continue
            if score > best_score or (
                score == best_score
                and best_span is not None
                and (j - i) < (best_span[1] - best_span[0])
            ):
                best_score = score
                best_span = (i, j)
                best_text = joined
    if best_span is None or best_score < _FUZZY_THRESHOLD:
        return None
    rect = _union_words_rect(words[best_span[0] : best_span[1] + 1], frame_w, frame_h)
    if rect is None:
        return None
    return rect, best_text, best_score


# ---------------------------------------------------------------------------
# Top-level partial finder: scan a whole frame and pick the best candidate.
# ---------------------------------------------------------------------------


def _find_partial(
    original_hit: Box,
    current_frame: FramePhase1,
    prev_frame: FramePhase1,
    query_norm: str,
) -> Box | None:
    """Find a partial of the query in `prev_frame`'s Textract data, within
    the region around `original_hit`. Returns a pixel-space `Box` suitable
    for adding to `prev_frame.matched`, or None if nothing qualifies."""
    raw = prev_frame.raw or {}
    blocks = raw.get("Blocks") or []
    if not blocks:
        return None

    # Define the search region in frame n-1's normalized coords. We translate
    # the pixel box from frame n into normalized units on frame n-1 under the
    # reasonable assumption that both frames have similar aspect ratios.
    src_rect = _box_to_normalized_rect(
        original_hit, current_frame.width, current_frame.height
    )
    region = _expand(src_rect, _SPATIAL_MARGIN)

    block_by_id = {b["Id"]: b for b in blocks if b.get("Id")}

    # Track the best candidate. Ranking key (per plan): longest text wins;
    # ties broken by match score, then confidence. All candidates already
    # clear _FUZZY_THRESHOLD so "quality" is a given.
    best_key: tuple[int, float, float] | None = None
    best_out: Box | None = None

    for b in blocks:
        btype = b.get("BlockType")
        if btype not in ("WORD", "LINE"):
            continue
        bbox = (b.get("Geometry") or {}).get("BoundingBox")
        if not _bbox_overlaps(bbox or {}, region):
            continue
        text = str(b.get("Text", "") or "")
        norm = _normalize(text)
        if len(norm) < _MIN_PARTIAL_CHARS:
            continue
        score = _substring_score(norm, query_norm)
        if score < _FUZZY_THRESHOLD:
            continue

        if btype == "LINE":
            tightened = _tighten_line_to_partial(
                b, block_by_id, query_norm, prev_frame.width, prev_frame.height
            )
            if tightened is not None:
                rect, ttext, tscore = tightened
                score = max(score, tscore)
                text = ttext
            else:
                rect = _bbox_to_rect(
                    bbox or {}, prev_frame.width, prev_frame.height
                )
        else:
            rect = _bbox_to_rect(bbox or {}, prev_frame.width, prev_frame.height)

        conf = float(b.get("Confidence", 0.0) or 0.0) / 100.0
        key = (len(text), score, conf)
        if best_key is None or key > best_key:
            x, y, w, h = rect
            best_key = key
            best_out = Box(x=x, y=y, w=w, h=h, text=text, score=conf)

    return best_out


# ---------------------------------------------------------------------------
# Duplicate suppression so we don't re-add boxes we already have.
# ---------------------------------------------------------------------------


def _box_overlaps_existing(box: Box, existing: Iterable[Box]) -> bool:
    """Rough dedup: any existing box whose IoU with `box` exceeds 0.5 counts
    as a duplicate. Guards both against re-emitting during recursion and
    against shadowing first-pass hits."""
    ax1, ay1, ax2, ay2 = box.x, box.y, box.x + box.w, box.y + box.h
    a_area = max(1, box.w * box.h)
    for e in existing:
        bx1, by1, bx2, by2 = e.x, e.y, e.x + e.w, e.y + e.h
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            continue
        b_area = max(1, e.w * e.h)
        union = a_area + b_area - inter
        if union > 0 and inter / union >= 0.5:
            return True
    return False


# ---------------------------------------------------------------------------
# Iterative backward driver.
# ---------------------------------------------------------------------------


def run_backtrack(entry: OcrEntry) -> list[AddedHit]:
    """Walk frames top-down and append backtrack hits in place on entry.

    Returns the ordered list of added hits so callers can stream them.
    """
    added: list[AddedHit] = []
    lo = entry.frame_from
    hi = entry.frame_to

    for idx in range(hi, lo, -1):
        current = entry.per_frame.get(idx)
        prev = entry.per_frame.get(idx - 1)
        if current is None or prev is None:
            continue
        # Iterate over a snapshot so we don't re-scan a box we add this loop.
        current_hits = list(current.matched)
        for hit in current_hits:
            if _has_correspondence(hit, current, prev):
                continue
            partial = _find_partial(hit, current, prev, entry.query_norm)
            if partial is None:
                continue
            if _box_overlaps_existing(partial, prev.matched):
                continue
            prev.matched.append(partial)
            added.append(AddedHit(frame_idx_1=idx - 1, box=partial))
            log.info(
                "backtrack: frame #%d <- #%d text=%r conf=%.2f",
                idx - 1,
                idx,
                partial.text,
                partial.score,
            )

    return added


def run_forward(entry: OcrEntry) -> list[AddedHit]:
    """Mirror of `run_backtrack` in the forward direction.

    Walk frames bottom-up. For each hit `P` in frame `idx`, ask whether any
    hit in frame `idx+1` is plausibly the same text (same correspondence
    predicate as backtrack). If not, `P` has "retired" -- the hit went away
    but the text may still be partially present in `idx+1`. Scan `idx+1`'s
    raw Textract response near `P`'s region for a partial of the query and
    append it in-place. Because we mutate `next_frame.matched` while still
    within the loop, the next iteration (which treats `idx+1` as "current")
    sees the newly-added hit and can itself forward into `idx+2`, giving us
    recursive forward propagation through iteration.

    Used for the "text is gradually deleted / fades off screen" case that's
    the mirror of the "text is gradually typed in" case that backtrack
    solves. Core helpers (`_has_correspondence`, `_find_partial`,
    `_box_overlaps_existing`) are all temporally neutral, so we reuse them
    verbatim with source/target frames swapped.
    """
    added: list[AddedHit] = []
    lo = entry.frame_from
    hi = entry.frame_to

    for idx in range(lo, hi):
        current = entry.per_frame.get(idx)
        nxt = entry.per_frame.get(idx + 1)
        if current is None or nxt is None:
            continue
        # Snapshot: don't re-scan a box we added in this iteration.
        current_hits = list(current.matched)
        for hit in current_hits:
            # Same predicate, different frame pair: has any hit in the NEXT
            # frame plausibly inherited this text?
            if _has_correspondence(hit, current, nxt):
                continue
            partial = _find_partial(hit, current, nxt, entry.query_norm)
            if partial is None:
                continue
            if _box_overlaps_existing(partial, nxt.matched):
                continue
            nxt.matched.append(partial)
            added.append(AddedHit(frame_idx_1=idx + 1, box=partial))
            log.info(
                "forward: frame #%d -> #%d text=%r conf=%.2f",
                idx,
                idx + 1,
                partial.text,
                partial.score,
            )

    return added

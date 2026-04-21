"""In-memory LRU cache of per-frame OCR results keyed by video + query.

Populated by the detect/stream endpoint as each frame completes so the
backtrack endpoint can walk frames in reverse without re-running Textract.
Each cache value holds per-frame matched boxes, frame dims, and the raw
Textract response (needed to locate partials within a region).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from app.ocr_service import Box

_MAX_ENTRIES = 8


@dataclass
class FramePhase1:
    width: int
    height: int
    # Mutable on purpose: backtrack augments this list in place so that when
    # the iterative driver reaches frame n-1 it already sees the hits that
    # were backtracked in from frame n.
    matched: list[Box] = field(default_factory=list)
    # Raw Textract DetectDocumentText response for the frame. Used by the
    # backtrack pass to look for partials inside a given region.
    raw: dict | None = None


@dataclass
class OcrEntry:
    video_hash: str
    query_norm: str
    fps: float | None
    dedup_threshold: int
    # ``max_gap`` is part of the cache identity because it changes which
    # raw frames survive dedup, which in turn changes the 1-indexed
    # frame numbering the per-frame OCR results live under. Caching
    # across different ``max_gap`` values would silently alias unrelated
    # frame sets.
    max_gap: int
    frame_from: int
    frame_to: int
    # 1-indexed frame number -> phase-1 state.
    per_frame: dict[int, FramePhase1] = field(default_factory=dict)


_CacheKey = tuple[str, str, float | None, int, int, int, int]
_cache: "OrderedDict[_CacheKey, OcrEntry]" = OrderedDict()
_lock = threading.Lock()


def _make_key(
    video_hash: str,
    query_norm: str,
    fps: float | None,
    dedup_threshold: int,
    max_gap: int,
    frame_from: int,
    frame_to: int,
) -> _CacheKey:
    return (
        video_hash,
        query_norm,
        fps,
        dedup_threshold,
        max_gap,
        frame_from,
        frame_to,
    )


def create_entry(
    *,
    video_hash: str,
    query_norm: str,
    fps: float | None,
    dedup_threshold: int,
    max_gap: int,
    frame_from: int,
    frame_to: int,
) -> OcrEntry:
    """Create a fresh entry and register it as the most-recently-used.

    If an entry with the same key already exists we overwrite it so callers
    always see a clean slate on a new phase-1 run (re-running detect with the
    same query should reset any backtracked state).
    """
    entry = OcrEntry(
        video_hash=video_hash,
        query_norm=query_norm,
        fps=fps,
        dedup_threshold=dedup_threshold,
        max_gap=max_gap,
        frame_from=frame_from,
        frame_to=frame_to,
    )
    key = _make_key(
        video_hash, query_norm, fps, dedup_threshold, max_gap, frame_from, frame_to,
    )
    with _lock:
        _cache[key] = entry
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)
    return entry


def get_entry(
    *,
    video_hash: str,
    query_norm: str,
    fps: float | None,
    dedup_threshold: int,
    max_gap: int,
    frame_from: int,
    frame_to: int,
) -> OcrEntry | None:
    """Return the cached entry for this key, or None. Touches LRU order."""
    key = _make_key(
        video_hash, query_norm, fps, dedup_threshold, max_gap, frame_from, frame_to,
    )
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
        return hit


def put_frame(
    entry: OcrEntry,
    *,
    frame_idx_1: int,
    width: int,
    height: int,
    matched: list[Box],
    raw: dict,
) -> None:
    """Record phase-1 output for a single frame on the given entry."""
    entry.per_frame[frame_idx_1] = FramePhase1(
        width=width,
        height=height,
        matched=list(matched),
        raw=raw,
    )

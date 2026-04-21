"""In-memory LRU cache of extracted+deduplicated frames keyed by video hash.

Keeps repeated OCR/detection calls against the same upload from paying the
ffmpeg + perceptual-hash cost every time.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

from app.frame_service import extract_deduplicated_frames

_MAX_ENTRIES = 4


@dataclass(frozen=True)
class FrameSet:
    frames: tuple[bytes, ...]
    raw_count: int
    kept_count: int
    # 0-based indices into the original ffmpeg-emitted sequence for each
    # kept frame. `kept_source_indices[k] / source_fps` is the source-video
    # timestamp of the k-th kept frame. Used by the video exporter to paint
    # detection boxes onto the right time windows.
    kept_source_indices: tuple[int, ...]
    # Effective frames-per-second of the extraction. When the caller passed
    # an explicit fps it's that value; when they let ffmpeg emit every
    # decoded frame we probe the source with ffprobe. `None` means probing
    # failed and timestamp-based features are unavailable.
    source_fps: float | None


_CacheKey = tuple[str, float | None, int, int]
_cache: "OrderedDict[_CacheKey, FrameSet]" = OrderedDict()
_lock = threading.Lock()


def _hash_video(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _make_key(
    video_hash: str,
    fps: float | None,
    dedup_threshold: int,
    max_width: int,
) -> _CacheKey:
    return (video_hash, fps, dedup_threshold, max_width)


def get_or_extract(
    body: bytes,
    *,
    suffix: str,
    fps: float | None,
    dedup_threshold: int,
    max_width: int = 1600,
) -> tuple[FrameSet, str]:
    """Return cached frames for this video+params or extract and cache them.

    Returns the FrameSet and the video hash so callers can reuse it as a key
    for their own derived caches (e.g. OCR results).
    """
    video_hash = _hash_video(body)
    key = _make_key(video_hash, fps, dedup_threshold, max_width)

    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit, video_hash

    (
        blobs,
        raw_count,
        kept_count,
        kept_source_indices,
        source_fps,
    ) = extract_deduplicated_frames(
        body,
        suffix=suffix,
        fps=fps,
        dedup_threshold=dedup_threshold,
        max_width=max_width,
    )
    frame_set = FrameSet(
        frames=tuple(blobs),
        raw_count=raw_count,
        kept_count=kept_count,
        kept_source_indices=kept_source_indices,
        source_fps=source_fps,
    )

    with _lock:
        _cache[key] = frame_set
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)

    return frame_set, video_hash

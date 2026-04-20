"""AWS Textract-backed text detection with robust fuzzy matching.

We call `DetectDocumentText` per frame (sync). It returns per-WORD blocks with
normalized bounding boxes and a 0-100 confidence score. The matching pipeline
(normalize + substring + fuzzy) and public API (`detect_frame`,
`normalize_query`, `ensure_reader_loaded`, `warm`) are unchanged so the
streaming endpoint and the overlay UI don't need to know we swapped engines.

Auth: boto3 reads creds from standard env vars (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, `AWS_REGION`) or the
shared config/credentials files.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Sequence

from PIL import Image

log = logging.getLogger(__name__)

# Fuzzy ratio threshold (0..1). Substring match always wins; this only governs
# the sliding-window fallback that absorbs 1-2 OCR errors.
_FUZZY_THRESHOLD = float(os.getenv("OCR_FUZZY_THRESHOLD", "0.82"))
# Textract confidence is 0-100. Screenshot text usually comes back > 90; a
# low floor here keeps small/faint text in the candidate pool so our matcher
# can still catch it.
_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "10"))
# Padding applied to each matched box so the redaction rectangle fully
# covers the glyphs.
_BOX_PAD_PX = int(os.getenv("OCR_BOX_PAD_PX", "2"))
# Simple retry for Textract throttling.
_MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "3"))

_client: Any = None
_client_lock = threading.Lock()


def _get_client():
    """Create (and cache) a Textract boto3 client. Network-less; creds are
    used on first request, not at construction time."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            import boto3  # lazy to keep module import cheap

            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
            t0 = time.perf_counter()
            log.info("Creating Textract client (region=%s)...", region)
            _client = boto3.client("textract", region_name=region)
            log.info(
                "Textract client ready in %.0fms (has_access_key=%s)",
                (time.perf_counter() - t0) * 1000,
                bool(os.getenv("AWS_ACCESS_KEY_ID")),
            )
    return _client


def warm() -> None:
    """Construct the client eagerly so the first request doesn't pay for it."""
    try:
        _get_client()
    except Exception:  # pragma: no cover
        log.exception("Textract client warmup failed; will retry per-request.")


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    text: str
    score: float


@dataclass(frozen=True)
class FrameDetections:
    width: int
    height: int
    boxes: tuple[Box, ...]


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _fuzzy_contains(haystack: str, needle: str, threshold: float) -> bool:
    """Return True if a sliding window of `haystack` roughly matches `needle`."""
    if not needle:
        return False
    if needle in haystack:
        return True
    n = len(needle)
    best = 0.0
    for delta in (0, -1, 1, -2, 2):
        wlen = n + delta
        if wlen < 3 or wlen > len(haystack):
            continue
        for i in range(0, len(haystack) - wlen + 1):
            window = haystack[i : i + wlen]
            ratio = SequenceMatcher(None, needle, window).ratio()
            if ratio > best:
                best = ratio
                if best >= threshold:
                    return True
    if len(haystack) <= n + 2:
        if SequenceMatcher(None, needle, haystack).ratio() >= threshold:
            return True
    return False


def _get_image_size(blob: bytes) -> tuple[int, int]:
    """Read only the JPEG header to get (width, height). Pillow's open is lazy."""
    with Image.open(io.BytesIO(blob)) as im:
        return im.size  # (w, h)


def _bbox_to_rect(
    bbox: dict, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Textract BoundingBox is normalized {Left, Top, Width, Height} in 0..1."""
    left = float(bbox.get("Left", 0.0)) * frame_w
    top = float(bbox.get("Top", 0.0)) * frame_h
    width = float(bbox.get("Width", 0.0)) * frame_w
    height = float(bbox.get("Height", 0.0)) * frame_h
    pad = _BOX_PAD_PX
    x = max(0, int(round(left)) - pad)
    y = max(0, int(round(top)) - pad)
    w = min(frame_w - x, int(round(width)) + 2 * pad)
    h = min(frame_h - y, int(round(height)) + 2 * pad)
    return x, y, max(1, w), max(1, h)


def _line_child_words(line_block: dict, block_by_id: dict) -> list[dict]:
    """Return the WORD blocks that make up a LINE, in reading order."""
    ids: list[str] = []
    for rel in line_block.get("Relationships") or []:
        if rel.get("Type") == "CHILD":
            ids.extend(rel.get("Ids") or [])
    words: list[dict] = []
    for cid in ids:
        b = block_by_id.get(cid)
        if b and b.get("BlockType") == "WORD":
            words.append(b)
    return words


def _subspan_bbox(
    line_block: dict,
    block_by_id: dict,
    query_norm: str,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    """Find the tightest contiguous run of WORD children of `line_block` that
    matches `query_norm`, and return its pixel bounding box.

    Strategy:
      1. Shortest contiguous subspan whose joined normalized text contains
         `query_norm` as a substring (ties broken by span length).
      2. Otherwise, subspan with the highest SequenceMatcher ratio against
         `query_norm`, if it clears `_FUZZY_THRESHOLD`.
      3. Otherwise, return None (caller falls back to the whole-line box).
    """
    words = _line_child_words(line_block, block_by_id)
    if not words:
        return None

    norms = [_normalize(str(w.get("Text", "") or "")) for w in words]
    n = len(norms)

    # Pass 1: shortest substring match.
    best_sub: tuple[int, int, int] | None = None  # (i, j, joined_len)
    for i in range(n):
        for j in range(i, n):
            joined = " ".join(norms[i : j + 1])
            if not joined:
                continue
            if query_norm in joined:
                length = len(joined)
                if best_sub is None or length < best_sub[2]:
                    best_sub = (i, j, length)

    if best_sub is not None:
        lo, hi = best_sub[0], best_sub[1]
    else:
        # Pass 2: fuzzy best-ratio.
        best: tuple[int, int, float] = (-1, -1, 0.0)
        for i in range(n):
            for j in range(i, n):
                joined = " ".join(norms[i : j + 1])
                if not joined:
                    continue
                ratio = SequenceMatcher(None, query_norm, joined).ratio()
                if ratio > best[2]:
                    best = (i, j, ratio)
        if best[2] < _FUZZY_THRESHOLD:
            return None
        lo, hi = best[0], best[1]

    # Union the chosen words' bounding boxes in normalized space, then
    # convert once at the end so padding/clamping is applied consistently.
    xs, ys, xe, ye = 1.0, 1.0, 0.0, 0.0
    any_box = False
    for w in words[lo : hi + 1]:
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


def _call_textract(blob: bytes) -> dict:
    """Call Textract with exponential backoff on throttling errors."""
    from botocore.exceptions import ClientError  # lazy

    client = _get_client()
    delay = 0.5
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return client.detect_document_text(Document={"Bytes": blob})
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            retriable = code in (
                "ThrottlingException",
                "ProvisionedThroughputExceededException",
                "ServiceUnavailable",
                "InternalServerError",
            )
            last_err = e
            log.warning(
                "Textract error (attempt %d/%d) code=%s: %s",
                attempt + 1,
                _MAX_RETRIES,
                code or "?",
                e,
            )
            if not retriable or attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    if last_err is not None:  # pragma: no cover
        raise last_err
    raise RuntimeError("Textract call failed without error")


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of botocore response to JSON-serializable form."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:  # pragma: no cover
        return {"__unserializable__": repr(obj)}


def _detect_frame(blob: bytes, query_norm: str) -> tuple[FrameDetections, dict]:
    t_start = time.perf_counter()
    frame_w, frame_h = _get_image_size(blob)
    t_dim = time.perf_counter()

    response = _call_textract(blob)
    t_ocr = time.perf_counter()

    blocks = response.get("Blocks", []) or []
    # Multi-word queries must be matched against LINE blocks, because each
    # WORD block only ever holds a single whitespace-free token. For
    # single-word queries we stick to WORD blocks so the redaction box hugs
    # just the matched token rather than its entire line.
    query_has_space = " " in query_norm
    target_type = "LINE" if query_has_space else "WORD"
    candidates = [b for b in blocks if b.get("BlockType") == target_type]
    # For LINE-mode tightening we need O(1) lookups into child WORD blocks.
    block_by_id: dict[str, dict] = (
        {b["Id"]: b for b in blocks if b.get("Id")}
        if target_type == "LINE"
        else {}
    )

    matched: list[Box] = []
    low_conf = 0
    tightened = 0
    for b in candidates:
        text = str(b.get("Text", "") or "")
        conf = float(b.get("Confidence", 0.0) or 0.0)
        if conf < _MIN_CONFIDENCE:
            low_conf += 1
            continue
        norm = _normalize(text)
        if not norm:
            continue
        if not _fuzzy_contains(norm, query_norm, _FUZZY_THRESHOLD):
            continue

        rect: tuple[int, int, int, int] | None = None
        if target_type == "LINE":
            rect = _subspan_bbox(b, block_by_id, query_norm, frame_w, frame_h)
            if rect is not None:
                tightened += 1
        if rect is None:
            bbox = (b.get("Geometry") or {}).get("BoundingBox")
            if not bbox:
                continue
            rect = _bbox_to_rect(bbox, frame_w, frame_h)

        x, y, bw, bh = rect
        matched.append(
            Box(x=x, y=y, w=bw, h=bh, text=text, score=conf / 100.0)
        )
    t_match = time.perf_counter()

    raw = _jsonable(response)

    log.debug(
        "frame OCR: size=%dx%d mode=%s candidates=%d low_conf=%d matched=%d "
        "tightened=%d dim=%.0fms textract=%.0fms match=%.1fms total=%.0fms "
        "bytes=%d",
        frame_w,
        frame_h,
        target_type,
        len(candidates),
        low_conf,
        len(matched),
        tightened,
        (t_dim - t_start) * 1000,
        (t_ocr - t_dim) * 1000,
        (t_match - t_ocr) * 1000,
        (t_match - t_start) * 1000,
        len(blob),
    )
    # Full Textract payload for post-mortem / UI copy button. Compact JSON so
    # each frame is a single grep-able line.
    log.debug(
        "textract response size=%dx%d blocks=%d mode=%s candidates=%d :: %s",
        frame_w,
        frame_h,
        len(blocks),
        target_type,
        len(candidates),
        json.dumps(raw, separators=(",", ":")),
    )

    return (
        FrameDetections(width=frame_w, height=frame_h, boxes=tuple(matched)),
        raw,
    )


def normalize_query(q: str) -> str:
    """Public: normalize a query string exactly as the matcher does."""
    return _normalize(q)


def ensure_reader_loaded() -> None:
    """Public: construct the Textract client (idempotent, thread-safe)."""
    _get_client()


def detect_frame(blob: bytes, query_norm: str) -> tuple[FrameDetections, dict]:
    """Public: run OCR + match on one JPEG frame.

    Returns (detections, raw_textract_response). `query_norm` must already be
    normalized via `normalize_query`. If `query_norm` is empty we skip the
    Textract call and return an empty response.
    """
    if not query_norm:
        return FrameDetections(width=0, height=0, boxes=()), {}
    return _detect_frame(blob, query_norm)


def detect_text_in_frames(
    frames: Sequence[bytes],
    query: str,
) -> list[tuple[FrameDetections, dict]]:
    """Batch helper for non-streaming callers. Sequential; the streaming
    endpoint does its own fan-out."""
    query_norm = _normalize(query)
    if not query_norm:
        return [
            (FrameDetections(width=0, height=0, boxes=()), {}) for _ in frames
        ]
    ensure_reader_loaded()
    return [_detect_frame(blob, query_norm) for blob in frames]

"""Extract frames from video with ffmpeg and deduplicate via perceptual hashing."""

from __future__ import annotations

import glob
import json
import subprocess
import tempfile
from pathlib import Path

import imagehash
from PIL import Image

# Perceptual hash resolution. 16x16 -> 256-bit phash (DCT-based, good at
# detecting global brightness/color/content changes) and a 16x15 = 240-bit
# dhash (gradient-based, much more sensitive to small local shifts — a slow
# pan or a slide-transition that phash smears across its low-frequency
# coefficients will still flip a chunk of dhash bits). We compare the
# current frame against the last kept frame with *both* hashes and keep
# the frame if *either* distance exceeds the threshold. This is
# deliberately conservative: over-keeping is cheap at the OCR step but
# cheap-to-wrong at the video exporter, which assumes each kept frame's
# boxes cover the whole window up to the next kept frame.
_HASH_SIZE = 16
_HASH_BITS = _HASH_SIZE * _HASH_SIZE
# Hamming distance threshold applied to both phash and dhash. At 2 on a
# ~250-bit hash that's <1% bit difference — only near-exact duplicates are
# collapsed, and even subtle motion (cursor nudge, slow scroll, fade)
# crosses the bar and keeps the frame.
_DEFAULT_DEDUP_THRESHOLD = 2
# Upper bound on the run of consecutive raw frames we'll skip between
# kept frames. Even when the perceptual hashes stay within the threshold,
# we force-keep a frame once `_DEFAULT_MAX_GAP` raw frames in a row have
# been dropped. This caps the wall-clock distance between adjacent kept
# frames at roughly `(max_gap + 1) / source_fps` seconds, which is what
# the UI tween and the gap-filler insertion logic rely on to bound their
# worst-case motion/resize per pair. At ``max_gap = 1`` — the default —
# kept frames are never more than 2 raw-frame steps apart: at 30 fps
# source that's ~66 ms, tight enough that a linear tween between adjacent
# kept frames cannot visibly under-cover moving or resizing text. Set to
# 0 to disable deduplication entirely (every decoded frame is kept).
_DEFAULT_MAX_GAP = 1


class FrameExtractionError(Exception):
    """Raised when ffmpeg fails or produces no frames."""


def _run_ffmpeg(
    video_path: Path,
    out_dir: Path,
    fps: float | None,
    max_width: int,
) -> None:
    # When fps is None we omit the fps filter so ffmpeg emits every decoded frame.
    filters = [f"scale='min({max_width},iw)':-2"]
    if fps is not None and fps > 0:
        filters.insert(0, f"fps={fps}")
    vf = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-q:v",
        "3",
        str(out_dir / "frame_%06d.jpg"),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or "unknown error"
        raise FrameExtractionError(f"ffmpeg failed: {err}")


def _dedupe_paths(
    frame_paths: list[Path],
    *,
    threshold: int,
    max_gap: int,
) -> list[tuple[int, Path]]:
    """Drop near-duplicate frames using sequential dual-hash comparison,
    with a hard upper bound on the run of consecutive raw frames we will
    skip between kept frames.

    For each frame we compute both a phash (DCT, low-frequency content
    change) and a dhash (gradient, local-motion shift), then keep the
    frame if *either* distance to the last kept frame exceeds `threshold`
    OR the number of raw frames already skipped since the last kept
    frame exceeds `max_gap`. Using both hashes catches two failure modes
    of either alone: phash is insensitive to small positional shifts
    because the DCT averages local motion across all coefficients; dhash
    is insensitive to large, uniform colour/brightness changes because
    it only looks at adjacent-pixel deltas. The ``max_gap`` cap catches a
    third failure mode both hashes share — slow continuous motion (e.g.
    a smoothly-scrolling list) where each frame is within threshold of
    its predecessor yet the cumulative drift across many frames is
    large. Without the cap, kept frames would be spaced arbitrarily far
    apart and the tween between them would visibly under-cover moving
    text.

    Returns `(source_index, path)` tuples where `source_index` is 0-based
    into the original ffmpeg-emitted sequence. Callers use those indices
    to map kept frames back to timestamps in the source video.
    """
    kept: list[tuple[int, Path]] = []
    last_phash: imagehash.ImageHash | None = None
    last_dhash: imagehash.ImageHash | None = None
    last_kept_i = -1
    for i, p in enumerate(frame_paths):
        with Image.open(p) as im:
            im = im.convert("RGB")
            ph = imagehash.phash(im, hash_size=_HASH_SIZE)
            dh = imagehash.dhash(im, hash_size=_HASH_SIZE)
        if last_phash is None or last_dhash is None:
            kept.append((i, p))
            last_phash, last_dhash, last_kept_i = ph, dh, i
            continue
        gap_exceeded = (i - last_kept_i) > max_gap + 1
        hash_moved = (
            (ph - last_phash) > threshold or (dh - last_dhash) > threshold
        )
        if hash_moved or gap_exceeded:
            kept.append((i, p))
            last_phash, last_dhash, last_kept_i = ph, dh, i
    return kept


def _probe_source_fps(video_path: Path) -> float | None:
    """Return the source stream's average frame rate in fps, or None if
    ffprobe is unavailable or the stream's metadata can't be parsed.

    Prefers `avg_frame_rate` (steady timebase) over `r_frame_rate` (raw
    container rate) because ffmpeg's frame extractor emits frames at the
    average rate by default.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return None
        s = streams[0]
    except (ValueError, KeyError):
        return None

    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = s.get(key)
        if not rate or "/" not in rate:
            continue
        num_s, den_s = rate.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:
            continue
        if den > 0 and num > 0:
            return num / den
    return None


def extract_deduplicated_frames(
    video_bytes: bytes,
    *,
    suffix: str,
    fps: float | None = None,
    dedup_threshold: int = _DEFAULT_DEDUP_THRESHOLD,
    max_gap: int = _DEFAULT_MAX_GAP,
    max_width: int = 1600,
) -> tuple[list[bytes], int, int, tuple[int, ...], float | None]:
    """
    Returns `(jpeg_bytes_list, raw_frame_count, kept_count,
    kept_source_indices, source_fps)`.

    `kept_source_indices[k]` is the 0-based index, into the original ffmpeg
    output sequence, of the k-th kept (i.e. deduplicated) frame. Paired
    with `source_fps`, callers can compute the source timestamp of each
    kept frame: `t = kept_source_indices[k] / source_fps`.

    `source_fps` is the effective frame rate of the extraction. When the
    caller passed an explicit `fps`, it's that value. When `fps is None`
    we probe the source via ffprobe; if probing fails we return `None`
    and the caller must treat timestamp-based export as unavailable.

    When `fps` is None (default), every decoded frame is extracted and
    perceptual-hash deduplication decides what to keep.
    """
    if fps is not None:
        fps = float(fps)
        if fps <= 0:
            fps = None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / f"input{suffix}"
        video_path.write_bytes(video_bytes)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _run_ffmpeg(video_path, frames_dir, fps=fps, max_width=max_width)

        pattern = str(frames_dir / "frame_*.jpg")
        all_frames = sorted(Path(p) for p in glob.glob(pattern))
        raw_count = len(all_frames)
        if raw_count == 0:
            raise FrameExtractionError("No frames extracted; check format and codec.")

        source_fps = fps if fps is not None else _probe_source_fps(video_path)

        kept_entries = _dedupe_paths(
            all_frames, threshold=dedup_threshold, max_gap=max_gap,
        )
        blobs = [Path(p).read_bytes() for _, p in kept_entries]
        kept_source_indices = tuple(i for i, _ in kept_entries)
        return blobs, raw_count, len(kept_entries), kept_source_indices, source_fps


def extract_frames_by_source_index(
    video_bytes: bytes,
    *,
    suffix: str,
    source_indices: list[int],
    fps: float | None = None,
    max_width: int = 1600,
) -> list[tuple[int, bytes]]:
    """Re-run ffmpeg for a single video and return JPEG bytes of only the
    specified source-frame indices.

    ``source_indices`` are 0-based indices into the original ffmpeg-
    emitted sequence — exactly the values that live in
    ``FrameSet.kept_source_indices``. The gap-filler uses this to pull
    raw frames that the initial dedup pass discarded but that now need
    detection because their position between two kept frames exceeds the
    smoothness-invariant thresholds.

    To keep behaviour consistent with the main extraction pipeline, we
    apply the same ``fps`` filter and ``max_width`` scale filter — so
    indices resolve to the same source-video frames that
    ``extract_deduplicated_frames`` would have seen. We then select the
    requested indices via an ffmpeg ``select='eq(n,i0)+eq(n,i1)+...'``
    filter so only those frames are encoded, ordered by ascending index.

    Returns a list of ``(source_index, jpeg_bytes)`` pairs in the input
    order of ``source_indices``. Missing indices (past end of stream) are
    silently omitted.
    """
    if fps is not None:
        fps = float(fps)
        if fps <= 0:
            fps = None
    uniq = sorted({int(i) for i in source_indices if int(i) >= 0})
    if not uniq:
        return []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / f"input{suffix}"
        video_path.write_bytes(video_bytes)
        out_dir = tmp_path / "frames"
        out_dir.mkdir()

        # Build a select expression matching just the requested raw
        # indices. ffmpeg's `select` takes `eq(n,i)` per desired frame,
        # OR-combined with `+`. `vsync=vfr` is important: without it,
        # ffmpeg would pad the missing frames with duplicates of the
        # previous kept frame and our output index order would diverge
        # from `uniq`.
        select_expr = "+".join(f"eq(n\\,{i})" for i in uniq)
        filters: list[str] = []
        if fps is not None and fps > 0:
            filters.append(f"fps={fps}")
        filters.append(f"scale='min({max_width},iw)':-2")
        filters.append(f"select='{select_expr}'")
        vf = ",".join(filters)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-vsync",
            "vfr",
            "-q:v",
            "3",
            str(out_dir / "frame_%06d.jpg"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or "unknown error"
            raise FrameExtractionError(f"ffmpeg select failed: {err}")

        emitted = sorted(Path(p) for p in glob.glob(str(out_dir / "frame_*.jpg")))
        # ``emitted`` is ordered by ffmpeg's encode sequence, which (with
        # `vsync=vfr`) matches ascending source index — i.e. `uniq` order.
        # Any trailing indices past end-of-stream simply don't get a file.
        by_uniq_index: dict[int, bytes] = {}
        for i_sorted, p in zip(uniq, emitted):
            by_uniq_index[i_sorted] = p.read_bytes()

        return [(i, by_uniq_index[i]) for i in uniq if i in by_uniq_index]

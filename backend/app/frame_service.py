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
) -> list[tuple[int, Path]]:
    """Drop near-duplicate frames using sequential dual-hash comparison.

    For each frame we compute both a phash (DCT, low-frequency content
    change) and a dhash (gradient, local-motion shift), then keep the
    frame if *either* distance to the last kept frame exceeds `threshold`.
    Using both hashes catches two failure modes of either alone: phash is
    insensitive to small positional shifts because the DCT averages local
    motion across all coefficients; dhash is insensitive to large,
    uniform colour/brightness changes because it only looks at adjacent-
    pixel deltas.

    Returns `(source_index, path)` tuples where `source_index` is 0-based
    into the original ffmpeg-emitted sequence. Callers use those indices
    to map kept frames back to timestamps in the source video.
    """
    kept: list[tuple[int, Path]] = []
    last_phash: imagehash.ImageHash | None = None
    last_dhash: imagehash.ImageHash | None = None
    for i, p in enumerate(frame_paths):
        with Image.open(p) as im:
            im = im.convert("RGB")
            ph = imagehash.phash(im, hash_size=_HASH_SIZE)
            dh = imagehash.dhash(im, hash_size=_HASH_SIZE)
        if last_phash is None or last_dhash is None:
            kept.append((i, p))
            last_phash, last_dhash = ph, dh
            continue
        if (ph - last_phash) > threshold or (dh - last_dhash) > threshold:
            kept.append((i, p))
            last_phash, last_dhash = ph, dh
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

        kept_entries = _dedupe_paths(all_frames, threshold=dedup_threshold)
        blobs = [Path(p).read_bytes() for _, p in kept_entries]
        kept_source_indices = tuple(i for i, _ in kept_entries)
        return blobs, raw_count, len(kept_entries), kept_source_indices, source_fps

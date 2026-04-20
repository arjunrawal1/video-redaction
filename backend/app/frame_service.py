"""Extract frames from video with ffmpeg and deduplicate via perceptual hashing."""

from __future__ import annotations

import glob
import subprocess
import tempfile
from pathlib import Path

import imagehash
from PIL import Image

# Perceptual hash resolution. 16x16 -> 256-bit hash; finer-grained than the
# default 8x8 (64-bit) so small changes (e.g. a few-pixel cursor move) flip
# enough bits to register as a new frame.
_HASH_SIZE = 16
_HASH_BITS = _HASH_SIZE * _HASH_SIZE
# Hamming distance threshold on the 256-bit hash. Only near-exact duplicates
# are dropped; any visible motion (including cursor movement) should exceed
# this and be kept.
_DEFAULT_DEDUP_THRESHOLD = 4


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
) -> list[Path]:
    """Drop near-duplicate frames using sequential perceptual hash comparison."""
    kept: list[Path] = []
    last_hash: imagehash.ImageHash | None = None
    for p in frame_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            h = imagehash.phash(im, hash_size=_HASH_SIZE)
        if last_hash is None:
            kept.append(p)
            last_hash = h
            continue
        if h - last_hash > threshold:
            kept.append(p)
            last_hash = h
    return kept


def extract_deduplicated_frames(
    video_bytes: bytes,
    *,
    suffix: str,
    fps: float | None = None,
    dedup_threshold: int = _DEFAULT_DEDUP_THRESHOLD,
    max_width: int = 1600,
) -> tuple[list[bytes], int, int]:
    """
    Returns (jpeg_bytes_list, raw_frame_count, kept_count).

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

        kept_paths = _dedupe_paths(all_frames, threshold=dedup_threshold)
        blobs = [Path(p).read_bytes() for p in kept_paths]
        return blobs, raw_count, len(kept_paths)

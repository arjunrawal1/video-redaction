"""Render a redacted MP4 by painting detection boxes onto the source video.

The detection pipeline produces boxes keyed to *deduplicated* frame indices,
each with coords in a scaled-down reference frame. The exporter:

1. Resolves each kept dedup frame `k` to a source-video time window
   `[t_k, t_{k+1})`, using the kept-source-index mapping and the extraction
   frame rate.
2. Scales every box from the reference frame's pixel space up to the source
   video's native resolution.
3. Emits a single ffmpeg `drawbox` filter chain — one filled rectangle per
   (frame, box) pair, gated on the right `enable='between(t,...)'` window.
4. Runs ffmpeg once, copying the audio stream if present.

The filter graph is written to a temp file and passed via
`-filter_script:v` so large detection sets don't blow the shell's argv
length limit.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class VideoExportError(Exception):
    """Raised when ffmpeg/ffprobe fail or inputs are inconsistent."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float


@dataclass(frozen=True)
class ExportBox:
    """A single detection box for export, in reference-frame pixel coords."""

    frame: int  # 1-based kept-frame index
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class ExportStyle:
    color: str = "black"
    padding_px: int = 4


def probe_video_info(video_path: Path) -> VideoInfo:
    """Return source width/height/duration via ffprobe. Raises on failure."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise VideoExportError("ffprobe not found on PATH") from e
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or "unknown error"
        raise VideoExportError(f"ffprobe failed: {err}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except ValueError as e:
        raise VideoExportError(f"ffprobe JSON parse error: {e}") from e

    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    duration_str = stream.get("duration") or fmt.get("duration")
    try:
        duration = float(duration_str) if duration_str else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    if width <= 0 or height <= 0:
        raise VideoExportError(
            f"ffprobe returned invalid dimensions ({width}x{height})",
        )
    if duration <= 0:
        raise VideoExportError("ffprobe could not determine video duration")
    return VideoInfo(width=width, height=height, duration=duration)


def _frame_window(
    *,
    kept_source_indices: tuple[int, ...],
    source_fps: float,
    duration: float,
    kept_index_1: int,
) -> tuple[float, float]:
    """Time window `[t_start, t_end)` during which dedup frame `kept_index_1`
    represents the source video.

    `kept_index_1` is 1-based; the first kept frame covers from t=0 (its
    source timestamp is effectively the start of the visible window, even
    if ffmpeg skipped a few decode frames before it). The last kept frame
    covers through end-of-video.
    """
    i = kept_index_1 - 1
    if i < 0 or i >= len(kept_source_indices):
        raise VideoExportError(
            f"kept frame index out of range: {kept_index_1}"
        )
    t_start = 0.0 if i == 0 else kept_source_indices[i] / source_fps
    if i + 1 < len(kept_source_indices):
        t_end = kept_source_indices[i + 1] / source_fps
    else:
        t_end = duration
    if t_end <= t_start:
        # Degenerate (shouldn't happen) — pin a tiny positive window so the
        # filter still parses rather than throwing on zero-length enable.
        t_end = t_start + 1.0 / max(source_fps, 1.0)
    return t_start, t_end


def _format_time(t: float) -> str:
    # 3 decimals is sub-millisecond; plenty for frame-time alignment. Using
    # `f` formatting avoids scientific notation which ffmpeg's expression
    # parser rejects.
    return f"{max(0.0, t):.3f}"


def build_drawbox_filter(
    *,
    boxes: list[ExportBox],
    kept_source_indices: tuple[int, ...],
    source_fps: float,
    duration: float,
    ref_width: int,
    ref_height: int,
    src_width: int,
    src_height: int,
    style: ExportStyle,
) -> str:
    """Build a comma-separated drawbox chain, one filled rectangle per box.

    Scales box coordinates from the reference frame (dedup-frame pixel
    space, where `ref_width` matches ffmpeg's `max_width` cap) up to the
    source resolution. All boxes are clamped to the source frame, widened
    by `padding_px` on every side, then gated on their kept frame's time
    window.
    """
    if ref_width <= 0 or ref_height <= 0:
        raise VideoExportError(
            f"invalid reference dims: {ref_width}x{ref_height}",
        )
    scale_x = src_width / ref_width
    scale_y = src_height / ref_height
    pad = max(0, int(style.padding_px))
    # Sanitize color: drawbox accepts names (`black`), `#rrggbb`, and
    # `black@0.8` alpha suffixes. We forbid colons and commas so they
    # can't break the filter chain syntax.
    color = style.color.replace(":", "").replace(",", "") or "black"

    parts: list[str] = []
    for b in boxes:
        try:
            t0, t1 = _frame_window(
                kept_source_indices=kept_source_indices,
                source_fps=source_fps,
                duration=duration,
                kept_index_1=b.frame,
            )
        except VideoExportError:
            continue

        x = b.x * scale_x - pad
        y = b.y * scale_y - pad
        w = b.w * scale_x + 2 * pad
        h = b.h * scale_y + 2 * pad

        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(src_width, int(round(x + w)))
        y1 = min(src_height, int(round(y + h)))
        w0 = x1 - x0
        h0 = y1 - y0
        if w0 <= 0 or h0 <= 0:
            continue

        parts.append(
            "drawbox="
            f"x={x0}:y={y0}:w={w0}:h={h0}"
            f":color={color}:t=fill"
            f":enable='between(t,{_format_time(t0)},{_format_time(t1)})'"
        )

    if not parts:
        # No-op filter so ffmpeg still produces valid output even when the
        # caller submitted zero usable boxes.
        return "null"
    return ",".join(parts)


def render_redacted_video(
    *,
    source_path: Path,
    out_path: Path,
    filter_graph: str,
) -> None:
    """Run ffmpeg with the given filter graph, re-encoding video and copying
    audio. Raises `VideoExportError` on failure.

    The filter graph is written to a temp file and passed via
    `-filter_script:v` so huge detection sets don't blow the argv limit.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(filter_graph)
        script_path = Path(f.name)

    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-filter_script:v",
            str(script_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            # Copy audio if present; ffmpeg silently skips when absent.
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False,
            )
        except FileNotFoundError as e:
            raise VideoExportError("ffmpeg not found on PATH") from e
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or "unknown"
            raise VideoExportError(f"ffmpeg failed: {err}")
    finally:
        try:
            script_path.unlink()
        except OSError:
            pass

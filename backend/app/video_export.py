"""Render a redacted MP4 by painting detection boxes onto the source video.

The detection pipeline produces boxes keyed to *deduplicated* frame indices,
each with coords in a scaled-down reference frame. The exporter:

1. Resolves each kept dedup frame ``k`` to a source-video time window
   ``[t_k, t_{k+1})``, using the kept-source-index mapping and the
   extraction frame rate.
2. Scales every box from the reference frame's pixel space up to the
   source video's native resolution.
3. Emits one ffmpeg ``drawbox`` per keyframe box. If the box has a
   ``track_id`` and the NEXT kept frame has a box with the SAME
   ``track_id``, the drawbox uses ``eval=frame`` with linear-interpolation
   expressions for x/y/w/h — tweening the redaction smoothly from
   its position at ``t_k`` to the successor's position at ``t_{k+1}``.
   Boxes without a tracked successor render statically over their
   keyframe window (legacy pre-linker behavior).
4. Runs ffmpeg once, copying the audio stream if present.

The filter graph is written to a temp file and passed via
``-filter_script:v`` so large detection sets don't blow the shell's argv
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
    """A single detection box for export, in reference-frame pixel coords.

    ``track_id`` is an optional cross-frame identity produced by the
    teamwork phase-1.5 linker. Two boxes in consecutive kept frames with
    the same ``track_id`` are the same real-world redaction, and the
    exporter will interpolate between them. ``None`` falls back to
    static-per-keyframe rendering.
    """

    frame: int  # 1-based kept-frame index
    x: float
    y: float
    w: float
    h: float
    track_id: str | None = None


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
    space, where ``ref_width`` matches ffmpeg's ``max_width`` cap) up to
    the source resolution. Boxes are widened by ``padding_px`` on every
    side and gated on their kept frame's time window.

    Tracked boxes (those with ``track_id`` AND a successor at the next
    kept frame sharing the same id) emit an *interpolating* drawbox:
    ``eval=frame`` is set and x/y/w/h are linear expressions of ``t``
    that slide from the start box to the end box across the window.
    Untracked boxes — and tracked boxes at the end of a chain — render
    statically, matching the legacy pre-linker behavior.
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

    # Index tracked boxes for O(1) "does my successor exist?" lookups.
    # By construction the linker assigns unique track_ids per frame, so
    # (frame, track_id) is a unique key; if a client bug sends duplicates,
    # last-wins is harmless — the resulting interp just uses the later
    # box as the endpoint.
    by_fk: dict[tuple[int, str], ExportBox] = {}
    for b in boxes:
        if b.track_id:
            by_fk[(b.frame, b.track_id)] = b

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

        xa = b.x * scale_x - pad
        ya = b.y * scale_y - pad
        wa = b.w * scale_x + 2 * pad
        ha = b.h * scale_y + 2 * pad

        successor = (
            by_fk.get((b.frame + 1, b.track_id)) if b.track_id else None
        )

        if successor is not None and t1 > t0:
            xb = successor.x * scale_x - pad
            yb = successor.y * scale_y - pad
            wb = successor.w * scale_x + 2 * pad
            hb = successor.h * scale_y + 2 * pad
            parts.extend(
                _tween_drawbox_parts(
                    xa=xa,
                    ya=ya,
                    wa=wa,
                    ha=ha,
                    xb=xb,
                    yb=yb,
                    wb=wb,
                    hb=hb,
                    t0=t0,
                    t1=t1,
                    kept_source_indices=kept_source_indices,
                    kept_index_1=b.frame,
                    source_fps=source_fps,
                    src_width=src_width,
                    src_height=src_height,
                    color=color,
                )
            )
            continue

        # Static path: either untracked, no successor, or degenerate
        # interp window. Clamp to source bounds and skip zero-area.
        x0 = max(0, int(round(xa)))
        y0 = max(0, int(round(ya)))
        x1i = min(src_width, int(round(xa + wa)))
        y1i = min(src_height, int(round(ya + ha)))
        w0 = x1i - x0
        h0 = y1i - y0
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


def _tween_drawbox_parts(
    *,
    xa: float,
    ya: float,
    wa: float,
    ha: float,
    xb: float,
    yb: float,
    wb: float,
    hb: float,
    t0: float,
    t1: float,
    kept_source_indices: tuple[int, ...],
    kept_index_1: int,
    source_fps: float,
    src_width: int,
    src_height: int,
    color: str,
) -> list[str]:
    """Emit one static drawbox per source frame across the tween window,
    each with linearly-interpolated coords.

    ffmpeg's ``drawbox`` filter does NOT support per-frame expression
    evaluation for x/y/w/h — the ``t`` variable available to drawbox
    expressions is THICKNESS, not timestamp, and the legacy
    ``eval=frame`` option was removed in ffmpeg 8.0. The only portable
    way to animate drawbox position is to emit many drawbox instances,
    each gated on a narrow ``enable='between(t, ...)'`` window (which
    IS evaluated per frame by ffmpeg's timeline engine).

    We subdivide the window into exactly one step per source frame
    that falls inside it. That's the smoothest motion drawbox can
    produce — no sub-frame gain is possible because the video itself
    is discrete. Continuity rule: the LAST step sits at ``u=1`` so
    that at the boundary ``t_{k+1}`` the position is exactly the
    successor's box, matching the start of the next kept-frame's
    window (static or tweening to k+2, both of which start at
    ``pos_{k+1}`` too).
    """
    # Source-frame range for this kept-frame window: [src_start, src_end).
    # kept_index_1 is 1-based. Guarded by callers, but stay defensive.
    idx = kept_index_1 - 1
    if idx < 0 or idx >= len(kept_source_indices):
        return []
    src_start = kept_source_indices[idx]
    if idx + 1 < len(kept_source_indices):
        src_end = kept_source_indices[idx + 1]
    else:
        src_end = src_start + 1
    steps = max(1, src_end - src_start)

    # Continuity semantic: step s=0 → u=0 (start pos), step s=steps-1 → u=1
    # (end pos). Makes the last sub-window of this tween end at pos_b,
    # which is also where the NEXT kept-frame's window begins. No jump at
    # the boundary. With steps==1 there's nothing to interpolate — emit a
    # single drawbox at pos_a (equivalent to the static path).
    denom = max(1, steps - 1)

    parts: list[str] = []
    for s in range(steps):
        u = s / denom if steps > 1 else 0.0
        xi = xa + (xb - xa) * u
        yi = ya + (yb - ya) * u
        wi = wa + (wb - wa) * u
        hi = ha + (hb - ha) * u

        x0 = max(0, int(round(xi)))
        y0 = max(0, int(round(yi)))
        x1i = min(src_width, int(round(xi + wi)))
        y1i = min(src_height, int(round(yi + hi)))
        w0 = x1i - x0
        h0 = y1i - y0
        if w0 <= 0 or h0 <= 0:
            continue

        # Sub-window timing in the source timeline. For a window of
        # `steps` source frames, sub-window s covers exactly one frame's
        # worth of time. First sub-window inherits t0 (which accounts
        # for the kept-index-0 special case of "starts at t=0"); last
        # sub-window runs through t1 so no gap is left at the boundary.
        if s == 0:
            ti_start = t0
        else:
            ti_start = (src_start + s) / source_fps
        if s == steps - 1:
            ti_end = t1
        else:
            ti_end = (src_start + s + 1) / source_fps
        if ti_end <= ti_start:
            continue

        # Half-open [ti_start, ti_end) via gte*lt rather than `between`
        # (which is inclusive on both ends). Every source frame sits at
        # exactly some multiple of 1/fps, and with closed intervals two
        # adjacent sub-windows both claim that boundary frame — resulting
        # in two drawboxes firing at slightly different positions and a
        # visible double-paint. gte*lt makes exactly one sub-window own
        # each source frame. We keep the final sub-window's end at t1 so
        # the next kept-frame's window (static or the next tween) picks
        # up cleanly at its own ti_start=t1.
        parts.append(
            "drawbox="
            f"x={x0}:y={y0}:w={w0}:h={h0}"
            f":color={color}:t=fill"
            f":enable='gte(t,{_format_time(ti_start)})"
            f"*lt(t,{_format_time(ti_end)})'"
        )
    return parts


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

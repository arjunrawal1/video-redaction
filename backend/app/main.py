import asyncio
import base64
import io
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from starlette.background import BackgroundTask

from app.frame_cache import get_or_extract
from app.frame_service import FrameExtractionError, extract_frames_by_source_index
from app.logging_config import configure as configure_logging
from app import ocr_backtrack, ocr_cache, video_export
from app.ocr_service import (
    detect_frame,
    detect_frame_raw,
    ensure_reader_loaded,
    normalize_query,
    warm as warm_ocr,
)

configure_logging()
log = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = 500 * 1024 * 1024
# Cap on concurrent per-frame OCR tasks for a single streaming request.
# With Textract this is an I/O-bound HTTP call; you can safely raise it up to
# the account's TPS quota (DetectDocumentText defaults to ~10 TPS).
_STREAM_OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("lifespan: warming OCR reader...")
    t0 = time.perf_counter()
    warm_ocr()
    log.info("lifespan: warmup finished in %.2fs", time.perf_counter() - t0)
    yield


app = FastAPI(title="Video Redaction API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Video Redaction API"}


def _guess_suffix(filename: str | None) -> str:
    if not filename:
        return ".mp4"
    suf = Path(filename).suffix.lower()
    if suf and len(suf) <= 8:
        return suf
    return ".mp4"


def _validate_video(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/")
        or content_type in ("application/octet-stream", "binary/octet-stream")
    ):
        raise HTTPException(
            status_code=415,
            detail="Expected a video Content-Type (video/*) or octet-stream.",
        )


@app.post("/api/frames/deduplicated")
async def deduplicated_frames(
    file: UploadFile = File(...),
    fps: float | None = Query(
        None,
        gt=0,
        description=(
            "Optional sampling rate. Omit to extract every decoded frame "
            "and let perceptual-hash dedup handle the rest."
        ),
    ),
    dedup_threshold: int = Query(
        2,
        ge=0,
        le=256,
        description=(
            "Hamming threshold applied to both a 256-bit phash and a "
            "240-bit dhash (lower = stricter; frame is kept when either "
            "hash distance exceeds the threshold)."
        ),
    ),
    max_gap: int = Query(
        1,
        ge=0,
        le=240,
        description=(
            "Upper bound on consecutive raw frames skipped between kept "
            "frames. Forces a keep once the run of skipped frames "
            "exceeds this value, even when both hashes stay under "
            "dedup_threshold. Default 1 (kept frames at most ~2 raw-"
            "frame steps apart, i.e. ~66 ms at 30 fps) to bound the "
            "motion/resize a tween has to cover between kept frames. "
            "Use 0 to keep every decoded frame."
        ),
    ),
) -> dict:
    """
    Accept a video file, extract frames (every decoded frame by default, or at
    `fps` if provided), drop near-duplicates (sequential phash), return JPEG
    thumbnails as base64.
    """
    _validate_video(file)
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    suffix = _guess_suffix(file.filename)
    try:
        frame_set, video_hash = get_or_extract(
            body,
            suffix=suffix,
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
        )
    except FrameExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    mime = "image/jpeg"
    frames_out = [
        {
            "mime": mime,
            "data_base64": base64.b64encode(b).decode("ascii"),
        }
        for b in frame_set.frames
    ]

    # All dedup frames share dimensions (ffmpeg's scale filter caps width
    # uniformly, with `-2` preserving aspect). Inspect the first kept
    # frame so the client has a reference coordinate space for exports.
    frame_width = 0
    frame_height = 0
    if frame_set.frames:
        try:
            with Image.open(io.BytesIO(frame_set.frames[0])) as im:
                frame_width, frame_height = im.size
        except Exception:  # pragma: no cover
            log.exception("failed to read dedup frame dimensions")

    return {
        "filename": file.filename,
        "video_hash": video_hash,
        "fps": fps,
        "dedup_threshold": dedup_threshold,
        "max_gap": max_gap,
        "source_fps": frame_set.source_fps,
        "raw_frame_count": frame_set.raw_count,
        "deduplicated_count": frame_set.kept_count,
        # 0-based index into the original ffmpeg-emitted sequence for
        # each kept frame. The Next.js gap-filler uses this to bisect
        # kept pairs in source-index space and request specific raw
        # frames from /api/frames/by_source_index.
        "kept_source_indices": list(frame_set.kept_source_indices),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frames": frames_out,
    }


@app.post("/api/frames/by_source_index")
async def frames_by_source_index(
    file: UploadFile = File(...),
    source_indices: str = Form(
        ...,
        description=(
            "JSON array of 0-based indices into the original ffmpeg-"
            "emitted sequence, e.g. `[12, 47, 133]`. These are the same "
            "values that appear in `kept_source_indices` from the "
            "/api/frames/deduplicated response — the gap-filler uses "
            "them to request raw frames that initial dedup discarded."
        ),
    ),
    fps: float | None = Form(
        None,
        description=(
            "Must match the `fps` used when the caller originally "
            "fetched /api/frames/deduplicated. The indices are "
            "interpreted against ffmpeg's decoded-frame stream after "
            "applying the same `fps` filter, so a mismatch would "
            "resolve to different source frames."
        ),
    ),
) -> dict:
    """Return JPEG bytes for specific raw source frames.

    Used by the Next.js gap-filler (``lib/server/gap-filler.ts``) when
    motion or resize between two adjacent kept frames exceeds the
    smoothness-invariant thresholds. The filler wants to insert
    detection results on a frame that dedup dropped; we re-run ffmpeg
    with a ``select=eq(n,i)+…`` filter to pull those exact frames
    without reprocessing the whole video.

    This endpoint does NOT touch the frame cache — it is a per-call
    re-extraction keyed by request. Results are small (a handful of
    frames per call) and the caller is responsible for deduping repeat
    requests.
    """
    _validate_video(file)
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        indices_raw = json.loads(source_indices)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"source_indices must be a JSON array: {e}",
        ) from e
    if not isinstance(indices_raw, list):
        raise HTTPException(
            status_code=400, detail="source_indices must be a JSON array.",
        )
    try:
        indices: list[int] = [int(i) for i in indices_raw]
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"source_indices must be integers: {e}",
        ) from e
    # Cap request size so a misbehaving caller can't ask for half the
    # video one frame at a time. At the default `max_gap=1` + bisection
    # depth 5, the worst-case ask is ~31 frames per original gap across
    # hundreds of gaps, which still fits under this cap per request.
    if len(indices) > 2048:
        raise HTTPException(
            status_code=400, detail="Too many source_indices (max 2048).",
        )

    if fps is not None and fps <= 0:
        fps = None

    suffix = _guess_suffix(file.filename)
    try:
        results = extract_frames_by_source_index(
            body,
            suffix=suffix,
            source_indices=indices,
            fps=fps,
        )
    except FrameExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    mime = "image/jpeg"
    frames_out = []
    dims_cache: tuple[int, int] | None = None
    for src_index, b in results:
        if dims_cache is None:
            try:
                with Image.open(io.BytesIO(b)) as im:
                    dims_cache = im.size
            except Exception:  # pragma: no cover
                dims_cache = (0, 0)
        frames_out.append(
            {
                "source_index": src_index,
                "mime": mime,
                "data_base64": base64.b64encode(b).decode("ascii"),
            }
        )
    frame_width, frame_height = dims_cache or (0, 0)

    return {
        "filename": file.filename,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frames": frames_out,
    }


def _ndjson(payload: dict) -> bytes:
    return (
        json.dumps(payload, separators=(",", ":"), default=str) + "\n"
    ).encode("utf-8")


@app.post("/api/ocr/detect/stream")
async def detect_stream(
    file: UploadFile = File(...),
    query: str = Form(..., min_length=1, max_length=200),
    frame_from: int = Form(1, ge=1),
    frame_to: int | None = Form(None),
    fps: float | None = Form(None),
    dedup_threshold: int = Form(2),
    max_gap: int = Form(1, ge=0, le=240),
) -> StreamingResponse:
    """Stream OCR detections across a frame range as NDJSON.

    Frame indices are 1-based and inclusive on both ends, matching the UI
    badges. Events:
      {"type":"start", "total": N, "frame_indices":[...], "deduplicated_count": M, ...}
      {"type":"frame", "index": i, "width":..., "height":..., "boxes":[...]}
      {"type":"done", "matched_frames": k, "total_boxes": b}
      {"type":"error", "message": "..."}
    """
    _validate_video(file)
    t_req = time.perf_counter()
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    if fps is not None and fps <= 0:
        fps = None

    log.info(
        "stream detect: recv file=%r bytes=%d query=%r range=%s..%s "
        "fps=%s dedup_threshold=%d max_gap=%d",
        file.filename,
        len(body),
        query,
        frame_from,
        frame_to,
        fps,
        dedup_threshold,
        max_gap,
    )

    suffix = _guess_suffix(file.filename)
    t_extract = time.perf_counter()
    try:
        frame_set, video_hash = get_or_extract(
            body,
            suffix=suffix,
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
        )
    except FrameExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    extract_ms = (time.perf_counter() - t_extract) * 1000

    n = frame_set.kept_count
    log.info(
        "stream detect: video_hash=%s frames=%d (raw=%d) extract/cache=%.0fms",
        video_hash[:12],
        n,
        frame_set.raw_count,
        extract_ms,
    )
    if n == 0:
        raise HTTPException(status_code=422, detail="No frames to process.")

    lo_1 = max(1, frame_from)
    hi_1 = frame_to if frame_to is not None else n
    hi_1 = min(max(1, hi_1), n)
    if lo_1 > hi_1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frame range: from={frame_from}, to={frame_to}",
        )

    selected_indices_1 = list(range(lo_1, hi_1 + 1))
    selected = [(i_1, frame_set.frames[i_1 - 1]) for i_1 in selected_indices_1]
    q_norm = normalize_query(query)

    async def stream() -> AsyncIterator[bytes]:
        yield _ndjson(
            {
                "type": "start",
                "video_hash": video_hash,
                "query": query,
                "deduplicated_count": n,
                "frame_from": lo_1,
                "frame_to": hi_1,
                "frame_indices": selected_indices_1,
                "total": len(selected),
            }
        )

        if not q_norm:
            yield _ndjson(
                {"type": "done", "matched_frames": 0, "total_boxes": 0}
            )
            return

        try:
            t_load = time.perf_counter()
            await asyncio.to_thread(ensure_reader_loaded)
            log.info(
                "stream detect: reader ready in %.0fms",
                (time.perf_counter() - t_load) * 1000,
            )
        except Exception as e:  # pragma: no cover
            log.exception("OCR reader load failed")
            yield _ndjson({"type": "error", "message": f"OCR init failed: {e}"})
            return

        sem = asyncio.Semaphore(_STREAM_OCR_CONCURRENCY)
        log.info(
            "stream detect: fanning out %d frames (range %d..%d) "
            "concurrency=%d query_norm=%r",
            len(selected),
            lo_1,
            hi_1,
            _STREAM_OCR_CONCURRENCY,
            q_norm,
        )

        # Create a fresh OCR cache entry; each completed frame populates it so
        # the /api/ocr/backtrack endpoint can walk the same data in reverse
        # without re-calling Textract.
        cache_entry = ocr_cache.create_entry(
            video_hash=video_hash,
            query_norm=q_norm,
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
            frame_from=lo_1,
            frame_to=hi_1,
        )

        loop_t0 = time.perf_counter()
        per_frame_ms: list[float] = []

        async def run_one(idx_1: int, blob: bytes):
            t_wait = time.perf_counter()
            async with sem:
                wait_ms = (time.perf_counter() - t_wait) * 1000
                t_work = time.perf_counter()
                det, raw = await asyncio.to_thread(detect_frame, blob, q_norm)
                work_ms = (time.perf_counter() - t_work) * 1000
                per_frame_ms.append(work_ms)
                ocr_cache.put_frame(
                    cache_entry,
                    frame_idx_1=idx_1,
                    width=det.width,
                    height=det.height,
                    matched=list(det.boxes),
                    raw=raw,
                )
                log.info(
                    "frame #%d: wait=%.0fms work=%.0fms matches=%d",
                    idx_1,
                    wait_ms,
                    work_ms,
                    len(det.boxes),
                )
                return idx_1, det, raw

        tasks = [asyncio.create_task(run_one(i, b)) for i, b in selected]
        matched_frames = 0
        total_boxes = 0
        completed = 0
        try:
            for fut in asyncio.as_completed(tasks):
                idx_1, det, raw = await fut
                completed += 1
                if det.boxes:
                    matched_frames += 1
                    total_boxes += len(det.boxes)
                yield _ndjson(
                    {
                        "type": "frame",
                        "index": idx_1,
                        "width": det.width,
                        "height": det.height,
                        "boxes": [
                            {
                                "x": b.x,
                                "y": b.y,
                                "w": b.w,
                                "h": b.h,
                                "text": b.text,
                                "score": round(b.score, 3),
                            }
                            for b in det.boxes
                        ],
                        "raw": raw,
                    }
                )
        except asyncio.CancelledError:
            log.warning(
                "stream detect: client disconnected after %d/%d frames",
                completed,
                len(selected),
            )
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise
        except Exception as e:  # pragma: no cover
            log.exception("Streaming OCR failed")
            for t in tasks:
                if not t.done():
                    t.cancel()
            yield _ndjson({"type": "error", "message": str(e)})
            return

        total_wall = time.perf_counter() - loop_t0
        total_req = time.perf_counter() - t_req
        if per_frame_ms:
            per_frame_ms.sort()
            mean = sum(per_frame_ms) / len(per_frame_ms)
            p50 = per_frame_ms[len(per_frame_ms) // 2]
            p95 = per_frame_ms[int(len(per_frame_ms) * 0.95)]
            log.info(
                "stream detect: done frames=%d matched=%d boxes=%d "
                "ocr_wall=%.1fs req_wall=%.1fs per_frame mean=%.0fms "
                "p50=%.0fms p95=%.0fms max=%.0fms throughput=%.2f fps",
                len(selected),
                matched_frames,
                total_boxes,
                total_wall,
                total_req,
                mean,
                p50,
                p95,
                max(per_frame_ms),
                len(selected) / total_wall if total_wall > 0 else 0.0,
            )
        else:
            log.info(
                "stream detect: done frames=0 wall=%.1fs", total_wall
            )

        yield _ndjson(
            {
                "type": "done",
                "matched_frames": matched_frames,
                "total_boxes": total_boxes,
            }
        )

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            # Disable proxy buffering so events arrive incrementally.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ocr/raw/stream")
async def detect_raw_stream(
    file: UploadFile = File(...),
    frame_from: int = Form(1, ge=1),
    frame_to: int | None = Form(None),
    fps: float | None = Form(None),
    dedup_threshold: int = Form(2),
    max_gap: int = Form(1, ge=0, le=240),
) -> StreamingResponse:
    """Stream raw Textract responses across a frame range as NDJSON.

    This endpoint is query-free and intended for prompt-mode planning /
    detection where filtering happens upstream in TypeScript.
    """
    _validate_video(file)
    t_req = time.perf_counter()
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    if fps is not None and fps <= 0:
        fps = None

    log.info(
        "stream raw: recv file=%r bytes=%d range=%s..%s fps=%s "
        "dedup_threshold=%d max_gap=%d",
        file.filename,
        len(body),
        frame_from,
        frame_to,
        fps,
        dedup_threshold,
        max_gap,
    )

    suffix = _guess_suffix(file.filename)
    t_extract = time.perf_counter()
    try:
        frame_set, video_hash = get_or_extract(
            body,
            suffix=suffix,
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
        )
    except FrameExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    extract_ms = (time.perf_counter() - t_extract) * 1000

    n = frame_set.kept_count
    log.info(
        "stream raw: video_hash=%s frames=%d (raw=%d) extract/cache=%.0fms",
        video_hash[:12],
        n,
        frame_set.raw_count,
        extract_ms,
    )
    if n == 0:
        raise HTTPException(status_code=422, detail="No frames to process.")

    lo_1 = max(1, frame_from)
    hi_1 = frame_to if frame_to is not None else n
    hi_1 = min(max(1, hi_1), n)
    if lo_1 > hi_1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frame range: from={frame_from}, to={frame_to}",
        )

    selected_indices_1 = list(range(lo_1, hi_1 + 1))
    selected = [(i_1, frame_set.frames[i_1 - 1]) for i_1 in selected_indices_1]

    async def stream() -> AsyncIterator[bytes]:
        yield _ndjson(
            {
                "type": "start",
                "video_hash": video_hash,
                "deduplicated_count": n,
                "frame_from": lo_1,
                "frame_to": hi_1,
                "frame_indices": selected_indices_1,
                "total": len(selected),
            }
        )

        try:
            t_load = time.perf_counter()
            await asyncio.to_thread(ensure_reader_loaded)
            log.info(
                "stream raw: reader ready in %.0fms",
                (time.perf_counter() - t_load) * 1000,
            )
        except Exception as e:  # pragma: no cover
            log.exception("OCR reader load failed")
            yield _ndjson({"type": "error", "message": f"OCR init failed: {e}"})
            return

        sem = asyncio.Semaphore(_STREAM_OCR_CONCURRENCY)
        log.info(
            "stream raw: fanning out %d frames (range %d..%d) concurrency=%d",
            len(selected),
            lo_1,
            hi_1,
            _STREAM_OCR_CONCURRENCY,
        )

        # Cache under query_norm="" so prompt-mode raw OCR shares the same
        # LRU as detect/backtrack without colliding with query-filtered keys.
        cache_entry = ocr_cache.create_entry(
            video_hash=video_hash,
            query_norm="",
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
            frame_from=lo_1,
            frame_to=hi_1,
        )

        loop_t0 = time.perf_counter()
        per_frame_ms: list[float] = []

        async def run_one(idx_1: int, blob: bytes):
            t_wait = time.perf_counter()
            async with sem:
                wait_ms = (time.perf_counter() - t_wait) * 1000
                t_work = time.perf_counter()
                width, height, raw = await asyncio.to_thread(detect_frame_raw, blob)
                work_ms = (time.perf_counter() - t_work) * 1000
                per_frame_ms.append(work_ms)
                ocr_cache.put_frame(
                    cache_entry,
                    frame_idx_1=idx_1,
                    width=width,
                    height=height,
                    matched=[],
                    raw=raw,
                )
                log.info(
                    "raw frame #%d: wait=%.0fms work=%.0fms",
                    idx_1,
                    wait_ms,
                    work_ms,
                )
                return idx_1, width, height, raw

        tasks = [asyncio.create_task(run_one(i, b)) for i, b in selected]
        completed = 0
        try:
            for fut in asyncio.as_completed(tasks):
                idx_1, width, height, raw = await fut
                completed += 1
                yield _ndjson(
                    {
                        "type": "frame",
                        "index": idx_1,
                        "width": width,
                        "height": height,
                        "raw": raw,
                    }
                )
        except asyncio.CancelledError:
            log.warning(
                "stream raw: client disconnected after %d/%d frames",
                completed,
                len(selected),
            )
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise
        except Exception as e:  # pragma: no cover
            log.exception("Streaming raw OCR failed")
            for t in tasks:
                if not t.done():
                    t.cancel()
            yield _ndjson({"type": "error", "message": str(e)})
            return

        total_wall = time.perf_counter() - loop_t0
        total_req = time.perf_counter() - t_req
        if per_frame_ms:
            per_frame_ms.sort()
            mean = sum(per_frame_ms) / len(per_frame_ms)
            p50 = per_frame_ms[len(per_frame_ms) // 2]
            p95 = per_frame_ms[int(len(per_frame_ms) * 0.95)]
            log.info(
                "stream raw: done frames=%d ocr_wall=%.1fs req_wall=%.1fs "
                "per_frame mean=%.0fms p50=%.0fms p95=%.0fms max=%.0fms "
                "throughput=%.2f fps",
                len(selected),
                total_wall,
                total_req,
                mean,
                p50,
                p95,
                max(per_frame_ms),
                len(selected) / total_wall if total_wall > 0 else 0.0,
            )
        else:
            log.info("stream raw: done frames=0 wall=%.1fs", total_wall)

        yield _ndjson({"type": "done", "frames": completed})

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


async def _populate_phase1_for_backtrack(
    *,
    frame_set,
    video_hash: str,
    q_norm: str,
    fps: float | None,
    dedup_threshold: int,
    max_gap: int,
    lo_1: int,
    hi_1: int,
) -> ocr_cache.OcrEntry:
    """Run phase-1 detection over the selected frames into the OCR cache.

    Used by /api/ocr/backtrack when no cached entry exists for the given
    (video_hash, query_norm, fps, dedup_threshold, max_gap, range). Same
    fan-out strategy as detect_stream, just no NDJSON emission.
    """
    await asyncio.to_thread(ensure_reader_loaded)
    entry = ocr_cache.create_entry(
        video_hash=video_hash,
        query_norm=q_norm,
        fps=fps,
        dedup_threshold=dedup_threshold,
        max_gap=max_gap,
        frame_from=lo_1,
        frame_to=hi_1,
    )
    sem = asyncio.Semaphore(_STREAM_OCR_CONCURRENCY)

    async def run_one(idx_1: int, blob: bytes):
        async with sem:
            det, raw = await asyncio.to_thread(detect_frame, blob, q_norm)
            ocr_cache.put_frame(
                entry,
                frame_idx_1=idx_1,
                width=det.width,
                height=det.height,
                matched=list(det.boxes),
                raw=raw,
            )

    tasks = [
        asyncio.create_task(run_one(i_1, frame_set.frames[i_1 - 1]))
        for i_1 in range(lo_1, hi_1 + 1)
    ]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for t in tasks:
            if not t.done():
                t.cancel()
        raise
    return entry


async def _run_pass_stream(
    *,
    file: UploadFile,
    query: str,
    frame_from: int,
    frame_to: int | None,
    fps: float | None,
    dedup_threshold: int,
    max_gap: int,
    pass_name: str,
    origin: str,
    driver,
) -> StreamingResponse:
    """Shared endpoint body for backward/forward passes.

    Handles all request validation, frame extraction, OCR-cache hit-or-
    populate, then hands off to `driver(entry)` (a sync callable returning
    `list[AddedHit]`) via a worker thread and streams the results as
    NDJSON. `origin` is stamped onto every emitted frame event so the
    client can visually distinguish pass sources.
    """
    _validate_video(file)
    t_req = time.perf_counter()
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    if fps is not None and fps <= 0:
        fps = None

    log.info(
        "%s: recv file=%r bytes=%d query=%r range=%s..%s "
        "fps=%s dedup_threshold=%d max_gap=%d",
        pass_name,
        file.filename,
        len(body),
        query,
        frame_from,
        frame_to,
        fps,
        dedup_threshold,
        max_gap,
    )

    suffix = _guess_suffix(file.filename)
    try:
        frame_set, video_hash = get_or_extract(
            body,
            suffix=suffix,
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
        )
    except FrameExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    n = frame_set.kept_count
    if n == 0:
        raise HTTPException(status_code=422, detail="No frames to process.")

    lo_1 = max(1, frame_from)
    hi_1 = frame_to if frame_to is not None else n
    hi_1 = min(max(1, hi_1), n)
    if lo_1 > hi_1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frame range: from={frame_from}, to={frame_to}",
        )

    q_norm = normalize_query(query)

    async def stream() -> AsyncIterator[bytes]:
        yield _ndjson(
            {
                "type": "start",
                "video_hash": video_hash,
                "query": query,
                "frame_from": lo_1,
                "frame_to": hi_1,
                "total_frames": hi_1 - lo_1 + 1,
            }
        )

        if not q_norm:
            yield _ndjson(
                {"type": "done", "added_frames": 0, "added_boxes": 0}
            )
            return

        entry = ocr_cache.get_entry(
            video_hash=video_hash,
            query_norm=q_norm,
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
            frame_from=lo_1,
            frame_to=hi_1,
        )
        if entry is None:
            log.info(
                "%s: no cached phase-1 for query=%r range=%d..%d; "
                "running inline",
                pass_name,
                q_norm,
                lo_1,
                hi_1,
            )
            try:
                entry = await _populate_phase1_for_backtrack(
                    frame_set=frame_set,
                    video_hash=video_hash,
                    q_norm=q_norm,
                    fps=fps,
                    dedup_threshold=dedup_threshold,
                    max_gap=max_gap,
                    lo_1=lo_1,
                    hi_1=hi_1,
                )
            except Exception as e:  # pragma: no cover
                log.exception("%s: inline phase-1 failed", pass_name)
                yield _ndjson(
                    {"type": "error", "message": f"phase-1 failed: {e}"}
                )
                return

        # Run the pass driver on a worker thread so we don't block the
        # event loop while we crunch through (potentially large) raw
        # Textract payloads.
        t_drv = time.perf_counter()
        try:
            added = await asyncio.to_thread(driver, entry)
        except Exception as e:  # pragma: no cover
            log.exception("%s: driver failed", pass_name)
            yield _ndjson({"type": "error", "message": str(e)})
            return
        drv_ms = (time.perf_counter() - t_drv) * 1000

        frames_touched: set[int] = set()
        for hit in added:
            frames_touched.add(hit.frame_idx_1)
            fr = entry.per_frame.get(hit.frame_idx_1)
            yield _ndjson(
                {
                    "type": "frame",
                    "index": hit.frame_idx_1,
                    "width": fr.width if fr else 0,
                    "height": fr.height if fr else 0,
                    "box": {
                        "x": hit.box.x,
                        "y": hit.box.y,
                        "w": hit.box.w,
                        "h": hit.box.h,
                        "text": hit.box.text,
                        "score": round(hit.box.score, 3),
                    },
                    "origin": origin,
                }
            )

        total_req = time.perf_counter() - t_req
        log.info(
            "%s: done added_frames=%d added_boxes=%d "
            "driver=%.0fms req_wall=%.1fs",
            pass_name,
            len(frames_touched),
            len(added),
            drv_ms,
            total_req,
        )
        yield _ndjson(
            {
                "type": "done",
                "added_frames": len(frames_touched),
                "added_boxes": len(added),
            }
        )

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ocr/detect/single")
async def detect_single_frame(
    file: UploadFile = File(
        ...,
        description="A single JPEG/PNG frame to OCR.",
    ),
    query: str = Form(..., min_length=1, max_length=200),
) -> dict:
    """Run Textract on one image and return boxes + raw response.

    Used by the Next.js gap-filler so inserted frames (fetched via
    /api/frames/by_source_index) can be curated with the same OCR
    evidence that kept frames get. Unlike the stream endpoint, this
    takes the JPEG bytes directly rather than resolving a frame index
    through the extraction cache, so callers can OCR ad-hoc frames that
    aren't part of any kept sequence.
    """
    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("image/")
        or content_type in ("application/octet-stream", "binary/octet-stream")
    ):
        raise HTTPException(
            status_code=415,
            detail="Expected an image Content-Type (image/*) or octet-stream.",
        )
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    q_norm = normalize_query(query)
    if not q_norm:
        return {"width": 0, "height": 0, "boxes": [], "raw": None}

    await asyncio.to_thread(ensure_reader_loaded)
    det, raw = await asyncio.to_thread(detect_frame, body, q_norm)
    return {
        "width": det.width,
        "height": det.height,
        "boxes": [
            {
                "x": b.x,
                "y": b.y,
                "w": b.w,
                "h": b.h,
                "text": b.text,
                "score": round(b.score, 3),
            }
            for b in det.boxes
        ],
        "raw": raw,
    }


@app.post("/api/ocr/backtrack")
async def backtrack_stream(
    file: UploadFile = File(...),
    query: str = Form(..., min_length=1, max_length=200),
    frame_from: int = Form(1, ge=1),
    frame_to: int | None = Form(None),
    fps: float | None = Form(None),
    dedup_threshold: int = Form(2),
    max_gap: int = Form(1, ge=0, le=240),
) -> StreamingResponse:
    """Stream backward-pass hits as NDJSON.

    Walks frames in reverse and, for each genuinely new hit (no
    corresponding hit in the previous frame), looks for a partial of the
    query near the same region in frame n-1. Iteratively walks back.
    """
    return await _run_pass_stream(
        file=file,
        query=query,
        frame_from=frame_from,
        frame_to=frame_to,
        fps=fps,
        dedup_threshold=dedup_threshold,
        max_gap=max_gap,
        pass_name="backtrack",
        origin="backtrack",
        driver=ocr_backtrack.run_backtrack,
    )


@app.post("/api/ocr/forward")
async def forward_stream(
    file: UploadFile = File(...),
    query: str = Form(..., min_length=1, max_length=200),
    frame_from: int = Form(1, ge=1),
    frame_to: int | None = Form(None),
    fps: float | None = Form(None),
    dedup_threshold: int = Form(2),
    max_gap: int = Form(1, ge=0, le=240),
) -> StreamingResponse:
    """Stream forward-pass hits as NDJSON.

    Walks frames in order and, for each hit that has retired (no
    corresponding hit in the next frame), looks for a partial of the query
    near the same region in frame n+1. Iteratively walks forward. The
    mirror of /api/ocr/backtrack; typically called after it so it sees any
    backward-added anchors.
    """
    return await _run_pass_stream(
        file=file,
        query=query,
        frame_from=frame_from,
        frame_to=frame_to,
        fps=fps,
        dedup_threshold=dedup_threshold,
        max_gap=max_gap,
        pass_name="forward",
        origin="forward",
        driver=ocr_backtrack.run_forward,
    )


def _parse_export_boxes(raw: object) -> list[video_export.ExportBox]:
    """Flatten the client's `boxes_by_frame` mapping into a list.

    Accepts ``{ "<frame_idx_1>": [ { x, y, w, h, track_id? }, ... ], ... }``.
    Unknown keys are ignored; invalid boxes are dropped rather than failing
    the whole request (the UI stats may legitimately carry zero-area entries
    after a no-op re-run).

    ``track_id`` is optional: when present and shared between a box at
    frame ``F`` and a box at frame ``F+1``, the exporter will interpolate
    the drawbox coords across that window so the redaction tweens
    smoothly (see ``video_export.build_drawbox_filter``). Absent or empty
    track ids render statically for the full keyframe window, matching
    legacy (pre-linker) behavior.
    """
    if not isinstance(raw, dict):
        return []
    out: list[video_export.ExportBox] = []
    for k, v in raw.items():
        try:
            frame_1 = int(k)
        except (TypeError, ValueError):
            continue
        if frame_1 < 1 or not isinstance(v, list):
            continue
        for entry in v:
            if not isinstance(entry, dict):
                continue
            try:
                x = float(entry["x"])
                y = float(entry["y"])
                w = float(entry["w"])
                h = float(entry["h"])
            except (KeyError, TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            track_raw = entry.get("track_id")
            track_id = (
                track_raw
                if isinstance(track_raw, str) and track_raw
                else None
            )
            out.append(
                video_export.ExportBox(
                    frame=frame_1,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    track_id=track_id,
                )
            )
    return out


@app.post("/api/video/export")
async def export_video(
    file: UploadFile = File(...),
    payload: str = Form(...),
) -> FileResponse:
    """Render a redacted MP4 with detection boxes painted as filled
    rectangles on the source video.

    Request (multipart):
      - `file`: source video
      - `payload`: JSON string with
          {
            "fps": null | number,          // same value used at detect time
            "dedup_threshold": 2,
            "frame_width": number,         // reference frame dims
            "frame_height": number,
            "boxes_by_frame": {
              "<kept_frame_idx_1>": [ { "x","y","w","h" }, ... ],
              ...
            },
            "style": { "color": "black", "padding_px": 4 }
          }

    Returns an MP4. The time range each kept frame represents is
    reconstructed from the (video_hash, fps, dedup_threshold) frame
    cache; boxes are scaled from reference frame dims to source video
    resolution before rendering.
    """
    _validate_video(file)
    t_req = time.perf_counter()
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        parsed = json.loads(payload)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid payload JSON: {e}",
        ) from e
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Payload must be an object.")

    fps = parsed.get("fps")
    if fps is not None:
        try:
            fps = float(fps)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid fps.") from e
        if fps <= 0:
            fps = None

    dedup_threshold_raw = parsed.get("dedup_threshold", 2)
    try:
        dedup_threshold = int(dedup_threshold_raw)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail="Invalid dedup_threshold.",
        ) from e

    max_gap_raw = parsed.get("max_gap", 1)
    try:
        max_gap = int(max_gap_raw)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail="Invalid max_gap.",
        ) from e
    if max_gap < 0:
        max_gap = 0

    try:
        ref_width = int(parsed.get("frame_width", 0))
        ref_height = int(parsed.get("frame_height", 0))
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail="Invalid reference dimensions.",
        ) from e
    if ref_width <= 0 or ref_height <= 0:
        raise HTTPException(
            status_code=400,
            detail="frame_width and frame_height must be positive.",
        )

    style_raw = parsed.get("style") or {}
    style = video_export.ExportStyle(
        color=str(style_raw.get("color") or "black"),
        padding_px=int(style_raw.get("padding_px") or 4),
    )

    boxes = _parse_export_boxes(parsed.get("boxes_by_frame"))

    log.info(
        "export: recv file=%r bytes=%d boxes=%d ref=%dx%d fps=%s "
        "dedup=%d max_gap=%d",
        file.filename,
        len(body),
        len(boxes),
        ref_width,
        ref_height,
        fps,
        dedup_threshold,
        max_gap,
    )

    suffix = _guess_suffix(file.filename)
    try:
        frame_set, video_hash = get_or_extract(
            body,
            suffix=suffix,
            fps=fps,
            dedup_threshold=dedup_threshold,
            max_gap=max_gap,
        )
    except FrameExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if frame_set.source_fps is None or frame_set.source_fps <= 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not determine source video frame rate; export "
                "requires a probable fps (re-run detection with an "
                "explicit fps or install ffprobe)."
            ),
        )

    # Write the source video to a stable temp file so ffmpeg/ffprobe can
    # work against a path. We keep it alive until after ffmpeg returns.
    tmp_dir = Path(tempfile.mkdtemp(prefix="export_"))
    try:
        source_path = tmp_dir / f"input{suffix}"
        source_path.write_bytes(body)

        try:
            info = video_export.probe_video_info(source_path)
        except video_export.VideoExportError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        filter_graph = video_export.build_drawbox_filter(
            boxes=boxes,
            kept_source_indices=frame_set.kept_source_indices,
            source_fps=frame_set.source_fps,
            duration=info.duration,
            ref_width=ref_width,
            ref_height=ref_height,
            src_width=info.width,
            src_height=info.height,
            style=style,
        )

        log.info(
            "export: video_hash=%s src=%dx%d duration=%.2fs "
            "source_fps=%.3f filter_chars=%d",
            video_hash[:12],
            info.width,
            info.height,
            info.duration,
            frame_set.source_fps,
            len(filter_graph),
        )

        out_path = tmp_dir / "redacted.mp4"
        try:
            await asyncio.to_thread(
                video_export.render_redacted_video,
                source_path=source_path,
                out_path=out_path,
                filter_graph=filter_graph,
            )
        except video_export.VideoExportError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        req_ms = (time.perf_counter() - t_req) * 1000
        log.info("export: done in %.0fms -> %s", req_ms, out_path)

        # FileResponse does not clean up on its own; schedule the temp dir
        # for removal after the response finishes streaming.
        download_name = _export_filename(file.filename)
        return FileResponse(
            path=out_path,
            media_type="video/mp4",
            filename=download_name,
            background=BackgroundTask(_rmtree, tmp_dir),
        )
    except BaseException:
        _rmtree(tmp_dir)
        raise


def _export_filename(source: str | None) -> str:
    base = Path(source or "video").stem or "video"
    return f"{base}.redacted.mp4"


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # pragma: no cover
        pass

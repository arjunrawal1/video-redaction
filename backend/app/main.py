import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.frame_cache import get_or_extract
from app.frame_service import FrameExtractionError
from app.logging_config import configure as configure_logging
from app import ocr_backtrack, ocr_cache
from app.ocr_service import (
    detect_frame,
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
        4,
        ge=0,
        le=256,
        description=(
            "Perceptual hash Hamming threshold on a 256-bit phash "
            "(lower = stricter; only near-exact duplicates are removed)."
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

    return {
        "filename": file.filename,
        "video_hash": video_hash,
        "fps": fps,
        "dedup_threshold": dedup_threshold,
        "raw_frame_count": frame_set.raw_count,
        "deduplicated_count": frame_set.kept_count,
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
    dedup_threshold: int = Form(4),
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
        "fps=%s dedup_threshold=%d",
        file.filename,
        len(body),
        query,
        frame_from,
        frame_to,
        fps,
        dedup_threshold,
    )

    suffix = _guess_suffix(file.filename)
    t_extract = time.perf_counter()
    try:
        frame_set, video_hash = get_or_extract(
            body,
            suffix=suffix,
            fps=fps,
            dedup_threshold=dedup_threshold,
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


async def _populate_phase1_for_backtrack(
    *,
    frame_set,
    video_hash: str,
    q_norm: str,
    fps: float | None,
    dedup_threshold: int,
    lo_1: int,
    hi_1: int,
) -> ocr_cache.OcrEntry:
    """Run phase-1 detection over the selected frames into the OCR cache.

    Used by /api/ocr/backtrack when no cached entry exists for the given
    (video_hash, query_norm, fps, dedup_threshold, range). Same fan-out
    strategy as detect_stream, just no NDJSON emission.
    """
    await asyncio.to_thread(ensure_reader_loaded)
    entry = ocr_cache.create_entry(
        video_hash=video_hash,
        query_norm=q_norm,
        fps=fps,
        dedup_threshold=dedup_threshold,
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
        "fps=%s dedup_threshold=%d",
        pass_name,
        file.filename,
        len(body),
        query,
        frame_from,
        frame_to,
        fps,
        dedup_threshold,
    )

    suffix = _guess_suffix(file.filename)
    try:
        frame_set, video_hash = get_or_extract(
            body,
            suffix=suffix,
            fps=fps,
            dedup_threshold=dedup_threshold,
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


@app.post("/api/ocr/backtrack")
async def backtrack_stream(
    file: UploadFile = File(...),
    query: str = Form(..., min_length=1, max_length=200),
    frame_from: int = Form(1, ge=1),
    frame_to: int | None = Form(None),
    fps: float | None = Form(None),
    dedup_threshold: int = Form(4),
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
    dedup_threshold: int = Form(4),
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
        pass_name="forward",
        origin="forward",
        driver=ocr_backtrack.run_forward,
    )

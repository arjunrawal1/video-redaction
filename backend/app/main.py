import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.frame_service import FrameExtractionError, extract_deduplicated_frames

app = FastAPI(title="Video Redaction API", version="0.1.0")

_MAX_UPLOAD_BYTES = 500 * 1024 * 1024

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
    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/")
        or content_type in ("application/octet-stream", "binary/octet-stream")
    ):
        raise HTTPException(
            status_code=415,
            detail="Expected a video Content-Type (video/*) or octet-stream.",
        )

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    suffix = _guess_suffix(file.filename)
    try:
        blobs, raw_count, kept_count = extract_deduplicated_frames(
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
        for b in blobs
    ]

    return {
        "filename": file.filename,
        "fps": fps,
        "dedup_threshold": dedup_threshold,
        "raw_frame_count": raw_count,
        "deduplicated_count": kept_count,
        "frames": frames_out,
    }

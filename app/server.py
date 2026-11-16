# Run with: python -m uvicorn app.server:app --reload --port 8000

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference_service import CaptionService


CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH", "checkpoints/checkpoint_img_caption.pth"
)
VOCAB_PATH = os.getenv("VOCAB_PATH") or None
DEVICE = os.getenv("DEVICE", "cpu")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the caption model once for the application lifetime."""
    application.state.caption_service = CaptionService(
        CHECKPOINT_PATH,
        VOCAB_PATH,
        DEVICE,
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/api/caption")
def caption(file: UploadFile) -> JSONResponse:
    """Generate a caption for a supported uploaded image."""
    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported image type. Use PNG, JPEG, or WebP."},
        )

    try:
        image_bytes = bytearray()
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            image_bytes.extend(chunk)
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": f"Image exceeds the {MAX_UPLOAD_MB} MB limit."},
                )

        result = app.state.caption_service.predict(bytes(image_bytes))
        return JSONResponse(content=result)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        logger.exception("Caption request failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Caption generation failed."},
        )


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

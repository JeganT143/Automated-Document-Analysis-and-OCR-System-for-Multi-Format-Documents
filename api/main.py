"""FastAPI application factory — the Document OCR + AI Pipeline API.

Run locally:   uvicorn api.main:app --reload
In Docker:     see Dockerfile.api (CMD runs the same, honouring $PORT).
"""

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .logging_config import configure_logging
from .routers import documents, models, samples, search
from .settings import settings

configure_logging()
logger = logging.getLogger("api")

app = FastAPI(
    title="Document OCR + AI Pipeline API",
    description=(
        "Classical CV/OCR pipeline (preprocessing, layout analysis, adaptive-PSM "
        "Tesseract) with an optional LLM structured-extraction, grounded Q&A and "
        "cross-document search layer, proxied through OpenRouter."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code}",
        extra={"request_id": request_id, "path": request.url.path,
               "duration_ms": duration_ms, "status_code": response.status_code},
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(documents.router)
app.include_router(models.router)
app.include_router(samples.router)
app.include_router(search.router)


@app.get("/healthz", tags=["health"])
async def healthz():
    return {"status": "ok"}

"""FastAPI dependency providers: the OCR pipeline singleton, the caller's
session id, and the shared document store.

Session id contract: the web client generates one UUID per browser session
and sends it as ``X-Session-Id`` on every request. Without that header, each
request is treated as its own isolated session (safe default, but it means
/ask and /search won't find documents from other calls) — this is a
deliberate "fail isolated, not leaky" choice.
"""

import uuid

from fastapi import Header

from src.pipeline import DocumentOCRPipeline

from .state import document_store

_pipeline: DocumentOCRPipeline | None = None


def get_pipeline() -> DocumentOCRPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentOCRPipeline()
    return _pipeline


def get_session_id(x_session_id: str | None = Header(default=None)) -> str:
    return x_session_id or str(uuid.uuid4())


def get_document_store():
    return document_store

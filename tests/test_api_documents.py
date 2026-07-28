"""api/routers/documents.py — contract tests via FastAPI TestClient.

The OCR pipeline is swapped for a fake via dependency_overrides: these tests
verify the HTTP contract (status codes, response shape, session-store side
effects), not OCR accuracy (that's scripts/evaluate.py's job) or LLM
behaviour (that's tests/test_llm_*.py's job). Image decoding/encoding still
runs for real against a tiny synthetic image.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.deps import get_document_store, get_pipeline
from api.main import app
from src.llm.search import DocumentStore


class FakePipeline:
    _ready = True

    def trace(self, image, analyze_layout=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        result = {
            "text": "INVOICE\nTOTAL: $10.00",
            "words": ["INVOICE", "TOTAL:", "$10.00"],
            "word_boxes": [{"text": "INVOICE", "conf": 90.0, "bbox": (0, 0, 5, 5)}],
            "word_count": 1,
            "mean_confidence": 90.0,
            "psm_used": 6,
            "fields": {"total": "10.00"},
            "regions": [{"bbox": (0, 0, 5, 5), "type": "text"}],
            "preprocess_info": {},
            "original": image,
            "processed_image": gray,
            "metrics": {"total_s": 0.01},
        }
        return {"pre_stages": [], "result": result}


class UnavailablePipeline:
    _ready = False

    def trace(self, image, analyze_layout=True):
        raise RuntimeError("Tesseract not available.")


def _sample_png_bytes() -> bytes:
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def store():
    return DocumentStore()


@pytest.fixture
def client(store):
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    app.dependency_overrides[get_document_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["models"]) >= 1
    assert "llm_configured" in body


def test_create_document_returns_stages_and_fields(client):
    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    resp = client.post("/v1/documents", files=files, headers={"X-Session-Id": "s1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "INVOICE\nTOTAL: $10.00"
    assert body["fields"] == {"total": "10.00"}
    assert body["extraction"] is None  # extract=False by default
    stage_keys = [s["key"] for s in body["stages"]]
    assert "input" in stage_keys
    assert "recognition" in stage_keys
    assert "layout" in stage_keys
    # every stage image should be non-empty base64
    assert all(s["image_jpeg_b64"] for s in body["stages"])


def test_create_document_empty_file_rejected(client):
    files = {"file": ("empty.png", b"", "image/png")}
    resp = client.post("/v1/documents", files=files)
    assert resp.status_code == 400


def test_create_document_bad_image_rejected(client):
    files = {"file": ("bad.png", b"not an image", "image/png")}
    resp = client.post("/v1/documents", files=files)
    assert resp.status_code == 400


def test_create_document_no_tesseract_returns_503(store):
    app.dependency_overrides[get_pipeline] = lambda: UnavailablePipeline()
    app.dependency_overrides[get_document_store] = lambda: store
    with TestClient(app) as c:
        files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
        resp = c.post("/v1/documents", files=files)
    app.dependency_overrides.clear()
    assert resp.status_code == 503


def test_ask_unknown_document_404(client):
    resp = client.post("/v1/documents/does-not-exist/ask", json={"question": "hi"})
    assert resp.status_code == 404


def test_ask_known_document_without_api_key_503(client, monkeypatch):
    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    created = client.post("/v1/documents", files=files, headers={"X-Session-Id": "s1"})
    doc_id = created.json()["id"]

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "is_configured", lambda: False)

    resp = client.post(
        f"/v1/documents/{doc_id}/ask",
        json={"question": "What's the total?"},
        headers={"X-Session-Id": "s1"},
    )
    assert resp.status_code == 503


def test_ask_known_document_with_mocked_llm(client, monkeypatch):
    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    created = client.post("/v1/documents", files=files, headers={"X-Session-Id": "s1"})
    doc_id = created.json()["id"]

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "is_configured", lambda: True)
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "$10.00")

    resp = client.post(
        f"/v1/documents/{doc_id}/ask",
        json={"question": "What's the total?"},
        headers={"X-Session-Id": "s1"},
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "$10.00"


def test_ask_returns_502_on_llm_call_failure(client, monkeypatch):
    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    created = client.post("/v1/documents", files=files, headers={"X-Session-Id": "s1"})
    doc_id = created.json()["id"]

    from src.llm import client as llm_client

    def boom(*a, **k):
        raise RuntimeError("404 - No endpoints found for this model")

    monkeypatch.setattr(llm_client, "is_configured", lambda: True)
    monkeypatch.setattr(llm_client, "chat", boom)

    resp = client.post(
        f"/v1/documents/{doc_id}/ask",
        json={"question": "What's the total?"},
        headers={"X-Session-Id": "s1"},
    )
    assert resp.status_code == 502


def test_session_isolation_for_ask(client):
    files = {"file": ("sample.png", _sample_png_bytes(), "image/png")}
    created = client.post("/v1/documents", files=files, headers={"X-Session-Id": "s1"})
    doc_id = created.json()["id"]

    # a different session must not be able to reach session s1's document
    resp = client.post(
        f"/v1/documents/{doc_id}/ask",
        json={"question": "hi"},
        headers={"X-Session-Id": "s2"},
    )
    assert resp.status_code == 404

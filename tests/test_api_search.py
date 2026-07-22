"""api/routers/search.py — contract tests via FastAPI TestClient.

Embeddings are mocked (same approach as tests/test_llm_search.py) so these
tests never touch the real fastembed model or the network.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.deps import get_document_store
from api.main import app
from src.llm import client as llm_client
from src.llm import search as llm_search
from src.llm.search import DocumentStore


def fake_embed_one(text: str) -> np.ndarray:
    vec = np.array([1.0 if "northwind" in text.lower() else 0.0, 1.0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch):
    monkeypatch.setattr(llm_search, "embed_one", fake_embed_one)


@pytest.fixture
def client_with_store():
    store = DocumentStore()
    app.dependency_overrides[get_document_store] = lambda: store
    with TestClient(app) as c:
        yield c, store
    app.dependency_overrides.clear()


def test_search_empty_session_returns_no_hits(client_with_store):
    client, _ = client_with_store
    resp = client.post("/v1/search", json={"query": "anything"}, headers={"X-Session-Id": "s1"})
    assert resp.status_code == 200
    assert resp.json()["hits"] == []


def test_search_finds_previously_added_document(client_with_store, monkeypatch):
    client, store = client_with_store
    store.add("s1", "doc-a", "Invoice from Northwind Traders, total $10")
    monkeypatch.setattr(llm_client, "is_configured", lambda: False)

    resp = client.post("/v1/search", json={"query": "Northwind"}, headers={"X-Session-Id": "s1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"]
    assert body["hits"][0]["document_id"] == "doc-a"


def test_search_respects_session_partitioning(client_with_store):
    client, store = client_with_store
    store.add("s1", "doc-a", "Invoice from Northwind Traders")
    resp = client.post("/v1/search", json={"query": "Northwind"}, headers={"X-Session-Id": "s2"})
    assert resp.status_code == 200
    assert resp.json()["hits"] == []

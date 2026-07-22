"""src/llm/search.py — session-partitioned store + cross-document search.

Embeddings are mocked with a tiny deterministic keyword vector so similarity
ordering is predictable without loading the real fastembed model (keeps
tests fast and offline).
"""

import numpy as np
import pytest

from src.llm import client, qa, search

KEYWORDS = ["vendor-a", "vendor-b", "widget", "gadget"]


def fake_embed_one(text: str) -> np.ndarray:
    t = text.lower()
    vec = np.array([1.0 if kw in t else 0.0 for kw in KEYWORDS], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch):
    monkeypatch.setattr(search, "embed_one", fake_embed_one)


def test_search_ranks_by_relevance():
    store = search.DocumentStore()
    store.add("session-1", "doc-a", "Invoice from vendor-a for widgets")
    store.add("session-1", "doc-b", "Invoice from vendor-b for gadgets")

    hits = store.search("session-1", "vendor-a widget order", top_k=2)
    assert hits[0].document_id == "doc-a"
    assert hits[0].score >= hits[1].score


def test_sessions_are_isolated():
    store = search.DocumentStore()
    store.add("session-1", "doc-a", "Invoice from vendor-a")
    hits = store.search("session-2", "vendor-a", top_k=5)
    assert hits == []


def test_get_respects_session_partitioning():
    store = search.DocumentStore()
    store.add("session-1", "doc-a", "Invoice from vendor-a")
    assert store.get("session-1", "doc-a") is not None
    assert store.get("session-2", "doc-a") is None


def test_lru_eviction_per_session(monkeypatch):
    monkeypatch.setattr(search, "MAX_DOCS_PER_SESSION", 2)
    store = search.DocumentStore()
    store.add("session-1", "doc-1", "vendor-a")
    store.add("session-1", "doc-2", "vendor-b")
    store.add("session-1", "doc-3", "widget")  # should evict doc-1

    assert store.get("session-1", "doc-1") is None
    assert store.get("session-1", "doc-2") is not None
    assert store.get("session-1", "doc-3") is not None


def test_answer_with_no_documents():
    store = search.DocumentStore()
    result = search.answer(store, "session-1", "anything")
    assert result.hits == []
    assert "no documents" in result.answer.lower()


def test_answer_without_api_key_lists_matches(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: False)
    store = search.DocumentStore()
    store.add("session-1", "doc-a", "Invoice from vendor-a")
    result = search.answer(store, "session-1", "vendor-a")
    assert result.hits
    assert "OPENROUTER_API_KEY" in result.answer


def test_answer_with_api_key_uses_qa(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(qa, "ask", lambda *a, **k: qa.QAAnswer(answer="vendor-a", model="m"))
    store = search.DocumentStore()
    store.add("session-1", "doc-a", "Invoice from vendor-a")
    result = search.answer(store, "session-1", "who is the vendor?")
    assert result.answer == "vendor-a"
    assert result.model == "m"
    assert len(result.hits) == 1


def test_answer_grounds_on_full_text_not_the_short_snippet(monkeypatch):
    """Regression test: answer() used to ground the LLM call on the
    280-char UI snippet, which could cut a document off before fields like
    the total ever appeared -- caught via live testing against the real
    API, where a search answer picked up a line-item amount instead of the
    actual total because it never saw that part of the text."""
    monkeypatch.setattr(client, "is_configured", lambda: True)
    captured = {}

    def fake_ask(context, query, model=None):
        captured["context"] = context
        return qa.QAAnswer(answer="ok", model="m")

    monkeypatch.setattr(qa, "ask", fake_ask)
    store = search.DocumentStore()
    long_text = "vendor-a header " + ("filler line. " * 50) + "TOTAL: $999.99"
    assert len(long_text) > 280  # would have been truncated by the old bug
    store.add("session-1", "doc-a", long_text, summary="vendor-a short summary")

    search.answer(store, "session-1", "what is the total?")
    assert "TOTAL: $999.99" in captured["context"]


def test_answer_degrades_gracefully_on_llm_call_error(monkeypatch):
    """qa.ask() can raise LLMCallError (bad model id, rate limit, network
    error) even when a key is configured -- search.answer() must not crash,
    it should fall back to listing the retrieved hits."""
    monkeypatch.setattr(client, "is_configured", lambda: True)

    def boom(*a, **k):
        raise qa.LLMCallError("404 - No endpoints found for this model")

    monkeypatch.setattr(qa, "ask", boom)
    store = search.DocumentStore()
    store.add("session-1", "doc-a", "Invoice from vendor-a")
    result = search.answer(store, "session-1", "who is the vendor?")
    assert "doc-a" in result.answer
    assert len(result.hits) == 1

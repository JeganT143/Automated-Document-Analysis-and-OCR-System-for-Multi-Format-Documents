"""src/llm/qa.py — grounded single-document Q&A. OpenRouter calls mocked."""

import pytest

from src.llm import client, qa

DOC_TEXT = "INVOICE\nNorthwind Traders\nTOTAL: $129.60"


def test_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: False)
    with pytest.raises(qa.NotConfiguredError):
        qa.ask(DOC_TEXT, "What is the total?")


def test_ask_returns_model_answer(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "chat", lambda *a, **k: "  $129.60  ")
    result = qa.ask(DOC_TEXT, "What is the total?", model="openai/gpt-4o-mini")
    assert result.answer == "$129.60"
    assert result.model == "openai/gpt-4o-mini"


def test_ask_wraps_api_errors(monkeypatch):
    """A live bug: OpenRouter can 404 on a retired/renamed model id, or hit
    rate limits/network errors — ask() must not let the raw SDK exception
    propagate uncaught (it did, until this was caught via manual testing
    against the real API)."""
    monkeypatch.setattr(client, "is_configured", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("404 - No endpoints found for this model")

    monkeypatch.setattr(client, "chat", boom)
    with pytest.raises(qa.LLMCallError, match="404"):
        qa.ask(DOC_TEXT, "What is the total?")


def test_ask_passes_document_and_question_into_prompt(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: True)
    captured = {}

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return "answer"

    monkeypatch.setattr(client, "chat", fake_chat)
    qa.ask(DOC_TEXT, "Who is the vendor?")
    user_msg = captured["messages"][-1]["content"]
    assert "Northwind Traders" in user_msg
    assert "Who is the vendor?" in user_msg

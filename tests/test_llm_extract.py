"""src/llm/extract.py — structured extraction, repair retry, regex fallback.

All OpenRouter calls are mocked: these tests run with zero API key and zero
network access, in CI or anywhere else.
"""

import json

from src.llm import client, extract

SAMPLE_TEXT = """INVOICE
Northwind Traders
Invoice No: INV-2026-4821
Date: 14 Mar 2026

Subtotal: $120.00
Tax (8%): $9.60
TOTAL: $129.60
"""

VALID_JSON = json.dumps({
    "vendor": "Northwind Traders",
    "invoice_no": "INV-2026-4821",
    "date": "14 Mar 2026",
    "currency": "$",
    "line_items": [],
    "subtotal": 120.0,
    "tax": 9.6,
    "total": 129.6,
})


def test_no_api_key_falls_back_to_regex(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: False)
    result = extract.extract(SAMPLE_TEXT)
    assert result.source == "regex_fallback"
    assert result.warning and "OPENROUTER_API_KEY" in result.warning
    assert result.data.total == 129.60


def test_empty_text_falls_back_without_calling_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "chat", lambda *a, **k: calls.append(1))
    result = extract.extract("   ")
    assert result.source == "regex_fallback"
    assert not calls


def test_valid_llm_response_is_used(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "chat", lambda *a, **k: VALID_JSON)
    result = extract.extract(SAMPLE_TEXT, model="openai/gpt-4o-mini")
    assert result.source == "llm"
    assert result.model == "openai/gpt-4o-mini"
    assert result.data.vendor == "Northwind Traders"
    assert result.data.total == 129.6


def test_invalid_json_then_valid_repair_retry(monkeypatch):
    responses = iter(["not json at all", VALID_JSON])
    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "chat", lambda *a, **k: next(responses))
    result = extract.extract(SAMPLE_TEXT)
    assert result.source == "llm"
    assert result.data.total == 129.6


def test_invalid_json_twice_falls_back(monkeypatch):
    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "chat", lambda *a, **k: "still not json")
    result = extract.extract(SAMPLE_TEXT)
    assert result.source == "regex_fallback"
    assert "invalid JSON twice" in result.warning


def test_schema_violation_triggers_fallback(monkeypatch):
    # subtotal must be a number — a string should fail Pydantic validation
    # on both attempts and fall back.
    bad = json.dumps({**json.loads(VALID_JSON), "subtotal": "one-twenty"})
    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "chat", lambda *a, **k: bad)
    result = extract.extract(SAMPLE_TEXT)
    assert result.source == "regex_fallback"


def test_llm_exception_falls_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "chat", boom)
    result = extract.extract(SAMPLE_TEXT)
    assert result.source == "regex_fallback"
    assert "rate limited" in result.warning

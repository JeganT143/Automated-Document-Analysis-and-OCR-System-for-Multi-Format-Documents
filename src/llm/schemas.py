"""Pydantic schemas for the LLM document-intelligence layer.

Kept separate from src/postprocessing.py's plain dict field extraction: this
is the *structured, validated* shape produced by the LLM extractor (or by
the regex fallback repackaged into the same shape), so API/UI code only ever
deals with one contract regardless of which path produced it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = ""
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class ExtractedInvoice(BaseModel):
    vendor: str | None = None
    invoice_no: str | None = None
    date: str | None = None
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None


class ExtractionResult(BaseModel):
    """The extracted invoice plus provenance, so callers can be honest about
    how the numbers were produced instead of silently mixing LLM and regex
    output."""

    data: ExtractedInvoice
    source: str  # "llm" | "regex_fallback"
    model: str | None = None
    warning: str | None = None


class QAAnswer(BaseModel):
    answer: str
    grounded: bool = True
    model: str | None = None


class SearchHit(BaseModel):
    document_id: str
    score: float
    snippet: str


class SearchAnswer(BaseModel):
    answer: str
    hits: list[SearchHit] = Field(default_factory=list)
    model: str | None = None

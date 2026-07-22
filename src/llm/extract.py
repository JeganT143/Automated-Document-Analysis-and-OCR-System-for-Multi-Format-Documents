"""LLM-powered structured extraction, with a safe regex fallback.

Flow: OCR text -> ask an LLM for JSON matching ``ExtractedInvoice`` -> validate.
On invalid JSON/schema, one repair retry (the validation error is fed back to
the model). On no API key configured, or both attempts failing, fall back to
the existing regex ``extract_fields()`` (src/postprocessing.py) so the
pipeline never hard-fails just because no LLM is configured — it degrades
instead, and says so via ``ExtractionResult.source``/``warning``.
"""

import json

from pydantic import ValidationError

from ..postprocessing import extract_fields
from . import client
from .schemas import ExtractedInvoice, ExtractionResult

_SYSTEM_PROMPT = """You are a precise document-extraction engine. You will be \
given the raw OCR text of an invoice or receipt, which may contain OCR noise \
(misread characters, broken lines). Extract structured data as a single JSON \
object with EXACTLY these keys:

{
  "vendor": string or null,
  "invoice_no": string or null,
  "date": string or null (as written in the document),
  "currency": string or null (ISO code or symbol as seen in the text),
  "line_items": [{"description": string, "quantity": number or null, \
"unit_price": number or null, "amount": number or null}, ...],
  "subtotal": number or null,
  "tax": number or null,
  "total": number or null
}

Rules:
- Only use information present in the text. Never invent values.
- If a field is not present, use null (or [] for line_items).
- Numbers must be plain JSON numbers: no currency symbols, no thousands separators.
- Respond with ONLY the JSON object. No prose, no markdown code fences."""


def _regex_fallback(ocr_text: str, warning: str) -> ExtractionResult:
    fields = extract_fields(ocr_text)
    total = None
    if fields.get("total"):
        try:
            total = float(fields["total"])
        except ValueError:
            total = None
    data = ExtractedInvoice(
        invoice_no=fields.get("invoice_no"),
        date=fields.get("date"),
        total=total,
    )
    return ExtractionResult(data=data, source="regex_fallback", warning=warning)


def extract(ocr_text: str, model: str | None = None) -> ExtractionResult:
    """Extract structured invoice fields from OCR text.

    Always returns a result — never raises — so callers (the API) don't have
    to special-case "no LLM configured" or "LLM had a bad day".
    """
    if not ocr_text or not ocr_text.strip():
        return _regex_fallback(ocr_text or "", warning="empty OCR text")

    if not client.is_configured():
        return _regex_fallback(ocr_text, warning="OPENROUTER_API_KEY not configured")

    resolved_model = client.resolve_model(model)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": ocr_text[:6000]},
    ]

    last_error = None
    for _attempt in range(2):
        try:
            raw = client.chat(messages, model=resolved_model, json_mode=True)
        except Exception as e:  # network / auth / rate-limit errors
            return _regex_fallback(ocr_text, warning=f"LLM call failed: {e}")

        try:
            payload = json.loads(raw)
            parsed = ExtractedInvoice.model_validate(payload)
            return ExtractionResult(data=parsed, source="llm", model=resolved_model)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"That response was invalid: {e}. Reply again with ONLY "
                    "a valid JSON object matching the schema, no other text.")},
            ]

    return _regex_fallback(
        ocr_text, warning=f"LLM returned invalid JSON twice ({last_error})")

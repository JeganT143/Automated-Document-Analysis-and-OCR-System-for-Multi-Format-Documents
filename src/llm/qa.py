"""Grounded single-document Q&A.

The document's OCR text is short enough (a single invoice/receipt page) to
stuff directly into the prompt — no retrieval needed here, that's what
search.py is for (cross-document). The model is instructed to answer only
from the given text and say so when it can't, to avoid inventing numbers.
"""

from . import client
from .schemas import QAAnswer

_SYSTEM_PROMPT = """You answer questions about ONE document using ONLY the \
OCR text provided below. If the answer isn't in the text, say so plainly \
instead of guessing. Be concise — a sentence or two, or a single value if \
that's what was asked. Never invent numbers, names, or dates that aren't in \
the text."""


class NotConfiguredError(RuntimeError):
    """Raised when OPENROUTER_API_KEY isn't set — Q&A has no safe fallback
    (unlike extraction, there's no regex substitute for open-ended
    questions), so callers must handle this explicitly rather than silently
    degrading."""


def ask(document_text: str, question: str, model: str | None = None) -> QAAnswer:
    if not client.is_configured():
        raise NotConfiguredError("OPENROUTER_API_KEY not configured")

    resolved_model = client.resolve_model(model)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"DOCUMENT TEXT:\n{document_text[:6000]}\n\nQUESTION: {question}"},
    ]
    answer = client.chat(messages, model=resolved_model, temperature=0.0, max_tokens=400)
    return QAAnswer(answer=answer.strip(), model=resolved_model)

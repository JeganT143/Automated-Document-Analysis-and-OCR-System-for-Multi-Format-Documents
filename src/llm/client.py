"""OpenRouter client.

OpenRouter exposes a single OpenAI-compatible endpoint that proxies dozens of
providers' models, so we use the official ``openai`` SDK against it instead
of maintaining a bespoke HTTP client — swapping models is just a different
``model`` string per request.

The API key lives server-side only (``OPENROUTER_API_KEY``). The UI never
sees it; it only ever sends a model *id* chosen from :data:`AVAILABLE_MODELS`.
"""

import os
from functools import lru_cache

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Curated, cross-provider allow-list for the model-selector dropdown.
# Deliberately small: one fast/cheap model per major provider family, so a
# demo user can compare quality *and* the pipeline exercises genuinely
# different instruction-following behaviour rather than N near-identical
# models. IDs come from https://openrouter.ai/models and occasionally
# change — review periodically. Prices are indicative (USD / 1M tokens) and
# only used for the cost estimate shown in the eval report / UI.
AVAILABLE_MODELS = [
    {"id": "google/gemini-2.0-flash-001", "label": "Gemini 2.0 Flash",
     "provider": "Google", "cost_per_1m_in": 0.10, "cost_per_1m_out": 0.40},
    {"id": "openai/gpt-4o-mini", "label": "GPT-4o mini",
     "provider": "OpenAI", "cost_per_1m_in": 0.15, "cost_per_1m_out": 0.60},
    {"id": "anthropic/claude-3.5-haiku", "label": "Claude 3.5 Haiku",
     "provider": "Anthropic", "cost_per_1m_in": 0.80, "cost_per_1m_out": 4.00},
    {"id": "meta-llama/llama-3.3-70b-instruct", "label": "Llama 3.3 70B",
     "provider": "Meta", "cost_per_1m_in": 0.12, "cost_per_1m_out": 0.30},
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]["id"]
_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}


def is_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def resolve_model(model: str | None) -> str:
    """Only ever pass through allow-listed model ids to OpenRouter."""
    if model and model in _MODEL_IDS:
        return model
    return DEFAULT_MODEL


@lru_cache
def _client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


def chat(messages, model=None, *, json_mode=False, temperature=0.0, max_tokens=1024):
    """One-shot chat completion. Callers must check `is_configured()` first —
    this raises if OPENROUTER_API_KEY isn't set."""
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(
        model=resolve_model(model),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": os.environ.get("PUBLIC_APP_URL", "https://jegant.dev"),
            "X-Title": "Document OCR + AI Pipeline",
        },
        **kwargs,
    )
    return resp.choices[0].message.content

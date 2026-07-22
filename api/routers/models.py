"""GET /v1/models — curated OpenRouter model allow-list for the UI dropdown.

Static and hardcoded (src/llm/client.AVAILABLE_MODELS) rather than proxied
live from OpenRouter's own /models endpoint: it's a deliberately small,
cross-provider curated set (see the comment there), and a static list means
this endpoint never depends on OpenRouter being reachable.
"""

from fastapi import APIRouter

from src.llm.client import AVAILABLE_MODELS, is_configured

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models():
    return {"models": AVAILABLE_MODELS, "llm_configured": is_configured()}

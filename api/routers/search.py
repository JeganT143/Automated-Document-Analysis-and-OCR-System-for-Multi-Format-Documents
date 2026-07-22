"""POST /v1/search — cross-document semantic search + grounded answer, scoped
to the caller's session (see src/llm/search.py)."""

from fastapi import APIRouter, Depends

from src.llm import search as llm_search
from src.llm.schemas import SearchAnswer

from ..deps import get_document_store, get_session_id
from ..schemas import SearchRequest

router = APIRouter(tags=["search"])


@router.post("/v1/search", response_model=SearchAnswer)
async def search_documents(
    body: SearchRequest,
    session_id: str = Depends(get_session_id),
    store=Depends(get_document_store),
):
    return llm_search.answer(store, session_id, body.query, model=body.model, top_k=body.top_k)

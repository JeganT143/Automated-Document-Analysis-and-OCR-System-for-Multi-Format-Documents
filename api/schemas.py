"""API request/response models.

Distinct from src/llm/schemas.py (the LLM extraction/QA/search payloads,
which are transport-agnostic): these wrap those plus the pipeline-specific
fields (word boxes, rendered stage images) that only make sense at the HTTP
boundary.
"""

from pydantic import BaseModel

from src.llm.schemas import ExtractionResult


class Stage(BaseModel):
    key: str
    title: str
    desc: str
    meta: dict
    image_jpeg_b64: str


class WordBox(BaseModel):
    text: str
    conf: float
    bbox: tuple[int, int, int, int]


class Region(BaseModel):
    type: str
    bbox: tuple[int, int, int, int]


class DocumentResult(BaseModel):
    id: str
    text: str
    word_count: int
    mean_confidence: float
    psm_used: int | None = None
    fields: dict
    regions: list[Region]
    word_boxes: list[WordBox]
    stages: list[Stage]
    metrics: dict
    extraction: ExtractionResult | None = None


class AskRequest(BaseModel):
    question: str
    model: str | None = None


class SearchRequest(BaseModel):
    query: str
    model: str | None = None
    top_k: int = 3

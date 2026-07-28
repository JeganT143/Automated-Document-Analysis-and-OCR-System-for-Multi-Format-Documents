"""POST /v1/documents — OCR + layout analysis + optional LLM structured
extraction, returned as a fully-rendered stage-by-stage trace (the same
"pipeline walkthrough" idea as the original Streamlit app, now served over
HTTP so any client can render it).

POST /v1/documents/{id}/ask — grounded Q&A over a previously processed
document (looked up from the session-scoped document store).
"""

import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.llm import extract as llm_extract
from src.llm import qa as llm_qa
from src.pipeline import DocumentOCRPipeline

from .. import visualize
from ..deps import get_document_store, get_pipeline, get_session_id
from ..schemas import AskRequest, DocumentResult, Region, Stage, WordBox

router = APIRouter(tags=["documents"])


def _decode_upload(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image — is this a valid image file?")
    return image


def _build_stages(traced: dict) -> list[Stage]:
    result = traced["result"]
    stages = [Stage(
        key="input", title="Input document",
        desc="The raw source handed to the pipeline.",
        meta={"size": f'{result["original"].shape[1]}x{result["original"].shape[0]}'},
        image_jpeg_b64=visualize.encode_jpeg_b64(visualize.fit(result["original"])),
    )]

    for s in traced["pre_stages"]:
        h, w = s["image"].shape[:2]
        stages.append(Stage(
            key=s["key"], title=s["title"], desc=s["desc"],
            meta={**s["meta"], "size": f"{w}x{h}"},
            image_jpeg_b64=visualize.encode_jpeg_b64(visualize.fit(s["image"])),
        ))

    proc = result["processed_image"]
    disp = visualize.fit(proc)

    if result["regions"]:
        boxes = visualize.scale_boxes(result["regions"], proc.shape, disp.shape)
        stages.append(Stage(
            key="layout", title="Layout analysis",
            desc="Connected components and morphological smearing classify regions.",
            meta={"regions": len(result["regions"])},
            image_jpeg_b64=visualize.encode_jpeg_b64(visualize.draw_regions(disp, boxes)),
        ))

    word_boxes = visualize.scale_boxes(result["word_boxes"], proc.shape, disp.shape)
    stages.append(Stage(
        key="recognition", title="Recognition",
        desc="Adaptive-PSM Tesseract. Boxes: green ≥75%, amber ≥50%, red below.",
        meta={"words": result["word_count"], "mean_confidence": result["mean_confidence"]},
        image_jpeg_b64=visualize.encode_jpeg_b64(visualize.draw_word_boxes(disp, word_boxes)),
    ))
    return stages


@router.post("/v1/documents", response_model=DocumentResult)
async def create_document(
    file: UploadFile = File(...),
    analyze_layout: bool = Form(True),
    extract: bool = Form(False),
    model: str | None = Form(None),
    pipeline: DocumentOCRPipeline = Depends(get_pipeline),
    session_id: str = Depends(get_session_id),
    store=Depends(get_document_store),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    image = _decode_upload(raw)

    try:
        traced = pipeline.trace(image, analyze_layout=analyze_layout)
    except RuntimeError as e:
        # pipeline._require_tesseract() raises this if the engine isn't found
        raise HTTPException(503, str(e)) from e

    result = traced["result"]
    doc_id = str(uuid.uuid4())

    extraction = llm_extract.extract(result["text"], model=model) if extract else None
    summary = result["text"]
    if extraction and extraction.data.vendor:
        summary = f"{extraction.data.vendor}\n{summary}"
    store.add(session_id, doc_id, text=result["text"], summary=summary)

    return DocumentResult(
        id=doc_id,
        text=result["text"],
        word_count=result["word_count"],
        mean_confidence=result["mean_confidence"],
        psm_used=result["psm_used"],
        fields=result["fields"],
        regions=[Region(**r) for r in result["regions"]],
        word_boxes=[WordBox(**w) for w in result["word_boxes"]],
        stages=_build_stages(traced),
        metrics=result["metrics"],
        extraction=extraction,
    )


@router.post("/v1/documents/{document_id}/ask")
async def ask_document(
    document_id: str,
    body: AskRequest,
    session_id: str = Depends(get_session_id),
    store=Depends(get_document_store),
):
    doc = store.get(session_id, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found in this session")
    try:
        return llm_qa.ask(doc.text, body.question, model=body.model)
    except llm_qa.NotConfiguredError as e:
        raise HTTPException(503, str(e)) from e
    except llm_qa.LLMCallError as e:
        raise HTTPException(502, str(e)) from e

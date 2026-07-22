"""GET /v1/sample-document — generates a synthetic invoice PNG server-side
(reusing the same generator the evaluation harness uses under
scripts/make_invoice_dataset.py), so the live demo works for a visitor who
doesn't have a real invoice image handy."""

import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Query, Response

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import make_invoice_dataset as mk  # noqa: E402

router = APIRouter(tags=["samples"])


@router.get("/v1/sample-document")
async def sample_document(degrade: str = Query("scan", pattern="^(clean|scan|heavy)$")):
    rng = np.random.default_rng()
    img, _ = mk.render_invoice(mk.build_invoice(rng))
    img = mk.degrade(img, rng, level=degrade)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")

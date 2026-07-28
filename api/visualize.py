"""Render pipeline trace output into base64-PNG stage images.

The drawing logic (region-type boxes, confidence-coloured word boxes) lives
here, server-side, so the web client never needs OpenCV of its own — it just
displays PNG bytes the API already produced.
"""

import base64

import cv2
import numpy as np

_REGION_COLORS = {
    "header_footer": (122, 127, 135), "table": (184, 134, 11),
    "text": (43, 75, 126), "image": (125, 91, 166), "unknown": (150, 150, 150),
}


def fit(img: np.ndarray, max_w: int = 760) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def _conf_color(c: float):
    return (47, 125, 91) if c >= 75 else (184, 134, 11) if c >= 50 else (176, 74, 74)


def scale_boxes(items: list[dict], from_shape, to_shape) -> list[dict]:
    """Rescale bbox tuples from one image size to another (e.g. native
    processed-image coordinates -> the smaller image sent to the client)."""
    s = to_shape[1] / from_shape[1]
    return [{**it, "bbox": tuple(int(v * s) for v in it["bbox"])} for it in items]


def draw_regions(gray: np.ndarray, regions: list[dict]) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    for r in regions:
        x, y, w, h = r["bbox"]
        color = _REGION_COLORS.get(r["type"], (150, 150, 150))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    return vis


def draw_word_boxes(gray: np.ndarray, boxes: list[dict]) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    for b in boxes:
        x, y, w, h = b["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), _conf_color(b["conf"]), 1)
    return vis


def encode_jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    """JPEG, not PNG: these are display-only trace images (never fed back
    into the pipeline), and PNG's lossless DEFLATE compresses noisy/degraded
    scan content badly — 9 stages of "scan"-degraded PNGs blew past AWS
    Lambda's 6MB response payload limit (measured ~6.8MB) even after each
    image was already downscaled by fit(). JPEG q85 cuts that by ~80%.
    cv2.imencode handles both 2-D grayscale and 3-D BGR arrays."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

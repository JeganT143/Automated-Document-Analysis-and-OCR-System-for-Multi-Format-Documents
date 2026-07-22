"""Thin HTTP client for the Document OCR + AI Pipeline API.

Deliberately dependency-light (just `requests`) — this module is the entire
boundary between the Streamlit UI and the API, so the UI never needs
opencv/pytesseract/openai/fastembed of its own.
"""

import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60


class APIError(RuntimeError):
    def __init__(self, status_code, detail):
        super().__init__(f"[{status_code}] {detail}")
        self.status_code = status_code
        self.detail = detail


def _headers(session_id: str) -> dict:
    return {"X-Session-Id": session_id}


def _handle(resp: requests.Response):
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise APIError(resp.status_code, detail)
    return resp.json()


def is_healthy() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def list_models() -> dict:
    resp = requests.get(f"{API_BASE_URL}/v1/models", timeout=TIMEOUT)
    return _handle(resp)


def sample_document(degrade: str = "scan") -> bytes:
    resp = requests.get(f"{API_BASE_URL}/v1/sample-document",
                        params={"degrade": degrade}, timeout=TIMEOUT)
    if resp.status_code >= 400:
        raise APIError(resp.status_code, resp.text)
    return resp.content


def process_document(session_id: str, file_bytes: bytes, filename: str,
                      analyze_layout: bool = True, extract: bool = False,
                      model: str | None = None) -> dict:
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data = {"analyze_layout": str(analyze_layout).lower(),
            "extract": str(extract).lower()}
    if model:
        data["model"] = model
    resp = requests.post(f"{API_BASE_URL}/v1/documents", files=files, data=data,
                         headers=_headers(session_id), timeout=TIMEOUT)
    return _handle(resp)


def ask_document(session_id: str, document_id: str, question: str,
                  model: str | None = None) -> dict:
    payload = {"question": question, "model": model}
    resp = requests.post(f"{API_BASE_URL}/v1/documents/{document_id}/ask", json=payload,
                         headers=_headers(session_id), timeout=TIMEOUT)
    return _handle(resp)


def search_documents(session_id: str, query: str, model: str | None = None,
                      top_k: int = 3) -> dict:
    payload = {"query": query, "model": model, "top_k": top_k}
    resp = requests.post(f"{API_BASE_URL}/v1/search", json=payload,
                         headers=_headers(session_id), timeout=TIMEOUT)
    return _handle(resp)

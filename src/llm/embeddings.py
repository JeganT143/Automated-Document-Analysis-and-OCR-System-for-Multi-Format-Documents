"""Lazy-loaded local embedding model for cross-document search.

Uses fastembed (ONNX runtime, CPU-only, no torch) rather than an embeddings
API: it's free per call, has no network dependency once the model is cached
on disk, and keeps semantic search working even without an OpenRouter key —
only *answer generation* needs the LLM, retrieval doesn't.

The model is loaded on first use, not at import time, so plain OCR /
extraction requests never pay its ~1-2s load cost.
"""

import os
import tempfile
from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# fastembed/huggingface_hub default to caching under ~/.cache/huggingface,
# but AWS Lambda's filesystem is read-only outside /tmp — HOME there points
# somewhere unwritable, so the default cache 500s on first embed call.
# Passing cache_dir= to TextEmbedding below isn't enough on its own: the
# newer "xet" download accelerator inside huggingface_hub reads HF_HOME
# directly for its own state/locks regardless of that argument. Setting
# HF_HOME here — before fastembed/huggingface_hub are imported anywhere —
# covers both. tempfile.gettempdir() is /tmp everywhere that matters
# (Lambda, Docker, local dev), so this is a no-downside fix, not a
# Lambda-only workaround.
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fastembed_cache")
os.environ.setdefault("HF_HOME", os.path.join(tempfile.gettempdir(), "hf_home"))


@lru_cache
def _model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=_CACHE_DIR)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns an (n, dim) float32 array, L2-normalised
    so a dot product between rows is a cosine similarity."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    vecs = np.array(list(_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]

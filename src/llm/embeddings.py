"""Lazy-loaded local embedding model for cross-document search.

Uses fastembed (ONNX runtime, CPU-only, no torch) rather than an embeddings
API: it's free per call, has no network dependency once the model is cached
on disk, and keeps semantic search working even without an OpenRouter key —
only *answer generation* needs the LLM, retrieval doesn't.

The model is loaded on first use, not at import time, so plain OCR /
extraction requests never pay its ~1-2s load cost.
"""

from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@lru_cache
def _model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=MODEL_NAME)


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

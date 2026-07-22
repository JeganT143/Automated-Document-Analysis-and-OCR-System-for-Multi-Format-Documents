"""In-memory, session-partitioned document store + cross-document search.

Deliberately session-scoped and ephemeral: Cloud Run can recycle an instance
between requests (scale-to-zero, multiple replicas), so this is documented
as "documents processed in your current session on this server instance"
rather than a persistent store — that's an explicit simplicity tradeoff for
a portfolio-scale demo. At real scale this would move to pgvector/Qdrant
keyed by an authenticated user id, not a bare session id.

Session partitioning matters for correctness, not just tidiness: without it,
one visitor's uploaded invoice would be searchable/citable by a different
concurrent visitor.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from . import client, qa
from .embeddings import embed_one
from .schemas import SearchAnswer, SearchHit

MAX_SESSIONS = 50
MAX_DOCS_PER_SESSION = 20


@dataclass
class StoredDocument:
    document_id: str
    text: str
    summary: str  # text actually embedded + shown as a search snippet
    embedding: np.ndarray
    created_at: float = field(default_factory=time.time)


class DocumentStore:
    """Thread-safe, LRU-bounded, session-partitioned in-memory store."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, OrderedDict[str, StoredDocument]] = OrderedDict()

    def add(self, session_id: str, document_id: str, text: str,
            summary: str | None = None) -> None:
        summary = (summary or text)[:2000]
        vec = embed_one(summary)
        with self._lock:
            session = self._sessions.setdefault(session_id, OrderedDict())
            self._sessions.move_to_end(session_id)
            session[document_id] = StoredDocument(document_id, text, summary, vec)
            session.move_to_end(document_id)
            while len(session) > MAX_DOCS_PER_SESSION:
                session.popitem(last=False)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    def get(self, session_id: str, document_id: str) -> StoredDocument | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return session.get(document_id) if session else None

    def search(self, session_id: str, query: str, top_k: int = 3) -> list[SearchHit]:
        with self._lock:
            session = self._sessions.get(session_id)
            docs = list(session.values()) if session else []
        if not docs:
            return []
        query_vec = embed_one(query)
        scored = [(float(np.dot(query_vec, d.embedding)), d) for d in docs]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            SearchHit(document_id=d.document_id, score=round(score, 4),
                      snippet=d.summary[:280])
            for score, d in scored[:top_k]
        ]


def answer(store: DocumentStore, session_id: str, query: str,
           model: str | None = None, top_k: int = 3) -> SearchAnswer:
    """Retrieve the most relevant documents in this session and generate a
    grounded, cited answer over them."""
    hits = store.search(session_id, query, top_k=top_k)
    if not hits:
        return SearchAnswer(
            answer="No documents in this session yet — process one first.", hits=[])

    if not client.is_configured():
        listing = "; ".join(f"{h.document_id} (score {h.score})" for h in hits)
        return SearchAnswer(
            answer=f"OPENROUTER_API_KEY not configured — closest matches: {listing}",
            hits=hits)

    context = "\n\n".join(f"[{h.document_id}]\n{h.snippet}" for h in hits)
    qa_result = qa.ask(context, query, model=model)
    return SearchAnswer(answer=qa_result.answer, hits=hits, model=qa_result.model)

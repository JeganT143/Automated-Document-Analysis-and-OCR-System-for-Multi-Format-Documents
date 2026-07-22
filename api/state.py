"""Process-wide singleton: the session-partitioned document store shared by
the documents and search routers. See src/llm/search.py for why this is
in-memory and session-scoped rather than persistent."""

from src.llm.search import DocumentStore

document_store = DocumentStore()

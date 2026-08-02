"""
ChromaDB persistent vector store.

One collection ("papers") holds all chunks; each chunk keeps its doc_id/page
metadata so we can filter by document and cite pages. We expose thin helpers for
adding a parsed document, similarity search, and listing/deleting docs.
"""
from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from backend.core.embeddings import get_embeddings
from backend.core.parsing import ParsedDoc

COLLECTION = "papers"

_store: Chroma | None = None


def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(settings.chroma_dir),
        )
    return _store


def add_document(doc: ParsedDoc) -> int:
    """Embed and persist all chunks of a parsed document. Returns #chunks."""
    store = get_store()
    docs = [
        Document(page_content=c.text, metadata=c.metadata) for c in doc.chunks
    ]
    ids = [f"{doc.doc_id}:{i}" for i in range(len(docs))]
    if docs:
        store.add_documents(docs, ids=ids)
    return len(docs)


def similarity(query: str, k: int, doc_ids: list[str] | None = None) -> list[Document]:
    store = get_store()
    flt: dict[str, Any] | None = None
    if doc_ids:
        flt = {"doc_id": {"$in": doc_ids}}
    return store.similarity_search(query, k=k, filter=flt)


def all_documents_for(doc_ids: list[str] | None = None) -> list[Document]:
    """Fetch raw chunk docs (used to build the BM25 index)."""
    store = get_store()
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    got = store.get(where=where, include=["documents", "metadatas"])
    out: list[Document] = []
    for text, meta in zip(got["documents"], got["metadatas"]):
        out.append(Document(page_content=text, metadata=meta or {}))
    return out


def list_doc_ids() -> list[str]:
    store = get_store()
    got = store.get(include=["metadatas"])
    seen = {}
    for meta in got["metadatas"]:
        if meta and "doc_id" in meta:
            seen[meta["doc_id"]] = meta.get("filename", meta["doc_id"])
    return [{"doc_id": k, "filename": v} for k, v in seen.items()]


def delete_document(doc_id: str) -> None:
    store = get_store()
    store.delete(where={"doc_id": doc_id})

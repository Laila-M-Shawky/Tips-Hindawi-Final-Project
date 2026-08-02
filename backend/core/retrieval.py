"""
Hybrid retrieval + reranking.

Pipeline:
    1. Dense retrieval  (BGE embeddings via Chroma)
    2. Sparse retrieval (BM25 over the same chunk set)
    3. Reciprocal Rank Fusion (RRF) to merge the two ranked lists
    4. BGE cross-encoder reranker for final precision ordering

Sparse + dense catch complementary things: dense handles paraphrase/semantics,
BM25 nails exact terms (author names, dataset names, equation symbols). RRF is a
robust, tuning-light way to fuse them. The reranker then re-scores the top
candidates with a cross-encoder that sees query+passage jointly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import settings
from backend.core import vectorstore


@dataclass
class Retrieved:
    text: str
    metadata: dict
    score: float

    @property
    def citation(self) -> str:
        fn = self.metadata.get("filename", "?")
        pg = self.metadata.get("page", "?")
        return f"[{fn} p.{pg}]"


# ---------------------------------------------------------------------------
# BM25 index (rebuilt when the corpus changes)
# ---------------------------------------------------------------------------
def _tok(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


@lru_cache(maxsize=8)
def _bm25_index(doc_key: str):
    """Build a BM25 index over the chunks of the given docs.

    doc_key is a stable string of doc_ids so we can cache per corpus slice.
    """
    doc_ids = doc_key.split("|") if doc_key else None
    corpus = vectorstore.all_documents_for(doc_ids)
    tokenized = [_tok(d.page_content) for d in corpus]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return bm25, corpus


def _bm25_search(query: str, k: int, doc_ids: list[str] | None) -> list[tuple[Document, float]]:
    key = "|".join(sorted(doc_ids)) if doc_ids else ""
    bm25, corpus = _bm25_index(key)
    if not bm25:
        return []
    scores = bm25.get_scores(_tok(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def bust_bm25_cache() -> None:
    """Call after ingesting new documents so BM25 re-reads the corpus."""
    _bm25_index.cache_clear()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _rrf(ranklists: list[list[Document]], k: int) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    keep: dict[str, Document] = {}
    for ranklist in ranklists:
        for rank, doc in enumerate(ranklist):
            key = _doc_key(doc)
            keep[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(keep[key], sc) for key, sc in fused[:k]]


def _doc_key(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('doc_id')}:{m.get('page')}:{hash(doc.page_content[:80])}"


# ---------------------------------------------------------------------------
# BGE reranker (cross-encoder)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, max_length=512,
                        device=settings.embed_device)


def _rerank(query: str, docs: list[Document], top_k: int) -> list[Retrieved]:
    if not docs:
        return []
    ce = _reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [Retrieved(d.page_content, d.metadata, float(s)) for d, s in ranked]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def hybrid_retrieve(
    query: str,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
    rerank: bool = True,
) -> list[Retrieved]:
    top_k = top_k or settings.rerank_top_k

    dense = vectorstore.similarity(query, settings.dense_top_k, doc_ids)
    sparse = [d for d, _ in _bm25_search(query, settings.bm25_top_k, doc_ids)]

    fused = _rrf([dense, sparse], k=max(settings.dense_top_k, settings.bm25_top_k))
    fused_docs = [d for d, _ in fused]

    if rerank:
        return _rerank(query, fused_docs, top_k)
    return [Retrieved(d.page_content, d.metadata, sc) for d, sc in fused[:top_k]]


def fuse_multi(
    ranklists: list[list[Retrieved]], query: str, top_k: int | None = None
) -> list[Retrieved]:
    """Fuse results from several sub-queries (multi-query retrieval) then rerank."""
    top_k = top_k or settings.rerank_top_k
    as_docs = [
        [Document(page_content=r.text, metadata=r.metadata) for r in rl]
        for rl in ranklists
    ]
    fused = _rrf(as_docs, k=settings.dense_top_k)
    return _rerank(query, [d for d, _ in fused], top_k)

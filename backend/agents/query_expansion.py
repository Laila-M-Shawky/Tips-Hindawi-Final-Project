"""
Query expansion + multi-query retrieval.

A single user query often misses relevant chunks because of vocabulary mismatch.
We ask the LLM to generate N diverse rephrasings (different terminology, broader
and narrower framings), retrieve for each, then fuse the ranked lists with RRF
and rerank. This meaningfully lifts recall on academic corpora.
"""
from __future__ import annotations

from config.settings import settings
from backend.core import retrieval
from backend.core.llm import structured_json
from backend.core.retrieval import Retrieved

_EXPAND_SYS = (
    "You expand a research question into diverse search queries that surface "
    "different but relevant passages. Vary the terminology and specificity."
)


def expand(query: str, n: int | None = None) -> list[str]:
    n = n or settings.multi_query_n
    out = structured_json(
        _EXPAND_SYS,
        f'Question: "{query}"\nReturn a JSON list of {n} alternative search '
        f"queries (strings). Include the original intent, reworded.",
    )
    queries = [query]
    if isinstance(out, list):
        queries += [str(q) for q in out if isinstance(q, (str, int, float))]
    # dedupe, keep order
    seen, uniq = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq[: n + 1]


def multi_query_retrieve(
    query: str, doc_ids: list[str] | None = None, top_k: int | None = None
) -> tuple[list[Retrieved], list[str]]:
    subqs = expand(query)
    ranklists = [
        retrieval.hybrid_retrieve(q, doc_ids=doc_ids, rerank=False) for q in subqs
    ]
    fused = retrieval.fuse_multi(ranklists, query=query, top_k=top_k)
    return fused, subqs

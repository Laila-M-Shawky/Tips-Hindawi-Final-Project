"""
Paper comparison.

Builds a per-paper digest (problem, method, data, results, limitations) from each
document's own chunks, then asks the model to produce a comparison matrix plus a
short verdict. Works on 2+ uploaded papers.
"""
from __future__ import annotations

from typing import Any

from backend.core import retrieval, vectorstore
from backend.core.llm import chat, structured_json

_DIGEST_SYS = (
    "Summarize one paper into JSON with keys: title, problem, method, datasets, "
    "key_results, limitations. Use only the given excerpts."
)
_VERDICT_SYS = (
    "You're given digests of two or more papers. Write a short paragraph (3-5 "
    "sentences) on the key trade-offs between them — where they agree, where "
    "they differ, and which is better suited to what. Plain prose, no JSON."
)

_ASPECTS = ["problem statement", "method and architecture", "datasets and metrics",
            "main results", "limitations"]
# maps each aspect to the digest key the model is asked to fill for it
_DIMENSIONS = [
    ("Problem statement", "problem"),
    ("Method / architecture", "method"),
    ("Datasets & metrics", "datasets"),
    ("Main results", "key_results"),
    ("Limitations", "limitations"),
]


def _digest(doc_id: str) -> dict[str, Any]:
    chunks = []
    for a in _ASPECTS:
        chunks += retrieval.hybrid_retrieve(a, doc_ids=[doc_id], top_k=2)
    excerpts = "\n\n".join(c.text for c in chunks[:8])
    d = structured_json(_DIGEST_SYS, f"Excerpts:\n{excerpts}")
    if not isinstance(d, dict):
        d = {}
    d.setdefault("title", doc_id)
    for _, key in _DIMENSIONS:
        d.setdefault(key, "not reported")
    return d


def compare(doc_ids: list[str] | None = None) -> dict[str, Any]:
    if not doc_ids:
        doc_ids = [d["doc_id"] for d in vectorstore.list_doc_ids()]
    if len(doc_ids) < 2:
        return {"error": "Need at least two documents to compare."}
    digests = [_digest(d) for d in doc_ids[:4]]

    # Build the matrix ourselves from the digests we already trust, rather than
    # asking a fast/small model to freely construct nested JSON in one shot —
    # that's exactly the kind of complex schema smaller models drop fields on
    # (observed: it returned bare {"title": ...} with nothing else).
    matrix = [
        {"dimension": label, "per_paper": {d["title"]: d.get(key, "not reported")
                                            for d in digests}}
        for label, key in _DIMENSIONS
    ]
    verdict = chat(_VERDICT_SYS, "Digests:\n" + "\n\n".join(str(d) for d in digests))

    return {"digests": digests, "comparison": {"matrix": matrix, "verdict": verdict}}

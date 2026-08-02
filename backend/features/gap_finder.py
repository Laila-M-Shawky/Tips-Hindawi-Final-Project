"""
Research-gap finder.

Targets the parts of papers most likely to reveal gaps (limitations, future
work, discussion) via focused sub-queries, then asks the model to extract
concrete, actionable gaps with the evidence that motivates each.
"""
from __future__ import annotations

from typing import Any

from backend.core import retrieval
from backend.core.llm import format_context, structured_json

_GAP_QUERIES = [
    "limitations of the approach",
    "future work and open problems",
    "unaddressed challenges and assumptions",
    "threats to validity",
]

_SYS = (
    "You identify research gaps from paper excerpts. For each gap return an "
    "object with keys: gap, why_it_matters, evidence (cite [n]), suggested_direction. "
    'Return JSON: {"gaps": [ ... ]}. Be specific and non-generic.'
)


def find(topic: str, doc_ids: list[str] | None = None) -> dict[str, Any]:
    seen, chunks = set(), []
    queries = [topic] + [f"{topic}: {q}" for q in _GAP_QUERIES]
    for q in queries:
        for r in retrieval.hybrid_retrieve(q, doc_ids=doc_ids, top_k=4):
            key = (r.metadata.get("doc_id"), r.metadata.get("page"), r.text[:60])
            if key not in seen:
                seen.add(key)
                chunks.append(r)
    chunks = chunks[:12]
    context = format_context(chunks)
    out = structured_json(_SYS, f"Topic: {topic}\n\nExcerpts:\n{context}")
    gaps = out.get("gaps", []) if isinstance(out, dict) else []
    return {"gaps": gaps, "n_sources": len(chunks)}

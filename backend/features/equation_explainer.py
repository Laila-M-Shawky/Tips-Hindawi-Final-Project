"""
Equation explanation.

Finds equation-bearing chunks (parsing tagged them with has_equation) relevant to
the user's request and asks the model to explain notation, intuition, and role in
the method, term by term. Cites the page the equation came from.
"""
from __future__ import annotations

from typing import Any

from backend.core import retrieval
from backend.core.llm import chat, format_context

_SYS = (
    "You explain mathematical notation from a paper for a graduate student. For "
    "each relevant equation: state what it computes in one line, define every "
    "symbol, give the intuition, and explain its role in the method. Cite [n]. "
    "Render math in LaTeX between $...$ so it displays cleanly."
)


def explain(request: str, doc_ids: list[str] | None = None) -> dict[str, Any]:
    hits = retrieval.hybrid_retrieve(request, doc_ids=doc_ids, top_k=12,
                                     rerank=True)
    eq_chunks = [h for h in hits if h.metadata.get("has_equation")] or hits[:4]
    eq_chunks = eq_chunks[:5]
    context = format_context(eq_chunks)
    explanation = chat(_SYS, f"Request: {request}\n\nExcerpts:\n{context}")
    cites = [{"n": i + 1, "source": c.metadata.get("filename"),
              "page": c.metadata.get("page")} for i, c in enumerate(eq_chunks)]
    return {"explanation": explanation, "citations": cites}

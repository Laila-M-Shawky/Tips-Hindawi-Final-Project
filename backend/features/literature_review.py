"""
Literature review generator.

Retrieves broadly across the corpus for the topic, then asks the model to
synthesize a structured review (themes, consensus, disagreements, methods) with
inline [n] citations mapped back to pages.
"""
from __future__ import annotations

from typing import Any

from backend.agents.query_expansion import multi_query_retrieve
from backend.core.llm import chat, format_context

_SYS = (
    "You are writing the Related Work section of a paper. Synthesize the sources "
    "into a coherent narrative organized by theme (not one-source-per-paragraph). "
    "Compare and contrast findings, note consensus and disagreement, and cite "
    "every claim with [n]. End with a short 'Open questions' paragraph."
)


def generate(topic: str, doc_ids: list[str] | None = None) -> dict[str, Any]:
    chunks, subqs = multi_query_retrieve(topic, doc_ids=doc_ids, top_k=10)
    context = format_context(chunks)
    review = chat(_SYS, f"Topic: {topic}\n\nSources:\n{context}")
    cites = [{"n": i + 1, "source": c.metadata.get("filename"),
              "page": c.metadata.get("page")} for i, c in enumerate(chunks)]
    return {"review": review, "citations": cites, "sub_queries": subqs}

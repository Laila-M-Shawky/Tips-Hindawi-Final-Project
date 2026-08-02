"""
Methodology generator.

Given a research topic/question, drafts a rigorous methodology grounded where
possible in methods seen in the uploaded papers (so it recommends techniques the
corpus actually supports, with citations), plus generic best practices.
"""
from __future__ import annotations

from typing import Any

from backend.agents.query_expansion import multi_query_retrieve
from backend.core.llm import chat, format_context

_SYS = (
    "You are a research methodologist. Propose a concrete methodology with "
    "sections: research questions, data collection, preprocessing, model/approach, "
    "baselines, evaluation metrics, ablations, and threats to validity. Where a "
    "choice is supported by the provided papers, cite it [n]; otherwise mark it "
    "as (general best practice)."
)


def generate(topic: str, doc_ids: list[str] | None = None) -> dict[str, Any]:
    chunks, _ = multi_query_retrieve(f"methodology for {topic}", doc_ids=doc_ids,
                                     top_k=8)
    context = format_context(chunks)
    plan = chat(_SYS, f"Topic: {topic}\n\nRelevant methods from papers:\n{context}")
    cites = [{"n": i + 1, "source": c.metadata.get("filename"),
              "page": c.metadata.get("page")} for i, c in enumerate(chunks)]
    return {"methodology": plan, "citations": cites}

"""
Grounded Q&A with page citations — the flagship feature.

Delegates to Self-RAG (multi-query retrieval -> relevance grading -> grounded
generation -> groundedness check -> optional retry), then runs an independent
citation-verification pass so the UI can flag any claim that isn't supported.
"""
from __future__ import annotations

from typing import Any

from backend.agents import self_rag
from backend.features.citation_verify import verify


def grounded_qa(question: str, doc_ids: list[str] | None = None) -> dict[str, Any]:
    result = self_rag.answer(question, doc_ids=doc_ids)
    verification = verify(result["answer"], result["citations"])
    result["verification"] = verification
    return result

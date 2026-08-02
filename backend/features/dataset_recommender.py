"""
Dataset recommender.

Two signals combined:
  1) Datasets *mentioned in the uploaded papers* (grounded, citable).
  2) The model's knowledge of standard public benchmarks for the task.
Returns structured recommendations with rationale and (when grounded) citations.
"""
from __future__ import annotations

from typing import Any

from backend.core import retrieval
from backend.core.llm import format_context, structured_json

_SYS = (
    "Recommend datasets for the user's task. Prefer datasets that appear in the "
    "provided excerpts (cite [n]); you may add well-known public benchmarks marked "
    'source:"general". Return JSON: {"datasets":[{name, description, size, '
    'modality, why_relevant, access, citation}]}.'
)


def recommend(task: str, doc_ids: list[str] | None = None) -> dict[str, Any]:
    chunks = retrieval.hybrid_retrieve(f"datasets benchmarks used for {task}",
                                       doc_ids=doc_ids, top_k=6)
    context = format_context(chunks)
    out = structured_json(_SYS, f"Task: {task}\n\nExcerpts:\n{context}")
    datasets = out.get("datasets", []) if isinstance(out, dict) else []
    return {"datasets": datasets, "n_sources": len(chunks)}

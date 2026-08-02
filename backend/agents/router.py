"""
Agent router.

Given a free-text request, classify the intent and dispatch to the right feature
module. We use an LLM classifier with a constrained label set (reliable + cheap),
plus keyword fallbacks so the demo never dead-ends if the model returns junk.

This is the "agentic" front door: one chat box, many tools.
"""
from __future__ import annotations

from typing import Any, Callable

from backend.core.llm import structured_json

# label -> (human description, handler import path resolved lazily)
INTENTS = {
    "qa": "Answer a specific question grounded in the papers with citations. "
          "Use this for any general or factual question about the paper's "
          "content, subject, or findings, and whenever no other tool clearly fits.",
    "literature_review": "Synthesize an organized literature review across papers.",
    "gap_finder": "Identify open problems / research gaps and future work.",
    "comparison": "Compare two or more papers across dimensions.",
    "methodology": "Draft a methodology/experimental design for a topic.",
    "dataset_recommender": "Recommend datasets suitable for a task.",
    "citation_generator": "Generate a formatted citation (BibTeX/APA/IEEE).",
    "equation_explainer": "Explain an equation in the paper.",
    "timeline": "Build a chronological timeline of the field/ideas.",
    "knowledge_graph": "Extract an entity-relation knowledge graph.",
}

_ROUTER_SYS = (
    "You route a user's request to exactly one tool. Tools:\n"
    + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    + '\nReturn JSON: {"intent": "<one label>", "confidence": 0-1, "reason": "..."}.'
)

_KEYWORDS = {
    "literature_review": ["literature review", "survey", "synthesize", "overview of"],
    "gap_finder": ["gap", "open problem", "future work", "unexplored", "limitation"],
    "comparison": ["compare", "versus", " vs ", "difference between"],
    "methodology": ["methodology", "experimental design", "how should i", "method for"],
    "dataset_recommender": ["dataset", "data set", "benchmark", "corpus"],
    "citation_generator": ["cite", "citation", "bibtex", "apa", "ieee", "reference"],
    "equation_explainer": ["equation", "formula", "derive", "notation"],
    "timeline": ["timeline", "chronolog", "evolution", "history of"],
    "knowledge_graph": ["knowledge graph", "entities", "relations", "graph of"],
}


_MIN_LLM_CONFIDENCE = 0.3


def route(request: str) -> dict[str, Any]:
    low = request.lower()
    for intent, kws in _KEYWORDS.items():
        if any(kw in low for kw in kws):
            return {"intent": intent, "confidence": 0.6, "reason": "keyword match",
                    "via": "rules"}

    verdict = structured_json(_ROUTER_SYS, f"Request: {request}")
    confidence = _as_float(verdict.get("confidence")) if isinstance(verdict, dict) else 0.0
    if isinstance(verdict, dict) and verdict.get("intent") in INTENTS \
            and confidence >= _MIN_LLM_CONFIDENCE:
        verdict["via"] = "llm"
        return verdict
    # A low/zero-confidence or malformed classification is worse than just
    # answering the question — fall back to qa instead of dispatching to
    # whatever label the model happened to emit (e.g. equation_explainer for
    # an unrelated general question).
    return {"intent": "qa", "confidence": 0.4,
            "reason": "fallback (router had no confident match)", "via": "default"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def dispatch(request: str, doc_ids: list[str] | None = None) -> dict[str, Any]:
    """Route then call the chosen feature. Imports are local to avoid cycles."""
    decision = route(request)
    intent = decision["intent"]
    handler = _resolve(intent)
    payload = handler(request, doc_ids)
    return {"routing": decision, "intent": intent, "result": payload}


def _resolve(intent: str) -> Callable[[str, list[str] | None], Any]:
    from backend.features import (
        qa, literature_review, gap_finder, comparison, methodology,
        dataset_recommender, citation_generator, equation_explainer, timeline,
        knowledge_graph,
    )
    table = {
        "qa": lambda r, d: qa.grounded_qa(r, d),
        "literature_review": lambda r, d: literature_review.generate(r, d),
        "gap_finder": lambda r, d: gap_finder.find(r, d),
        "comparison": lambda r, d: comparison.compare(d),
        "methodology": lambda r, d: methodology.generate(r, d),
        "dataset_recommender": lambda r, d: dataset_recommender.recommend(r, d),
        "citation_generator": lambda r, d: citation_generator.generate(d),
        "equation_explainer": lambda r, d: equation_explainer.explain(r, d),
        "timeline": lambda r, d: timeline.build(d),
        "knowledge_graph": lambda r, d: knowledge_graph.build(d),
    }
    return table[intent]

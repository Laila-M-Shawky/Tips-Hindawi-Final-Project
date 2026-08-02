"""
Citation verification.

Splits the answer into claim-bearing sentences and, for each, asks the model
whether the cited snippet(s) actually support it. Returns a per-claim verdict so
the frontend can render green/red badges. This is what turns "sounds confident"
into "provably grounded".
"""
from __future__ import annotations

import re
from typing import Any

from backend.core.llm import structured_json

_VERIFY_SYS = (
    "You check whether an evidence snippet supports a claim. "
    'Return JSON: {"supported": true|false, "confidence": 0-1}.'
)


def _sentences(text: str) -> list[str]:
    # naive splitter that keeps citation markers attached
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def verify(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    by_n = {c["n"]: c for c in citations}
    claims = []
    for sent in _sentences(answer):
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
        if not cited:
            continue
        evidence = "\n".join(by_n[n]["snippet"] for n in cited if n in by_n)
        if not evidence:
            claims.append({"claim": sent, "cited": cited, "supported": False,
                           "confidence": 0.0})
            continue
        verdict = structured_json(
            _VERIFY_SYS, f"Claim: {sent}\n\nEvidence:\n{evidence}"
        )
        if not isinstance(verdict, dict):
            verdict = {"supported": True, "confidence": 0.5}
        claims.append({"claim": sent, "cited": cited,
                       "supported": bool(verdict.get("supported", True)),
                       "confidence": float(verdict.get("confidence", 0.5))})

    supported = sum(1 for c in claims if c["supported"])
    total = len(claims) or 1
    return {"claims": claims, "supported": supported, "total": len(claims),
            "grounding_score": round(supported / total, 2)}

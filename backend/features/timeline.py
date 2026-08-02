"""
Timeline generation.

Scans the corpus for year mentions and their surrounding context, then asks the
model to assemble a de-duplicated chronological timeline of milestones/ideas.
Useful for "evolution of X" narratives and for slides.
"""
from __future__ import annotations

from typing import Any

from backend.core import parsing, vectorstore
from backend.core.llm import structured_json

_SYS = (
    "Build a chronological research timeline from excerpts that mention years. "
    'Return JSON: {"timeline":[{year, milestone, detail, source}]} sorted by year. '
    "Merge duplicates; keep only substantive milestones."
)


def build(doc_ids: list[str] | None = None) -> dict[str, Any]:
    docs = vectorstore.all_documents_for(doc_ids)
    dated = []
    for d in docs:
        years = list(parsing.iter_years(d.page_content))
        if years:
            dated.append({"years": sorted(set(years)),
                          "text": d.page_content[:400],
                          "source": d.metadata.get("filename")})
    # Keep the prompt well under the free-tier per-request token cap — this cap
    # was fine for one paper but a growing corpus (more year-mentioning chunks,
    # especially reference lists) can push a 40-excerpt payload over the limit.
    dated = dated[:15]
    payload = "\n\n".join(f"({e['source']}, years {e['years']}): {e['text']}"
                          for e in dated)
    out = structured_json(_SYS, f"Excerpts:\n{payload}")
    timeline = out.get("timeline", []) if isinstance(out, dict) else []
    return {"timeline": timeline, "n_dated_excerpts": len(dated)}

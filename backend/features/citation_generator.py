"""
Citation generator.

Pulls the front-matter of each paper (title/authors/venue/year live on the first
page) and formats citations in BibTeX, APA, and IEEE. Grounded in the actual PDF
text rather than hallucinated metadata.
"""
from __future__ import annotations

from typing import Any

from backend.core import vectorstore
from backend.core.llm import structured_json

_SYS = (
    "Extract bibliographic metadata from the first-page text of a paper and format "
    "citations. Return JSON: {title, authors:[...], year, venue, bibtex, apa, ieee}. "
    "If a field is missing, infer conservatively or use 'n.d.'."
)


def _front_matter(doc_id: str) -> str:
    # Pull page-1 chunks directly by metadata rather than via embedding search —
    # a semantic query for "title authors abstract" can miss page 1 on papers
    # where the abstract reads more like body text, silently falling back to
    # whatever chunk it did find (e.g. a references page) and producing garbage.
    chunks = vectorstore.all_documents_for([doc_id])
    page1 = [c.page_content for c in chunks if c.metadata.get("page") == 1]
    if not page1 and chunks:
        page1 = [chunks[0].page_content]
    return "\n".join(page1)[:1500]


def generate(doc_ids: list[str] | None = None) -> dict[str, Any]:
    if not doc_ids:
        doc_ids = [d["doc_id"] for d in vectorstore.list_doc_ids()]
    results = []
    for d in doc_ids:
        meta = structured_json(_SYS, f"First-page text:\n{_front_matter(d)}")
        if isinstance(meta, dict):
            meta["doc_id"] = d
            results.append(meta)
    return {"citations": results}

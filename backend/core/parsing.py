"""
PDF parsing with PyMuPDF (fitz).

Design goal: every chunk carries the page number and source filename so that
downstream Q&A can produce *grounded citations* like  [paper.pdf p.4].
We also do light structure detection (section headings, equations, references)
so feature modules (equation explainer, citation generator, timeline) have
signal to work with.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

# Rough regexes — good enough for signal, not meant to be a full grammar.
_SECTION_RE = re.compile(r"^\s*(\d+(\.\d+)*)?\s*(abstract|introduction|related work|"
                         r"background|method(ology)?|approach|experiments?|results?|"
                         r"discussion|conclusion|references|appendix)\b",
                         re.IGNORECASE)
_EQUATION_HINT = re.compile(r"[=∑∫√≈≤≥∈∇∂μσλθ]|\\frac|\\sum|\\int")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDoc:
    doc_id: str
    filename: str
    n_pages: int
    pages: list[str]                 # raw text per page (index 0 == page 1)
    chunks: list[Chunk]
    title: str = ""
    has_equations: bool = False


def _detect_title(pages: list[str]) -> str:
    """Heuristic: first non-trivial line of page 1."""
    if not pages:
        return ""
    for line in pages[0].splitlines():
        line = line.strip()
        if len(line) > 8 and not line.lower().startswith(("http", "www", "doi")):
            return line[:200]
    return ""


def parse_pdf(path: str | Path, doc_id: str | None = None) -> ParsedDoc:
    path = Path(path)
    doc_id = doc_id or path.stem
    pages: list[str] = []

    with fitz.open(path) as pdf:
        for page in pdf:
            # "text" preserves reading order reasonably well for research papers.
            pages.append(page.get_text("text"))

    title = _detect_title(pages)
    has_eq = any(_EQUATION_HINT.search(p) for p in pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page_idx, page_text in enumerate(pages, start=1):
        page_text = _clean(page_text)
        if not page_text.strip():
            continue
        current_section = _current_section(page_text)
        for piece in splitter.split_text(page_text):
            chunks.append(
                Chunk(
                    text=piece,
                    metadata={
                        "doc_id": doc_id,
                        "filename": path.name,
                        "page": page_idx,
                        "section": current_section,
                        "has_equation": bool(_EQUATION_HINT.search(piece)),
                    },
                )
            )

    return ParsedDoc(
        doc_id=doc_id,
        filename=path.name,
        n_pages=len(pages),
        pages=pages,
        chunks=chunks,
        title=title,
        has_equations=has_eq,
    )


def _clean(text: str) -> str:
    # Collapse hyphenation at line breaks and squeeze whitespace.
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _current_section(page_text: str) -> str:
    for line in page_text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            return m.group(3).title()
    return ""


def iter_years(text: str) -> Iterable[int]:
    for m in _YEAR_RE.finditer(text):
        yield int(m.group())

"""
Smoke tests — verify wiring without downloading models or running Ollama.

Run:  pytest -q         (from the repo root)
These check that modules import, config loads, JSON extraction is robust, and the
parser produces page-tagged chunks from a tiny generated PDF.
"""
from __future__ import annotations

from config.settings import get_settings
from backend.core.llm import _extract_json


def test_settings_load():
    s = get_settings()
    assert s.rerank_top_k > 0
    assert s.llm_model


def test_json_extraction_handles_fences():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prose {"b": [1,2]} more') == {"b": [1, 2]}
    assert _extract_json("not json at all") is None


def test_parser_page_metadata(tmp_path):
    import fitz  # PyMuPDF

    path = tmp_path / "mini.pdf"
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 72), "Introduction\nThis paper studies retrieval. E = mc^2")
    doc.save(path)
    doc.close()

    from backend.core.parsing import parse_pdf

    parsed = parse_pdf(path)
    assert parsed.n_pages == 1
    assert parsed.chunks
    assert parsed.chunks[0].metadata["page"] == 1
    assert parsed.chunks[0].metadata["filename"] == "mini.pdf"

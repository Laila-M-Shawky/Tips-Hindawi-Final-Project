"""
FastAPI routes.

Endpoints:
  POST /ingest                 upload + parse + embed a PDF
  GET  /documents              list ingested docs
  DELETE /documents/{doc_id}   remove a doc
  POST /ask                    grounded Q&A (Self-RAG + citation verify)
  POST /agent                  agentic router -> best tool
  POST /feature/{name}         call a specific feature directly
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.settings import settings
from backend.models import schemas
from backend.core import parsing, vectorstore, retrieval
from backend.features import (
    qa, literature_review, gap_finder, comparison, methodology,
    dataset_recommender, citation_generator, equation_explainer, timeline,
    knowledge_graph,
)
from backend.agents import router as agent_router

router = APIRouter()


@router.post("/ingest", response_model=schemas.IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    dest = settings.upload_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = parsing.parse_pdf(dest)
    n = vectorstore.add_document(doc)
    retrieval.bust_bm25_cache()  # new corpus -> rebuild sparse index
    return schemas.IngestResponse(
        doc_id=doc.doc_id, filename=doc.filename, n_pages=doc.n_pages,
        n_chunks=n, title=doc.title,
    )


@router.get("/documents", response_model=schemas.DocList)
async def documents():
    return schemas.DocList(documents=vectorstore.list_doc_ids())


@router.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    vectorstore.delete_document(doc_id)
    retrieval.bust_bm25_cache()
    return {"deleted": doc_id}


@router.post("/ask")
async def ask(req: schemas.AskRequest):
    return qa.grounded_qa(req.question, req.doc_ids)


@router.post("/agent")
async def agent(req: schemas.AgentRequest):
    return agent_router.dispatch(req.request, req.doc_ids)


_FEATURES = {
    "literature_review": lambda q, d: literature_review.generate(q, d),
    "gap_finder": lambda q, d: gap_finder.find(q, d),
    "comparison": lambda q, d: comparison.compare(d),
    "methodology": lambda q, d: methodology.generate(q, d),
    "dataset_recommender": lambda q, d: dataset_recommender.recommend(q, d),
    "citation_generator": lambda q, d: citation_generator.generate(d),
    "equation_explainer": lambda q, d: equation_explainer.explain(q, d),
    "timeline": lambda q, d: timeline.build(d),
    "knowledge_graph": lambda q, d: knowledge_graph.build(d),
    "qa": lambda q, d: qa.grounded_qa(q, d),
}


@router.post("/feature/{name}")
async def feature(name: str, req: schemas.FeatureRequest):
    if name not in _FEATURES:
        raise HTTPException(404, f"Unknown feature '{name}'. "
                                 f"Options: {list(_FEATURES)}")
    return _FEATURES[name](req.query or "", req.doc_ids)


@router.get("/health")
async def health():
    return {"status": "ok", "llm": settings.active_llm_model, "llm_backend": settings.llm_backend,
            "embed": settings.embed_model}

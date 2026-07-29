"""
ResearchAI FastAPI application.

Run:  uvicorn backend.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from backend.api.routes import router

app = FastAPI(
    title="ResearchAI",
    description="Agentic RAG research assistant for academic papers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"name": "ResearchAI", "docs": "/docs", "llm": settings.llm_model}

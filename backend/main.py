"""
ResearchAI FastAPI application.

Run:  uvicorn backend.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Without this, an unhandled error (e.g. an LLM call failing) reaches the
    # client as a plain-text 500, which crashes the Streamlit frontend's
    # response.json() call with a confusing JSONDecodeError instead of showing
    # the actual cause.
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


@app.get("/")
async def root():
    return {"name": "ResearchAI", "docs": "/docs", "llm": settings.active_llm_model,
            "llm_backend": settings.llm_backend}

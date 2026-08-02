"""Pydantic schemas for API requests/responses."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocList(BaseModel):
    documents: list[dict]


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    n_pages: int
    n_chunks: int
    title: str = ""


class AskRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None


class AgentRequest(BaseModel):
    request: str = Field(..., description="Free-text task; the router picks a tool.")
    doc_ids: list[str] | None = None


class FeatureRequest(BaseModel):
    query: str | None = None
    doc_ids: list[str] | None = None

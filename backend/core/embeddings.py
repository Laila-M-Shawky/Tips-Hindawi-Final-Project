"""
BAAI BGE embeddings.

BGE models expect an instruction prefix on the *query* side only. We pass it via
`encode_kwargs`/`query_encode_kwargs` prompts, which the installed version of
langchain-huggingface accepts. Lazy-loaded so importing stays cheap.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": settings.embed_device},
        encode_kwargs={"normalize_embeddings": True},
    )
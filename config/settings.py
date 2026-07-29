"""
Central configuration for ResearchAI.

Everything (models, paths, retrieval params) is read from here so you can tune
the whole system from one place or via a .env file. Values are validated by
pydantic-settings and can be overridden with environment variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Paths -------------------------------------------------------------
    root_dir: Path = ROOT
    upload_dir: Path = ROOT / "data" / "uploads"
    chroma_dir: Path = ROOT / "data" / "chroma"
    kg_dir: Path = ROOT / "data" / "kg"

    # ---- LLM (Ollama) ------------------------------------------------------
    # Swap between "llama3.1:8b" and "qwen2.5:7b" with one env var.
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = Field(default="llama3.1:8b")
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192  # context window (raise for long papers)
    llm_timeout: int = 180

    # ---- Embeddings (BAAI BGE) --------------------------------------------
    embed_model: str = "BAAI/bge-base-en-v1.5"
    # BGE recommends a query instruction; docs are embedded without it.
    embed_query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )
    embed_device: str = "cpu"  # "cuda" if you have a GPU

    # ---- Reranker (BGE cross-encoder) -------------------------------------
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5  # how many chunks survive reranking -> LLM context

    # ---- Chunking ----------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 150

    # ---- Retrieval ---------------------------------------------------------
    dense_top_k: int = 20      # candidates from vector search
    bm25_top_k: int = 20       # candidates from keyword search
    rrf_k: int = 60            # reciprocal-rank-fusion constant
    multi_query_n: int = 3     # paraphrases per query in multi-query retrieval

    # ---- Self-RAG ----------------------------------------------------------
    self_rag_max_retries: int = 2
    relevance_threshold: float = 0.5

    # ---- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    backend_url: str = "http://localhost:8000"

    # ---- ngrok -------------------------------------------------------------
    ngrok_authtoken: str = ""

    def ensure_dirs(self) -> None:
        for p in (self.upload_dir, self.chroma_dir, self.kg_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()

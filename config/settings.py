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

    # ---- LLM -----------------------------------------------------------------
    # All backends run remotely except hf_local, so nothing heavy loads on this
    # machine by default — use this if a local model (e.g. via Ollama) made your
    # laptop lag or freeze.
    #   "groq"     (default) — free, fast, generous quota. https://console.groq.com/keys
    #   "hf_api"   — Hugging Face Inference API. Free tier credits are small and
    #                model availability varies per account (see README troubleshooting).
    #   "hf_local" — needs a real GPU; runs fully offline via `transformers`.
    llm_backend: str = "groq"  # "groq" | "hf_api" | "hf_local"

    groq_api_key: str = Field(default="")  # https://console.groq.com/keys
    groq_model: str = "llama-3.1-8b-instant"  # or llama-3.3-70b-versatile for quality

    hf_token: str = Field(default="")  # https://huggingface.co/settings/tokens
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"  # or Qwen/Qwen2.5-7B-Instruct
    hf_local_4bit: bool = True  # hf_local only: 4-bit (bitsandbytes) to fit consumer GPUs

    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: int = 120

    @property
    def active_llm_model(self) -> str:
        """Whichever model name is actually in play for the configured backend."""
        return self.groq_model if self.llm_backend == "groq" else self.hf_model

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

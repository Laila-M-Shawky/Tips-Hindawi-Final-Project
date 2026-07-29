# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Laila Shawky                 |
| Project Name     | ResearchAI — Agentic RAG Research Assistant |
| GitHub Username  | `<your-github-username>`             |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |

---

# 📖 Project Overview

**ResearchAI** is a fully local, **agentic RAG (Retrieval-Augmented Generation)** assistant for academic papers. You upload PDFs and it returns **grounded, page-cited** answers — and goes far beyond simple Q&A with eleven research tools: literature review, research-gap analysis, paper comparison, methodology drafting, dataset recommendation, citation generation, equation explanation, timeline building, and an interactive **knowledge graph** — all coordinated by an **agentic router** with **self-correcting (Self-RAG)** retrieval.

The whole system runs offline on **Ollama** (Llama 3.1 / Qwen 2.5), so there are **no API costs** and no data ever leaves the machine. What sets it apart from a typical demo RAG is real retrieval engineering: **hybrid BM25 + dense (BGE) retrieval** fused with **Reciprocal Rank Fusion**, a **BGE cross-encoder reranker**, and an independent **citation-verification** pass that scores how much of every answer is actually supported by the source pages.

---

# ✨ Features

* **Grounded Q&A with page citations** — every claim ends with `[paper.pdf p.4]`, plus a grounding score.
* **Self-RAG self-correction** — grades relevance & groundedness, rewrites the query, and retries.
* **Hybrid retrieval + reranking** — BM25 + dense BGE embeddings fused via RRF, reranked by a BGE cross-encoder.
* **Query expansion / multi-query retrieval** — higher recall through diverse rephrasings.
* **Citation verification** — per-claim ✅/⚠️ support checking against the retrieved evidence.
* **Literature review generator** — theme-organized synthesis with inline citations.
* **Research-gap finder** — actionable gaps + suggested future directions with evidence.
* **Paper comparison** — per-paper digest → comparison matrix + verdict.
* **Methodology generator** — grounded experimental design (RQs, baselines, metrics, ablations).
* **Dataset recommender** — datasets mentioned in the papers + known public benchmarks.
* **Citation generator** — BibTeX / APA / IEEE from real extracted metadata.
* **Equation explainer** — term-by-term math explanation rendered in LaTeX.
* **Timeline generator** — chronological milestones of the field.
* **Knowledge graph** — entity–relation triple extraction with an interactive visualization.
* **Agentic router** — one chat box automatically dispatches to the right tool.

---

# 🛠️ Technologies Used

| Layer | Tools |
| ----- | ----- |
| **LLM (local)** | [Ollama](https://ollama.com) — Llama 3.1 8B / Qwen 2.5 7B (swappable via `.env`) |
| **Orchestration** | LangChain |
| **Backend / API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Vector store** | ChromaDB (persistent) |
| **Embeddings** | BAAI **BGE** (`bge-base-en-v1.5`) |
| **Reranker** | BAAI **BGE reranker** (`bge-reranker-base`, cross-encoder) |
| **Sparse retrieval** | BM25 (`rank-bm25`) + Reciprocal Rank Fusion |
| **PDF parsing** | PyMuPDF (page-aware chunking) |
| **Knowledge graph** | NetworkX + PyVis |
| **Deployment / demo** | ngrok, Google Colab (GPU) |
| **Optimization (advanced)** | Quantization (Q4_K_M via Ollama Modelfile), optional LoRA / QLoRA fine-tuning |

---

# ⚙️ Installation

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/download) installed (on Windows it runs automatically as a background service)
* (Optional) an NVIDIA GPU for faster embeddings / reranking

### Local (Linux / macOS)
```bash
git clone https://github.com/<your-github-username>/research-ai.git
cd research-ai
./setup.sh          # creates venv, installs deps, copies .env, pulls the model
```

### Local (Windows)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
ollama pull llama3.1:8b
```
> On Windows, **do not run `ollama serve`** — Ollama already runs as a service in the system tray (it owns port `11434`). Check with `ollama list`, or open `http://localhost:11434` (it should say *"Ollama is running"*).

### Configure
Edit `.env` to switch models or tune retrieval:
```
LLM_MODEL=qwen2.5:7b        # swap Llama <-> Qwen with one line
EMBED_DEVICE=cuda           # if you have a GPU
RERANK_TOP_K=5
SELF_RAG_MAX_RETRIES=2
```

---

# 🚀 Usage

### Run everything (Linux / macOS)
```bash
./run.sh            # Ollama + FastAPI (:8000) + Streamlit (:8501)
```

### Run manually (any OS — 2 or 3 terminals)
```bash
# terminal 1 (skip on Windows — service already running)
ollama serve

# terminal 2 — backend
uvicorn backend.main:app --port 8000        # API docs at http://localhost:8000/docs

# terminal 3 — frontend
streamlit run frontend/app.py               # open http://localhost:8501
```

### Cloud demo (Google Colab + ngrok)
Open `ResearchAI_Colab_Demo.ipynb`, set **Runtime → GPU (T4)**, paste your ngrok token in the config cell, and run top to bottom. The last cell prints a **public URL** you can open on any device.

### API example
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What loss function is used and why?"}'
```

---

# 📸 Demo

> Add your screenshots / GIFs / demo video here.

* `docs/demo_qa.png` — grounded Q&A with grounding score and page citations
* `docs/demo_agent.gif` — the agentic router picking the right tool
* `docs/demo_kg.png` — interactive knowledge graph
* Demo video: `<link>`

---

# 📈 Results

* **Grounded answers, verifiably.** Every answer carries page-level citations and a **grounding score** (share of claims independently verified against the retrieved evidence), turning "sounds confident" into "provably supported."
* **Higher retrieval quality.** Combining BM25 (exact terms: author, dataset, symbol) with dense BGE (semantics/paraphrase), fusing via RRF, and reranking with a cross-encoder consistently surfaces more relevant passages than dense-only retrieval.
* **Reliability via Self-RAG.** When retrieval is weak or the draft answer isn't grounded, the system rewrites the query and retries instead of hallucinating — and the UI shows the correction.
* **Runs fully offline** on consumer hardware / a single Colab T4, with zero API cost.

> Add your own numbers here (e.g., grounding score on a sample of questions, retrieval hit-rate, average latency per model).

---

# 🔮 Future Improvements

* Evaluation harness with RAGAS / custom groundedness & retrieval metrics.
* Cross-paper knowledge graph merging and graph-based retrieval (GraphRAG).
* Streaming responses and caching for faster interactive use.
* Multi-format ingestion (arXiv links, DOCX, HTML) and OCR for scanned PDFs.
* Domain adaptation via the included LoRA / QLoRA fine-tuning script.
* User authentication and per-user document workspaces.

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.

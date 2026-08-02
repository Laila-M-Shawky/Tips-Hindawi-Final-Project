# 🔬 ResearchAI — Agentic RAG Research Assistant

A **competition-ready**, fully local research assistant for academic papers.
Upload PDFs and get **grounded, page-cited** answers, literature reviews,
research-gap analysis, paper comparisons, methodology drafts, dataset
recommendations, citations, equation explanations, timelines, and an
interactive **knowledge graph** — all orchestrated by an **agentic router**
with **self-correcting (Self-RAG)** retrieval.

The LLM runs on the **free Groq API** by default, so nothing heavy loads on your
machine — the fix if a local model (e.g. Ollama) made your laptop lag or freeze.
Hugging Face's Inference API is available as an alternate `hf_api` backend, and
an `hf_local` backend is there too if you have a real GPU and want fully
offline inference instead.

> 🏆 This repository is my official submission for the
> [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Youssef Dawoud (Joe)                 |
| Project Name     | ResearchAI — Agentic RAG Research Assistant |
| GitHub Username  | `<your-github-username>`             |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en) |

---

## ✨ Why this wins

- **Grounded, not vibes.** Every answer cites the exact `paper.pdf p.4`, and an
  independent **citation-verification** pass scores how much of the answer is
  actually supported by the retrieved evidence.
- **Real retrieval engineering.** Hybrid **BM25 + dense (BGE)** retrieval fused
  with **Reciprocal Rank Fusion**, then a **BGE cross-encoder reranker** — plus
  **query expansion / multi-query** for recall.
- **Self-correcting.** **Self-RAG** grades relevance and groundedness, rewrites
  the query, and retries — and the UI *shows* the correction, which demos well.
- **Agentic.** One chat box, eleven tools. An LLM+rules **router** dispatches to
  the right feature.
- **Modular & runnable.** Clean MVP core; advanced modules (quantization,
  LoRA/QLoRA, knowledge graph) are clearly separated and optional.

---

## 🏗️ Architecture

```
                     ┌─────────────────────────────┐
                     │   Streamlit frontend         │
                     │   (upload · agent · 11 tabs) │
                     └──────────────┬──────────────┘
                                    │  HTTP (requests)
                     ┌──────────────▼──────────────┐
                     │   FastAPI backend            │
                     │   /ingest /ask /agent        │
                     │   /feature/{name}            │
                     └──────────────┬──────────────┘
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
     ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
     │  Agents          │   │  Features (11)  │   │  Core RAG        │
     │  · router        │   │  · qa           │   │  · PyMuPDF parse │
     │  · self_rag      │   │  · lit review   │   │  · BGE embed     │
     │  · query expand  │   │  · gap finder   │   │  · Chroma store  │
     └────────┬────────┘   │  · comparison   │   │  · hybrid BM25   │
              │            │  · methodology  │   │    + dense + RRF │
              │            │  · datasets     │   │  · BGE reranker  │
              │            │  · citations    │   └────────┬────────┘
              │            │  · equations    │            │
              │            │  · timeline     │   ┌────────▼────────┐
              │            │  · knowledge KG │   │  Groq API        │
              │            │  · citation ✓   │   │  Groq (Llama 3.1)│
              │            └─────────────────┘   └─────────────────┘
              └─── LangChain orchestration throughout ───┘
```

**Query lifecycle (Self-RAG):**

```
question ─► expand to N sub-queries ─► hybrid retrieve (BM25+dense, RRF)
        ─► BGE rerank ─► grade chunk relevance ─► grounded generate (cited)
        ─► grade groundedness ─┬─ grounded ─► verify citations ─► answer
                               └─ not grounded ─► rewrite query ─► retry (≤N)
```

---

## 📁 Folder structure

```
research-ai/
├── config/
│   └── settings.py            # single source of truth (pydantic-settings)
├── backend/
│   ├── main.py                # FastAPI app
│   ├── api/routes.py          # all endpoints
│   ├── models/schemas.py      # request/response models
│   ├── core/
│   │   ├── parsing.py         # PyMuPDF -> page-tagged chunks
│   │   ├── embeddings.py      # BAAI BGE embeddings
│   │   ├── vectorstore.py     # ChromaDB persistence
│   │   ├── retrieval.py       # BM25 + dense + RRF + BGE rerank
│   │   └── llm.py             # Hugging Face backend + grounded/JSON helpers
│   ├── agents/
│   │   ├── router.py          # intent routing / dispatch
│   │   ├── self_rag.py        # relevance+groundedness grading, retry
│   │   └── query_expansion.py # multi-query retrieval
│   └── features/
│       ├── qa.py              # grounded Q&A (flagship)
│       ├── citation_verify.py # per-claim support checking
│       ├── literature_review.py
│       ├── gap_finder.py
│       ├── comparison.py
│       ├── methodology.py
│       ├── dataset_recommender.py
│       ├── citation_generator.py
│       ├── equation_explainer.py
│       ├── timeline.py
│       └── knowledge_graph.py # triples + pyvis visualization
├── frontend/app.py            # Streamlit UI
├── scripts/
│   ├── ngrok_deploy.py        # public demo tunnel
│   ├── quantize.py            # local 4-bit quantization (hf_local, advanced)
│   └── finetune_lora.py       # optional LoRA/QLoRA (advanced)
├── tests/test_smoke.py
├── requirements.txt
├── .env.example
├── setup.sh                   # one-shot install
└── run.sh                     # start API + UI
```

---

## 🧩 Features — MVP vs Advanced

### ✅ MVP (must-have core)

| Feature | Endpoint | What it does |
|---|---|---|
| PDF ingestion | `POST /ingest` | PyMuPDF parse → page-tagged chunks → BGE embed → Chroma |
| Grounded Q&A | `POST /ask` | Hybrid retrieval + rerank → cited answer with `page` refs |
| Citation verification | (in `/ask`) | Scores each claim against its cited evidence |
| Literature review | `POST /feature/literature_review` | Theme-organized synthesis with citations |
| Paper comparison | `POST /feature/comparison` | Per-paper digest → comparison matrix + verdict |
| Methodology generator | `POST /feature/methodology` | Grounded experimental design |
| Dataset recommender | `POST /feature/dataset_recommender` | Datasets from papers + known benchmarks |
| Citation generator | `POST /feature/citation_generator` | BibTeX / APA / IEEE from real metadata |
| Equation explainer | `POST /feature/equation_explainer` | Term-by-term math explanation (LaTeX) |
| Timeline generator | `POST /feature/timeline` | Chronological milestones |
| Research-gap finder | `POST /feature/gap_finder` | Actionable gaps + suggested directions |

### 🚀 Advanced modules (optional, clearly separated)

| Module | Where | Notes |
|---|---|---|
| **Agentic router** | `agents/router.py`, `POST /agent` | One box → right tool |
| **Self-RAG / self-correction** | `agents/self_rag.py` | Relevance + groundedness grading, query rewrite, retry |
| **Query expansion / multi-query** | `agents/query_expansion.py` | Higher recall via RRF over paraphrases |
| **Knowledge graph** | `features/knowledge_graph.py` | Triple extraction + interactive pyvis HTML |
| **Local inference (`hf_local`)** | `backend/core/llm.py` | Run the LLM on your own GPU via `transformers` instead of the HF API |
| **Quantization** | `scripts/quantize.py` | 4-bit (bitsandbytes NF4) local model prep, for `hf_local` |
| **LoRA / QLoRA fine-tuning** | `scripts/finetune_lora.py` | 4-bit domain adaptation → merge → `hf_local` or push to the Hub |
| **ngrok deployment** | `scripts/ngrok_deploy.py` | Public URL for live demos / Colab |

> The system is **fully functional without any advanced module.** Enable them to
> raise your score — self-RAG and the knowledge graph are the highest-impact,
> lowest-effort wins to demo.

---

## ⚡ Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)
- (Optional) an NVIDIA GPU for faster embeddings/reranking, or for the `hf_local` backend

### Install
```bash
git clone <your-repo> research-ai && cd research-ai
./setup.sh                      # venv + deps + copies .env
```

Or manually:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Then set `GROQ_API_KEY` in `.env`.

### Configure
Edit `.env` to switch models or tune retrieval:
```
GROQ_MODEL=llama-3.3-70b-versatile   # swap for quality vs. llama-3.1-8b-instant for speed
LLM_BACKEND=groq                # or hf_api / hf_local (see .env.example)
EMBED_DEVICE=cuda               # if you have a GPU
RERANK_TOP_K=5
SELF_RAG_MAX_RETRIES=2
```

---

## ▶️ Run

**All at once:**
```bash
./run.sh                        # FastAPI (:8000) + Streamlit (:8501)
```

**Or separately (2 terminals):**
```bash
uvicorn backend.main:app --reload --port 8000     # API docs at /docs
streamlit run frontend/app.py --server.port 8501
```

**Public demo (ngrok):**
```bash
# set NGROK_AUTHTOKEN in .env, then
python scripts/ngrok_deploy.py --port 8501
```

**Local inference / fine-tune (advanced, needs a GPU):**
```bash
python scripts/quantize.py --base Qwen/Qwen2.5-7B-Instruct --out models/researchai-qwen-4bit
python scripts/finetune_lora.py --base <hf-model> --data data/train.jsonl --qlora
```

---

## 🎬 Demo flow (5 minutes for judges)

1. **Upload** 2–3 papers from the sidebar (watch the page/chunk counts).
2. **Q&A tab** — ask a hard, specific question. Point out:
   - the **grounding score** and green/⚠️ per-claim badges,
   - the **citations** with page numbers,
   - the **Self-RAG trace** (sub-queries + any self-correction).
3. **Agent tab** — type *"compare the two papers' methods"* and show it **routes**
   to the comparison tool automatically.
4. **Gaps tab** — surface concrete research gaps with evidence.
5. **Knowledge Graph tab** — extract and render the interactive graph.
6. (Optional) **ngrok** — open the public URL on your phone to prove it's live.

**One-line pitch:** *"A local, agentic RAG assistant that doesn't just answer —
it retrieves with hybrid search, reranks, self-corrects, and proves every claim
against the source page."*

---

## 🔌 API quick reference

```bash
# ingest
curl -F "file=@paper.pdf" http://localhost:8000/ingest

# grounded Q&A
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What loss function is used and why?"}'

# agentic routing
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"request":"give me a literature review of the uploaded papers"}'

# any feature directly
curl -X POST http://localhost:8000/feature/knowledge_graph \
  -H "Content-Type: application/json" -d '{}'
```

Interactive docs: **http://localhost:8000/docs**

---

## 🧠 Design notes / talking points

- **Why hybrid + RRF?** Dense (BGE) captures semantics/paraphrase; BM25 nails
  exact tokens (author names, dataset names, symbols). RRF fuses them with almost
  no tuning. The **cross-encoder reranker** then does high-precision ordering on
  the fused top-k.
- **Why page-tagged chunks?** Citations are only trustworthy if they point at a
  page. Parsing keeps `page`, `section`, and `has_equation` metadata so features
  can target the right content (e.g. the equation explainer filters on
  `has_equation`).
- **Why Self-RAG?** Naive RAG confidently answers even when retrieval failed.
  Grading relevance and groundedness — then rewriting and retrying — is what
  turns a demo into something reliable.
- **Why Groq for the LLM?** The default `groq` backend calls Groq's free API, so
  nothing heavy runs on your machine — no multi-GB download, no local server, no
  risk of a laptop lagging or freezing under an 8B model, and no risk of hitting
  a tiny free-tier quota mid-demo (unlike Hugging Face's free Inference API,
  which is available as an alternate `hf_api` backend). `hf_local` is there too
  if you'd rather run fully offline on your own GPU.
- **How many papers can I upload?** There's no hard cap in the code. ChromaDB
  and the BGE embeddings scale to thousands of chunks without issue. The one
  cost that grows with corpus size is the BM25 index: it's rebuilt in pure
  Python from every chunk in scope on the first query after an ingest (then
  cached until the next upload/delete), so it's instant for a handful of
  papers and still fine into the tens-to-~100-paper range; a very large corpus
  (many hundreds of papers) would start to feel that rebuild. **Compare**
  specifically caps itself at the first 4 selected documents regardless of how
  many you've uploaded. For a live demo, 2–5 papers (as in the demo flow below)
  keeps everything snappy.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| "Backend not reachable" in the UI | Start FastAPI before Streamlit |
| First request is slow | Embedding/reranker models download on first use; subsequent calls are fast |
| 401 / auth error from the LLM | Set `GROQ_API_KEY` (or `HF_TOKEN` if using `hf_api`) in `.env` |
| Groq: "model not found" / decommissioned | Groq retires old model ids periodically — check https://console.groq.com/docs/models and update `GROQ_MODEL` |
| HF: "not supported by any provider" | `HF_MODEL` isn't hosted by any Inference Provider enabled for your account — try another `HF_MODEL`, or switch `LLM_BACKEND=groq` |
| HF: "402 Payment Required" | Your Hugging Face free Inference credits are exhausted for the month — switch `LLM_BACKEND=groq` (recommended), wait for the reset, or upgrade to HF PRO |
| Empty/garbled JSON from a feature | Lower `LLM_TEMPERATURE`; use an instruct model |
| Reranker OOM on GPU | Set `EMBED_DEVICE=cpu` or lower `DENSE_TOP_K`/`BM25_TOP_K` |

---

## 🔮 Future Improvements

- Evaluation harness with RAGAS / custom groundedness & retrieval metrics.
- Cross-paper knowledge graph merging and graph-based retrieval (GraphRAG).
- Streaming responses and caching for faster interactive use.
- Multi-format ingestion (arXiv links, DOCX, HTML) and OCR for scanned PDFs.
- Domain adaptation via the included LoRA / QLoRA fine-tuning script.
- User authentication and per-user document workspaces.

---

## 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/)
**Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of
[**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages
participants to build real-world projects, apply practical skills, and
showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming
batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

## 📜 License & credits

Built with LangChain, Groq, Hugging Face, ChromaDB, BAAI BGE, PyMuPDF, FastAPI,
and Streamlit. Shared for educational and portfolio purposes.

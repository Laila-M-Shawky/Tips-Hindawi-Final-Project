"""
ResearchAI — Streamlit frontend.

Run:  streamlit run frontend/app.py
Talks to the FastAPI backend (BACKEND_URL). Keep the backend running first.
"""
from __future__ import annotations

import os
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ResearchAI", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def api_get(path):
    return requests.get(f"{BACKEND}{path}", timeout=600).json()


def api_post(path, json=None, files=None):
    return requests.post(f"{BACKEND}{path}", json=json, files=files, timeout=600).json()


def selected_doc_ids():
    docs = st.session_state.get("docs", [])
    chosen = st.session_state.get("chosen_docs", [])
    if not chosen:
        return None  # None == search across everything
    id_by_name = {d["filename"]: d["doc_id"] for d in docs}
    return [id_by_name[n] for n in chosen if n in id_by_name]


def render_citations(cites):
    if not cites:
        return
    with st.expander(f"📎 {len(cites)} citations"):
        for c in cites:
            n = c.get("n", "")
            st.markdown(f"**[{n}]** `{c.get('source')}` — page {c.get('page')}"
                        + (f"\n\n> {c['snippet']}" if c.get("snippet") else ""))


# --------------------------------------------------------------------------- #
# Sidebar: documents
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🔬 ResearchAI")
    st.caption("Agentic RAG for academic papers")

    up = st.file_uploader("Upload a PDF", type=["pdf"])
    if up and st.button("Ingest", use_container_width=True):
        with st.spinner("Parsing, chunking, embedding…"):
            res = api_post("/ingest", files={"file": (up.name, up.getvalue(),
                                                       "application/pdf")})
        st.success(f"{res['filename']} — {res['n_pages']} pages, "
                   f"{res['n_chunks']} chunks")

    try:
        st.session_state.docs = api_get("/documents")["documents"]
    except Exception:
        st.session_state.docs = []
        st.error("Backend not reachable. Start FastAPI first.")

    names = [d["filename"] for d in st.session_state.docs]
    st.session_state.chosen_docs = st.multiselect(
        "Scope (blank = all docs)", names, default=[])
    st.caption(f"{len(names)} document(s) in the index")


# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs([
    "🤖 Agent", "💬 Q&A", "📚 Lit Review", "🕳️ Gaps", "⚖️ Compare",
    "🧪 Methodology", "🗂️ Datasets", "🔖 Citations", "➗ Equations",
    "🕰️ Timeline", "🕸️ Knowledge Graph",
])

# --- Agent ---------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Agentic router")
    st.caption("Type any request. The router picks the right tool automatically.")
    req = st.text_input("Your request",
                        placeholder="e.g. Compare the two papers' methods")
    if st.button("Run agent") and req:
        with st.spinner("Routing…"):
            out = api_post("/agent", {"request": req, "doc_ids": selected_doc_ids()})
        r = out.get("routing", {})
        st.info(f"Routed to **{out.get('intent')}** "
                f"(via {r.get('via')}, confidence {r.get('confidence')}) — "
                f"{r.get('reason')}")
        st.json(out.get("result", {}), expanded=False)

# --- Q&A ------------------------------------------------------------------ #
with tabs[1]:
    st.subheader("Grounded Q&A with citations")
    q = st.text_input("Ask a question about your papers")
    if st.button("Ask") and q:
        with st.spinner("Retrieving, grading, answering, verifying…"):
            out = api_post("/ask", {"question": q, "doc_ids": selected_doc_ids()})
        if out.get("self_corrected"):
            st.warning("🔁 Self-corrected: the first answer wasn't grounded enough.")
        st.markdown(out.get("answer", ""))
        v = out.get("verification", {})
        if v:
            st.metric("Grounding score", f"{v.get('grounding_score', 0):.0%}",
                      f"{v.get('supported')}/{v.get('total')} claims supported")
            for c in v.get("claims", []):
                icon = "✅" if c["supported"] else "⚠️"
                st.markdown(f"{icon} {c['claim']}")
        render_citations(out.get("citations"))
        with st.expander("🔍 Self-RAG trace"):
            st.json(out.get("trace", []))

# --- Literature review ---------------------------------------------------- #
with tabs[2]:
    st.subheader("Literature review generator")
    topic = st.text_input("Topic", key="lr")
    if st.button("Generate review") and topic:
        with st.spinner("Synthesizing…"):
            out = api_post("/feature/literature_review",
                           {"query": topic, "doc_ids": selected_doc_ids()})
        st.markdown(out.get("review", ""))
        render_citations(out.get("citations"))

# --- Gaps ----------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Research-gap finder")
    topic = st.text_input("Topic / area", key="gap")
    if st.button("Find gaps") and topic:
        with st.spinner("Mining limitations & future work…"):
            out = api_post("/feature/gap_finder",
                           {"query": topic, "doc_ids": selected_doc_ids()})
        for g in out.get("gaps", []):
            st.markdown(f"### 🕳️ {g.get('gap')}")
            st.markdown(f"**Why it matters:** {g.get('why_it_matters')}")
            st.markdown(f"**Evidence:** {g.get('evidence')}")
            st.markdown(f"**Suggested direction:** {g.get('suggested_direction')}")
            st.divider()

# --- Compare -------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Paper comparison")
    st.caption("Select 2+ documents in the sidebar (or leave blank to use all).")
    if st.button("Compare papers"):
        with st.spinner("Digesting and comparing…"):
            out = api_post("/feature/comparison", {"doc_ids": selected_doc_ids()})
        if out.get("error"):
            st.error(out["error"])
        else:
            comp = out.get("comparison") or {}
            for row in comp.get("matrix", []):
                st.markdown(f"**{row.get('dimension')}**")
                st.json(row.get("per_paper", {}))
            st.info(comp.get("verdict", ""))

# --- Methodology ---------------------------------------------------------- #
with tabs[5]:
    st.subheader("Methodology generator")
    topic = st.text_input("Research topic/question", key="meth")
    if st.button("Draft methodology") and topic:
        with st.spinner("Designing…"):
            out = api_post("/feature/methodology",
                           {"query": topic, "doc_ids": selected_doc_ids()})
        st.markdown(out.get("methodology", ""))
        render_citations(out.get("citations"))

# --- Datasets ------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Dataset recommender")
    task = st.text_input("Describe your task", key="ds")
    if st.button("Recommend datasets") and task:
        with st.spinner("Searching…"):
            out = api_post("/feature/dataset_recommender",
                           {"query": task, "doc_ids": selected_doc_ids()})
        for d in out.get("datasets", []):
            st.markdown(f"### 🗂️ {d.get('name')}")
            st.write(d.get("description"))
            cols = st.columns(3)
            cols[0].caption(f"Size: {d.get('size', '—')}")
            cols[1].caption(f"Modality: {d.get('modality', '—')}")
            cols[2].caption(f"Access: {d.get('access', '—')}")
            st.markdown(f"*Why:* {d.get('why_relevant')}")
            st.divider()

# --- Citations ------------------------------------------------------------ #
with tabs[7]:
    st.subheader("Citation generator")
    if st.button("Generate citations"):
        with st.spinner("Extracting metadata…"):
            out = api_post("/feature/citation_generator",
                           {"doc_ids": selected_doc_ids()})
        for c in out.get("citations", []):
            st.markdown(f"### {c.get('title', c.get('doc_id'))}")
            st.code(c.get("bibtex", ""), language="bibtex")
            st.markdown(f"**APA:** {c.get('apa', '')}")
            st.markdown(f"**IEEE:** {c.get('ieee', '')}")
            st.divider()

# --- Equations ------------------------------------------------------------ #
with tabs[8]:
    st.subheader("Equation explainer")
    req = st.text_input("Which equation / concept?", key="eq",
                        placeholder="e.g. the attention score equation")
    if st.button("Explain") and req:
        with st.spinner("Reading the math…"):
            out = api_post("/feature/equation_explainer",
                           {"query": req, "doc_ids": selected_doc_ids()})
        st.markdown(out.get("explanation", ""))
        render_citations(out.get("citations"))

# --- Timeline ------------------------------------------------------------- #
with tabs[9]:
    st.subheader("Timeline generator")
    if st.button("Build timeline"):
        with st.spinner("Ordering milestones…"):
            out = api_post("/feature/timeline", {"doc_ids": selected_doc_ids()})
        for e in out.get("timeline", []):
            st.markdown(f"**{e.get('year')}** — {e.get('milestone')}")
            st.caption(e.get("detail", ""))

# --- Knowledge graph ------------------------------------------------------ #
with tabs[10]:
    st.subheader("Knowledge graph")
    if st.button("Extract graph"):
        with st.spinner("Extracting triples…"):
            out = api_post("/feature/knowledge_graph", {"doc_ids": selected_doc_ids()})
        st.caption(f"{out.get('n_nodes')} nodes · {out.get('n_edges')} edges")
        path = out.get("html_path", "")
        if path and path.endswith(".html") and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=650, scrolling=True)
        st.json(out.get("triples", []), expanded=False)

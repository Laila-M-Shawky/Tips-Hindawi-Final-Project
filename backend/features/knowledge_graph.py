"""
Knowledge graph extraction + visualization.

The model extracts (subject, relation, object) triples from paper excerpts. We
build a NetworkX graph, return nodes/edges as JSON (for the API / a D3 frontend),
and also render a standalone interactive HTML with pyvis for the demo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from config.settings import settings
from backend.core import retrieval, vectorstore
from backend.core.llm import structured_json

_SYS = (
    "Extract a knowledge graph of scientific concepts from the excerpts. "
    'Return JSON: {"triples":[{subject, relation, object, type}]} where type is '
    "one of: method, dataset, metric, task, concept. Keep entities canonical "
    "(merge synonyms). 15-40 high-quality triples."
)


def _collect_context(doc_ids: list[str] | None) -> str:
    seeds = ["main method and components", "datasets and metrics",
             "task and problem", "key concepts and relationships"]
    seen, chunks = set(), []
    for s in seeds:
        for r in retrieval.hybrid_retrieve(s, doc_ids=doc_ids, top_k=4):
            key = (r.metadata.get("doc_id"), r.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                chunks.append(r.text)
    return "\n\n".join(chunks[:14])


def build(doc_ids: list[str] | None = None) -> dict[str, Any]:
    context = _collect_context(doc_ids)
    out = structured_json(_SYS, f"Excerpts:\n{context}")
    triples = out.get("triples", []) if isinstance(out, dict) else []

    g = nx.DiGraph()
    for t in triples:
        s, r, o = t.get("subject"), t.get("relation"), t.get("object")
        if not (s and o):
            continue
        g.add_node(s, type=t.get("type", "concept"))
        g.add_node(o, type=t.get("type", "concept"))
        g.add_edge(s, o, label=r or "related_to")

    nodes = [{"id": n, "type": d.get("type", "concept")} for n, d in g.nodes(data=True)]
    edges = [{"source": u, "target": v, "label": d.get("label")}
             for u, v, d in g.edges(data=True)]

    html_path = _render_html(g)
    return {"triples": triples, "nodes": nodes, "edges": edges,
            "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges(),
            "html_path": str(html_path)}


_COLORS = {"method": "#4C78A8", "dataset": "#F58518", "metric": "#54A24B",
           "task": "#E45756", "concept": "#B279A2"}


def _render_html(g: "nx.DiGraph") -> Path:
    try:
        from pyvis.network import Network
    except Exception:
        return settings.kg_dir / "knowledge_graph.json"

    # Dark background + light labels to match the app's theme (default pyvis
    # white background clashed badly), and enough physics-stabilization
    # iterations to settle the layout before it's shown — otherwise the graph
    # renders mid-simulation, still flying apart with no readable labels.
    net = Network(height="650px", width="100%", directed=True,
                  bgcolor="#111827", font_color="#e5e7eb")
    for n, d in g.nodes(data=True):
        net.add_node(n, label=n, color=_COLORS.get(d.get("type"), "#888"),
                     title=d.get("type", "concept"), size=18)
    for u, v, d in g.edges(data=True):
        net.add_edge(u, v, title=d.get("label"), label=d.get("label"))
    net.set_options("""
    var options = {
      "nodes": {"font": {"size": 16, "color": "#e5e7eb"}},
      "edges": {
        "color": {"color": "#8b8f98"},
        "font": {"size": 10, "color": "#c9cdd4", "strokeWidth": 0},
        "smooth": false
      },
      "physics": {
        "stabilization": {"enabled": true, "iterations": 300, "fit": true},
        "barnesHut": {"gravitationalConstant": -12000, "springLength": 140,
                       "springConstant": 0.04}
      }
    }
    """)
    out = settings.kg_dir / "knowledge_graph.html"
    net.write_html(str(out), notebook=False, open_browser=False)
    return out

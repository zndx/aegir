"""Live **Atlas-sourced provenance / lineage DAG** — the lineup Training ▸ Provenance panel.

The *Verifiable Tasks & Lineage* view: a type-level **directed** graph of the pipeline's versioned
artifacts and their derivation edges, read live from the ``aegir_hx`` graph (Atlas / Apache AGE) via
``aegir.governance.graph``. v1 renders the **convergence chain** at the artifact-type level
(Family / Topic → Template → Chapter → Column / Dataset → Job / Run) — compact and situational; the
``vertex`` Atlas-core + ``__rdbms_*`` table-internals are excluded (that detail is the Schema lens).

Served by the bokeh server (topology B), embedded via ``<PanelView app="provenance_app">`` — same
``hv.render`` → bokeh-server → ``PanelView`` path as the other panels (air-gapped; ``hv.Graph`` is a
GraphRenderer element, which renders correctly server-side, unlike the npm ``@bokeh/bokehjs`` build).
Augment as we go: instance-level drill-in, more artifact types, and **gate-verdict overlays** on the
edges (R-pass / HermiT-consistent / coverage-R1 / downstream-eval-lift). See
docs/current/src/roadmap/provenance.md.
"""
from __future__ import annotations

import holoviews as hv
import networkx as nx
import pandas as pd
from bokeh.io import curdoc

hv.extension("bokeh", logo=False)

# Curated pipeline artifacts → stage (left→right column). The convergence chain; everything else
# (Atlas `vertex` core, `__rdbms_*` table internals) is excluded for the high-level view.
STAGE = {"Family": 0, "Topic": 0, "Template": 1, "Chapter": 2,
         "Column": 3, "Dataset": 3, "Job": 4, "Run": 4}


def _placeholder(msg: str):
    return hv.Div(f"<div style='padding:1em;color:#888;font-family:sans-serif'>{msg}</div>")


def _meta_edges() -> list[dict]:
    from aegir.governance import graph as G
    with G.connect() as conn:
        return G.run(conn, "MATCH (a)-[r]->(b) WITH labels(a)[0] AS s, type(r) AS e, "
                           "labels(b)[0] AS d, count(*) AS c RETURN {s:s, e:e, d:d, c:c}")


def _dag():
    try:
        rows = _meta_edges()
    except Exception as e:   # noqa: BLE001 — Atlas may be down/restarting; degrade gracefully
        return _placeholder(f"Atlas/aegir_hx not reachable ({type(e).__name__}) — is the graph up?")

    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        s, d, c, e = r.get("s"), r.get("d"), r.get("c", 0), r.get("e", "")
        if s in STAGE and d in STAGE and s != d and not str(e).startswith("__rdbms"):
            cur = agg.setdefault((s, d), {"c": 0, "types": set()})
            cur["c"] += c
            cur["types"].add(e)
    if not agg:
        return _placeholder("no provenance edges yet — run the pipeline / OpenLineage ingest")

    names = sorted({n for pair in agg for n in pair}, key=lambda n: (STAGE[n], n))
    idx = {n: i for i, n in enumerate(names)}
    g = nx.DiGraph()
    for n in names:
        g.add_node(idx[n], layer=STAGE[n])
    for (s, d) in agg:
        g.add_edge(idx[s], idx[d])
    pos = nx.multipartite_layout(g, subset_key="layer", align="vertical")

    nodes_df = pd.DataFrame([{"index": idx[n], "x": float(pos[idx[n]][0]), "y": float(pos[idx[n]][1]),
                              "label": n, "stage": STAGE[n]} for n in names])
    edges_df = pd.DataFrame([{"source": idx[s], "target": idx[d], "count": v["c"],
                              "kinds": ", ".join(sorted(v["types"]))} for (s, d), v in agg.items()])

    graph = hv.Graph(
        (edges_df, hv.Nodes((nodes_df["x"], nodes_df["y"], nodes_df["index"],
                             nodes_df["label"], nodes_df["stage"]), vdims=["label", "stage"]))
    ).opts(hv.opts.Graph(
        directed=True, arrowhead_length=0.025, node_color="stage", cmap="Category10", node_size=18,
        edge_line_width=hv.dim("count").norm() * 6 + 1, edge_alpha=0.5, edge_color="#9aa",
        width=760, height=430, xaxis=None, yaxis=None, tools=["hover"], padding=(0.08, (0.08, 0.2)),
        title="Provenance — Atlas lineage (type-level convergence chain)"))
    labels = hv.Labels((nodes_df["x"], nodes_df["y"], nodes_df["label"]), ["x", "y"], "label").opts(
        text_font_size="9pt", yoffset=0.07, text_color="#333")
    return (graph * labels).opts(  # type: ignore[operator]  # holoviews has no stubs; * is valid at runtime
        hv.opts.Overlay(show_frame=False, xaxis=None, yaxis=None, padding=(0.08, (0.08, 0.2))))


curdoc().add_root(hv.render(_dag(), backend="bokeh"))

"""Live **Atlas-sourced provenance / lineage DAG** — the lineup Training ▸ Provenance panel.

The *Verifiable Tasks & Lineage* view: a type-level **directed** graph of the pipeline's versioned
artifacts and their derivation edges, read live from the ``aegir_hx`` graph (Atlas / Apache AGE) via
``aegir.governance.graph``. v1 renders the **convergence chain** at the artifact-type level
(Family / Topic → Template → Chapter → Column / Dataset → Job / Run) — compact and situational; the
``vertex`` Atlas-core + ``__rdbms_*`` table-internals are excluded (that detail is the Schema lens).

Served by the bokeh server (topology B), embedded via ``<PanelView app="provenance_app">`` — same
``hv.render`` → bokeh-server → ``PanelView`` path as the other panels (air-gapped; ``hv.Graph`` is a
GraphRenderer element, which renders correctly server-side, unlike the npm ``@bokeh/bokehjs`` build).

**Status: ILLUSTRATIVE, not definitive (RH 2026-06-19).** This is a legibility *sketch* — it proves the
surface (live Atlas graph → HoloViews DAG → tap-to-navigate), but every modelling choice is provisional:
the artifact-type set, the ``STAGE`` layout, the node→lens ``TARGET`` routing, and the edge styling are
all placeholders for the instance-level, topology-derived, verification-overlaid graph described under
"Maturity" in docs/current/src/roadmap/provenance.md. Iterate; don't build heavily on the current maps.
Known next axes: instance-level drill-in, topology-derived artifact set, contextual node→panel routing,
and **gate-verdict overlays** on the edges (R-pass / HermiT-consistent / coverage-R1 / downstream-eval-lift).
"""
from __future__ import annotations

import holoviews as hv
import networkx as nx
import pandas as pd
from bokeh.io import curdoc

hv.extension("bokeh", logo=False)

# Curated pipeline artifacts → stage (left→right column). The convergence chain; everything else
# (Atlas `vertex` core, `__rdbms_*` table internals) is excluded for the high-level view.
# ILLUSTRATIVE: hand-assigned set + columns that force a clean multipartite layout — real lineage is not
# strictly layered (a Run spans stages; the RE_GROUNDS_TO loop is a genuine cycle). Derive from topology.
STAGE = {"Family": 0, "Topic": 0, "Template": 1, "Chapter": 2,
         "Column": 3, "Dataset": 3, "Job": 4, "Run": 4}

# Each artifact-type node drills into the lineup lens that represents its data product — tap a node →
# open that lens as a narrow panel in the trail (same navigation as a lens link).
# ILLUSTRATIVE: coarse type→whole-lens routing that ignores the tapped instance; Run/Job→content is a
# known stretch (their real home is a run-detail/Sweeps panel or a provenance instance note). See roadmap.
TARGET = {"Family": "lens/terms", "Template": "lens/terms",
          "Topic": "lens/content", "Chapter": "lens/content", "Run": "lens/content", "Job": "lens/content",
          "Column": "lens/schema", "Dataset": "lens/schema"}

# Tap a node → dispatch a same-window CustomEvent the React LineupPanel listens for (no iframe → shared
# window); it opens `target` as a narrow panel in the trail. Clear the selection so re-tapping refires.
_TAP_JS = """
const idx = src.selected.indices;
if (!idx.length) return;
const target = src.data['note_id'][idx[0]];
if (target) window.dispatchEvent(new CustomEvent('aegir:open-note',
    {detail: {id: target, label: src.data['label'][idx[0]], app: 'provenance_app'}}));
src.selected.indices = [];
"""


def _tap_hook(plot, _element):
    """Wire node-tap → CustomEvent via the bokeh GraphRenderer's node selection (HoloViews escape hatch)."""
    from bokeh.models import CustomJS, GraphRenderer  # pyright: ignore[reportPrivateImportUsage]  # canonical runtime facade
    gr = next((r for r in plot.state.renderers if isinstance(r, GraphRenderer)), None)
    if gr is None:
        return
    src = gr.node_renderer.data_source
    src.selected.js_on_change("indices", CustomJS(args=dict(src=src), code=_TAP_JS))


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
                              "label": n, "stage": STAGE[n], "note_id": TARGET.get(n, "")} for n in names])
    edges_df = pd.DataFrame([{"source": idx[s], "target": idx[d], "count": v["c"],
                              "kinds": ", ".join(sorted(v["types"]))} for (s, d), v in agg.items()])

    graph = hv.Graph(
        (edges_df, hv.Nodes((nodes_df["x"], nodes_df["y"], nodes_df["index"], nodes_df["label"],
                             nodes_df["stage"], nodes_df["note_id"]), vdims=["label", "stage", "note_id"]))
    ).opts(hv.opts.Graph(
        directed=True, arrowhead_length=0.025, node_color="stage", cmap="Category10", node_size=18,
        edge_line_width=hv.dim("count").norm() * 6 + 1, edge_alpha=0.5, edge_color="#9aa",
        width=760, height=430, xaxis=None, yaxis=None, tools=["hover", "tap"], hooks=[_tap_hook],
        padding=(0.08, (0.08, 0.2)),
        title="Provenance — Atlas lineage (type-level convergence chain)"))
    labels = hv.Labels((nodes_df["x"], nodes_df["y"], nodes_df["label"]), ["x", "y"], "label").opts(
        text_font_size="9pt", yoffset=0.07, text_color="#333")
    return (graph * labels).opts(  # type: ignore[operator]  # holoviews has no stubs; * is valid at runtime
        hv.opts.Overlay(show_frame=False, xaxis=None, yaxis=None, padding=(0.08, (0.08, 0.2))))


from aegir.viz.theme import apply_color_mode  # noqa: E402

_MODE, _K = apply_color_mode()   # org design norm: doc theme follows the UI's data-mode
curdoc().add_root(hv.render(_dag(), backend="bokeh"))

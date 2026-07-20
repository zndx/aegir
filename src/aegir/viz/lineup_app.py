"""Live HoloViews lineup chord — served by the **Bokeh server** (topology B), embedded into React via
``bokeh.embed.server_document`` behind the gateway's reverse proxy. The lens is the ``?lens=`` query
arg (lens/terms | lens/schema | lens/content).

Why bokeh serve, not panel serve: the chord is a pure HoloViews→Bokeh figure, so the Bokeh server
keeps the ``server_document`` embed **fully air-gapped**. Panel's embed bakes a handful of
``cdn.holoviz.org`` resources (design theme CSS, listpanel.css, loading.css) into class/template attrs
at import — bypassing the (correct, local) ``use_cdn`` resolution and resisting runtime localization.
For pure HoloViews/Bokeh/Datashader viz, bokeh serve avoids them entirely; Panel-*widget* dashboards
(later) will mirror the holoviz dist at the deployment layer (the standard air-gapped-Panel pattern).
The same Bokeh server renders ``hv.Chord`` correctly (the npm ``@bokeh/bokehjs`` build does not).

Serve (BOKEH_RESOURCES=server → local, air-gapped resources):
    BOKEH_RESOURCES=server bokeh serve src/aegir/viz/lineup_app.py \\
        --prefix /viz --port 5006 --allow-websocket-origin='*'
"""
from __future__ import annotations

import holoviews as hv
import pandas as pd
from bokeh.io import curdoc

from aegir.viz import lineup_data as D

hv.extension("bokeh", logo=False)


def _chord(lens: str, root: str = "current"):
    # Roots are refs (RH): the chord SHAPE is invariant; the substrate is root-true —
    # current = the release-era collections pivot, scratch = the inverted topic layer's
    # term-grounded topics associated by shared documents.
    trunk = root == "scratch" and lens in D.TRUNK_SIMS
    sims, ring, ring_idx, name = ((D.TRUNK_SIMS, D.TRUNK_RING, D.TRUNK_RING_IDX, D.TRUNK_NAME)
                                  if trunk else (D.SIMS, D.RING, D.RING_IDX, D.NAME))
    cids, s = sims[lens]
    pos = {c: j for j, c in enumerate(cids)}
    edges = []
    for a in range(len(ring)):
        if ring[a] not in pos:
            continue
        for b in range(a + 1, len(ring)):
            if ring[b] not in pos:
                continue
            v = float(s[pos[ring[a]], pos[ring[b]]])
            if v > 1e-6:
                edges.append((ring_idx[ring[a]], ring_idx[ring[b]], v))
    edges.sort(key=lambda e: -e[2])
    edges = edges[:50]
    present = sorted({e[0] for e in edges} | {e[1] for e in edges})
    local = {gi: k for k, gi in enumerate(present)}
    nodes = pd.DataFrame([{"index": local[gi], "name": name[gi]} for gi in present])
    eds = pd.DataFrame([(local[a], local[b], v) for a, b, v in edges],
                       columns=["source", "target", "value"])
    label = "trunk topics × shared docs" if trunk else "collections"
    return hv.Chord((eds, hv.Dataset(nodes, "index"))).opts(
        hv.opts.Chord(labels="name", node_color="index", edge_color="source", cmap="Category20",
                      width=520, height=520, tools=["hover"],
                      title=""))




def _pathways(onto: str = "sdg"):
    """The INHERENT navigable-pathways chord (RH 2026-07-20): groups = the ontology's own
    discriminating ancestors, edges = its restriction web + property lattice — universally
    available for ANY ontology under management (native or foreign), no maxsim, no topics.
    Substrate: build/<onto>_coverage.json (foreign_ontology_coverage.py)."""
    import json as _json
    from pathlib import Path as _P
    cov = _P(__file__).resolve().parents[3] / "build" / f"{onto}_coverage.json"
    try:
        pw = _json.loads(cov.read_text()).get("pathways") or {}
    except Exception:  # noqa: BLE001
        pw = {}
    groups = pw.get("groups") or []
    edges_in = pw.get("edges") or []
    names = [g["group"][:28] for g in groups] + ["(other)"]
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for e in edges_in:
        a, b = e["src"][:28], e["dst"][:28]
        if a not in idx:
            idx[a] = len(names); names.append(a)
        if b not in idx:
            idx[b] = len(names); names.append(b)
        rows.append((idx[a], idx[b], int(e["n"]), " · ".join(e.get("props", [])[:3])))
    present = sorted({r[0] for r in rows} | {r[1] for r in rows})
    local = {gi: k for k, gi in enumerate(present)}
    nodes = pd.DataFrame([{"index": local[gi], "name": names[gi]} for gi in present])
    eds = pd.DataFrame([(local[a], local[b], v, pr) for a, b, v, pr in rows],
                       columns=["source", "target", "value", "props"])
    return hv.Chord((eds, hv.Dataset(nodes, "index")), vdims=["value", "props"]).opts(
        hv.opts.Chord(labels="name", node_color="index", edge_color="source", cmap="Category20",
                      width=560, height=560, tools=["hover"], title=""))


def _args() -> "tuple[str, str, str, str]":
    sc = curdoc().session_context
    args = sc.request.arguments if (sc and sc.request) else {}

    def _get(key: str, default: str) -> str:
        raw = args.get(key, [default.encode()])
        return raw[0].decode() if raw and isinstance(raw[0], (bytes, bytearray)) else (raw[0] if raw else default)

    lens = _get("lens", "lens/terms")
    root = _get("root", "current")
    return (lens if lens in D.SIMS else "lens/terms"), root, _get("view", ""), _get("onto", "sdg")


from aegir.viz.theme import apply_color_mode, themed  # noqa: E402

_MODE, _K = apply_color_mode()   # org design norm: doc theme follows the UI's data-mode
_lens, _root, _view, _onto = _args()
_plot = _pathways(_onto) if _view == "pathways" else _chord(_lens, _root)
curdoc().add_root(themed(hv.render(_plot, backend="bokeh"), _K))

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


def _args() -> "tuple[str, str]":
    sc = curdoc().session_context
    args = sc.request.arguments if (sc and sc.request) else {}

    def _get(key: str, default: str) -> str:
        raw = args.get(key, [default.encode()])
        return raw[0].decode() if raw and isinstance(raw[0], (bytes, bytearray)) else (raw[0] if raw else default)

    lens = _get("lens", "lens/terms")
    root = _get("root", "current")
    return (lens if lens in D.SIMS else "lens/terms"), root


from aegir.viz.theme import apply_color_mode, themed  # noqa: E402

_MODE, _K = apply_color_mode()   # org design norm: doc theme follows the UI's data-mode
curdoc().add_root(themed(hv.render(_chord(*_args()), backend="bokeh"), _K))

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


def _chord(lens: str):
    cids, s = D.SIMS[lens]
    pos = {c: j for j, c in enumerate(cids)}
    edges = []
    for a in range(len(D.RING)):
        if D.RING[a] not in pos:
            continue
        for b in range(a + 1, len(D.RING)):
            if D.RING[b] not in pos:
                continue
            v = float(s[pos[D.RING[a]], pos[D.RING[b]]])
            if v > 1e-6:
                edges.append((D.RING_IDX[D.RING[a]], D.RING_IDX[D.RING[b]], v))
    edges.sort(key=lambda e: -e[2])
    edges = edges[:50]
    present = sorted({e[0] for e in edges} | {e[1] for e in edges})
    local = {gi: k for k, gi in enumerate(present)}
    nodes = pd.DataFrame([{"index": local[gi], "name": D.NAME[gi]} for gi in present])
    eds = pd.DataFrame([(local[a], local[b], v) for a, b, v in edges],
                       columns=["source", "target", "value"])
    return hv.Chord((eds, hv.Dataset(nodes, "index"))).opts(
        hv.opts.Chord(labels="name", node_color="index", edge_color="source", cmap="Category20",
                      width=520, height=520, tools=["hover"], title=f"{lens} · live via Bokeh server"))


def _lens_arg() -> str:
    sc = curdoc().session_context
    args = sc.request.arguments if (sc and sc.request) else {}
    raw = args.get("lens", [b"lens/terms"])
    lens = raw[0].decode() if raw and isinstance(raw[0], (bytes, bytearray)) else (raw[0] if raw else "lens/terms")
    return lens if lens in D.SIMS else "lens/terms"


curdoc().add_root(hv.render(_chord(_lens_arg()), backend="bokeh"))

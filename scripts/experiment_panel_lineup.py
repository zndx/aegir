"""PROTOTYPE / convergence spike — NOT wired into the app.

De-risks converging Aegir's viz onto the full HoloViews/Datashader/Dask/Panel stack (the air-gapped
stack already used elsewhere). It serves, live via Panel (Bokeh-server protocol, python-bokeh's own
JS — *not* the npm `@bokeh/bokehjs` build):

  • the exact ``hv.Chord`` the vite-bundled npm BokehJS mishandles (GraphRenderer edge path), over
    the real lineup ``maps`` — terms / schema / content; and
  • the existing leaderboard HoloViews curve (``runs._loss_curve``) — proving the leaderboard surface
    converges too.

Run (air-gapped — resources served from the local server, no CDN):
    BOKEH_RESOURCES=server uv run --no-sync panel serve scripts/experiment_panel_lineup.py \
        --port 5007 --allow-websocket-origin='*'

Finding write-up: docs/scratch/<date>/panel_convergence.md
"""
from __future__ import annotations

from collections import defaultdict

import holoviews as hv
import pandas as pd
import panel as pn

from aegir.lineup import build, chords as C, sources as S
from aegir.utils import runs

pn.extension(sizing_mode="stretch_width")
hv.extension("bokeh", logo=False)

# ── real data (the same path build.run uses) ────────────────────────────────────────────────
rows = S.load_ontology()
corpus, _crun = S.corpus_recs()
coverage, _cov = S.coverage_recs()
_notes, maps = build.project_collections(corpus, coverage)
tid_table, fks = build._relational_spine(rows)

# ── TF-IDF chord substrate (reuse chords.py internals, but emit hv.Chord, not native Bokeh) ───
feats, fk_ok = C._features(maps, tid_table, fks)
sims = {lens: (cids, s) for lens in C.LENSES for cids, s in [C._sim(feats[lens])] if s is not None}
strength: dict[str, float] = defaultdict(float)
for _cids, _s in sims.values():
    _tot = _s.sum(axis=1)
    for _i, _c in enumerate(_cids):
        strength[_c] += float(_tot[_i])
ring = [c for c, _ in sorted(strength.items(), key=lambda kv: -kv[1])[:30]]
ring_idx = {c: i for i, c in enumerate(ring)}
name_by_idx = {ring_idx[c]: C._label(c) for c in ring}


def hv_chord_for(lens: str):
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
    nodes_df = pd.DataFrame([{"index": local[gi], "name": name_by_idx[gi]} for gi in present])
    edges_df = pd.DataFrame([(local[a], local[b], v) for a, b, v in edges],
                            columns=["source", "target", "value"])
    return hv.Chord((edges_df, hv.Dataset(nodes_df, "index"))).opts(
        hv.opts.Chord(labels="name", node_color="index", edge_color="source", cmap="Category20",
                      width=520, height=520, tools=["hover"],
                      title=f"hv.Chord · {lens} (live via Panel, air-gapped)"))


# ── the existing leaderboard curve path (synthetic epochs — proves the surface, no run needed) ─
epochs = [{"epoch": i, "train_loss": 2.5 / (i + 1) + 0.30, "val_loss": 2.7 / (i + 1) + 0.36}
          for i in range(12)]
loss_curve = runs._loss_curve(epochs).opts(title="runs._loss_curve (live via Panel)")

status = "fk-spanning" if fk_ok else "fallback: tables≈terms"
tabs = pn.Tabs(
    ("Terms", pn.pane.HoloViews(hv_chord_for("lens/terms"))),
    ("Schema", pn.pane.HoloViews(hv_chord_for("lens/schema"))),
    ("Content", pn.pane.HoloViews(hv_chord_for("lens/content"))),
    ("Leaderboard · Loss", pn.pane.HoloViews(loss_curve)),
)
pn.Column(
    pn.pane.Markdown(
        f"## Panel-served lineup — convergence spike\n"
        f"`hv.Chord` rendered live through Panel/Bokeh-server (python-bokeh JS, **no npm `@bokeh/"
        f"bokehjs`, no CDN**). Schema chord status: **{status}** · {len(ring)}-node ring · "
        f"{len(maps.get('collections', []))} collections."),
    tabs,
).servable(title="Aegir · Panel convergence spike")

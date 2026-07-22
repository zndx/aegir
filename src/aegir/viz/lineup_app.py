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
    base = "ontology/self-census/group/" if onto == "sdg" else f"ontology/foreign/{onto}/group/"
    # note id rides the node DATA (slug — identity), never the displayed name (RH 2026-07-21)
    nid_by_name = {}
    for g in groups:
        if g.get("slug"):
            nid_by_name[g["group"][:28]] = base + g["slug"]
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
    nodes = pd.DataFrame([{"index": local[gi], "name": names[gi],
                           "nid": nid_by_name.get(names[gi], "")} for gi in present])
    eds = pd.DataFrame([(local[a], local[b], v, pr) for a, b, v, pr in rows],
                       columns=["source", "target", "value", "props"])
    # tap-to-open: node tap posts the group-panel id to the host page (script embed — same
    # document), which the React PanelView forwards into the lineup trail. Navigation, not
    # just orientation (RH 2026-07-20).

    def _tap_hook(plot, element):  # noqa: ANN001
        try:
            from bokeh.models import CustomJS, TapTool
            st = plot.state
            if not any(isinstance(t, TapTool) for t in st.tools):
                st.add_tools(TapTool())
            for r in st.renderers:
                ds = getattr(r, "data_source", None)
                if ds is not None and "nid" in getattr(ds, "data", {}):
                    ds.selected.js_on_change("indices", CustomJS(
                        args={"ds": ds},
                        code="const i=ds.selected.indices[0];"
                             "if(i!=null){const nid=(ds.data['nid']||[])[i];"
                             "if(nid&&window.__lineupOpen)window.__lineupOpen(nid);}"))
        except Exception:  # noqa: BLE001 — navigation is an enhancement, never a crash
            pass

    return hv.Chord((eds, hv.Dataset(nodes, "index")), vdims=["value", "props"]).opts(
        hv.opts.Chord(labels="name", node_color="index", edge_color="source", cmap="Category20",
                      width=560, height=560, tools=["hover", "tap"], title="",
                      hooks=[_tap_hook]))




def _aperture_substrate(root: str, ref: str) -> dict:
    """Root-true chord substrate (RH 2026-07-22, panel-drift correction): scratch reads the
    LIVE derivation (build/aperture_constituents.json); current reads the RELEASED strategy
    lens snapshot (strategy/components/lens/aperture.snapshot.json — the aperture as
    shipped); archive reads the SAME component at the release's strategy ref via git show
    (corpus-v0.6 …). One panel class, three root-true substrates — browsing compares data,
    not rendering eras."""
    import json as _json
    import subprocess as _sp
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[3]
    if root in ("", "scratch"):
        d = _json.loads((repo / "build" / "aperture_constituents.json").read_text())
        return d.get("anchors") or {}
    comp = "components/lens/aperture.snapshot.json"
    if ref:
        raw = _sp.run(["git", "show", f"{ref}:{comp}"], cwd=repo / "strategy",
                      capture_output=True, text=True, check=True).stdout
    else:
        raw = (repo / "strategy" / comp).read_text()
    rows = _json.loads(raw)
    rows = rows if isinstance(rows, list) else rows.get("points") or []
    return {r.get("label") or r.get("iri"): {"iri": r.get("iri"), "point_id": r.get("id"),
                                             "constituents": r.get("constituents") or []}
            for r in rows}


def _aperture_chord(root: str = "scratch", ref: str = "", target: str = ""):
    """The APERTURE as the Lexicon's orienting chord (RH 2026-07-21): arcs = the admission
    anchors (labeled by their IRI FRAGMENT — display IS identity, so the instrument cannot
    drift), chords = SHARED CONSTITUENT CONCEPTS between anchors (the M:N lattice made
    visual). Heavy chords = anchors insufficiently differentiated — the chord doubles as the
    aperture-refinement instrument. Tap → the anchor's panel."""
    import json as _json
    from itertools import combinations
    from pathlib import Path as _P
    try:
        anchors = _aperture_substrate(root, ref)
    except Exception:  # noqa: BLE001
        return hv.Chord(([], hv.Dataset(pd.DataFrame({"index": [], "name": []}), "index")))
    # scheme-aware ring order (SKOS S5-S8): arcs group by TOP-CONCEPT family — the ring
    # itself reads as the ConceptScheme's structure; edges stay the constituent lattice
    fam = {}
    try:
        from aegir.ontology import domain_index as DI
        ov = {k: c for k, c in DI.load_skos(str(DI.DEFAULT_OVERLAY)).items()
              if not getattr(c, "deprecated", False)}
        def _root(k, seen=frozenset()):
            c = ov.get(k)
            b = (getattr(c, "broader", "") or "") if c else ""
            return k if not b or b in seen or b not in ov else _root(b, seen | {k})
        fam = {str(k).rsplit("#", 1)[-1]: str(_root(str(k))).rsplit("#", 1)[-1] for k in ov}
    except Exception:  # noqa: BLE001
        pass
    rows_ = []
    df = {}
    for lbl, e in anchors.items():
        if not isinstance(e, dict):
            continue
        frag = str(e.get("iri") or lbl).rsplit("#", 1)[-1]
        cons = {x.get("iri") for x in e.get("constituents", [])
                if x.get("kind") == "concept" and x.get("iri")}
        rows_.append({"frag": frag, "pid": e.get("point_id"), "cons": cons})
        for c in cons:
            df[c] = df.get(c, 0) + 1
    rows_.sort(key=lambda r: (fam.get(r["frag"], r["frag"]), r["frag"]))
    idx = {r["frag"]: i for i, r in enumerate(rows_)}
    # SELECTIVE edges (RH 2026-07-21 sanity check): promiscuous constituents (df≥4 across
    # anchors) carry no discrimination and made the chord a 91%-dense hairball; counting only
    # df≤3 shares yields 12% density with 58% intra-family edge mass — nameable bridges,
    # scheme structure visible. The df distribution doubles as the MaxSim selectivity-dilution
    # watchlist for aperture extensions.
    eds_rows = []
    for a, b in combinations(rows_, 2):
        n = sum(1 for c in a["cons"] & b["cons"] if df.get(c, 9) <= 3)
        if n:
            eds_rows.append((idx[a["frag"]], idx[b["frag"]], n))
    # Every tap lands on the ANCHOR's panel — the facet vertex (RH 2026-07-22): the
    # panel leads schema-forward (identity → encoded text → relational affordances), so
    # the lens flavor lives in the PANEL, and the tapped IRI is never lost to a shared
    # landing. (`target` reserved for future lens-specific tap policies.)
    nodes = pd.DataFrame([{"index": i, "name": r["frag"][:30],
                           "nid": f"lexicon/aperture/{r['pid']}" if r["pid"] is not None else ""}
                          for i, r in enumerate(rows_)])
    eds = pd.DataFrame(eds_rows, columns=["source", "target", "value"])

    def _tap_hook(plot, element):  # noqa: ANN001
        try:
            from bokeh.models import CustomJS, TapTool
            st = plot.state
            if not any(isinstance(t, TapTool) for t in st.tools):
                st.add_tools(TapTool())
            for r in st.renderers:
                ds = getattr(r, "data_source", None)
                if ds is not None and "nid" in getattr(ds, "data", {}):
                    ds.selected.js_on_change("indices", CustomJS(
                        args={"ds": ds},
                        code="const i=ds.selected.indices[0];"
                             "if(i!=null){const nid=(ds.data['nid']||[])[i];"
                             "if(nid&&window.__lineupOpen)window.__lineupOpen(nid);}"))
        except Exception:  # noqa: BLE001
            pass

    return hv.Chord((eds, hv.Dataset(nodes, "index"))).opts(
        hv.opts.Chord(labels="name", node_color="index", edge_color="source", cmap="Category20",
                      width=560, height=560, tools=["hover", "tap"], title="",
                      hooks=[_tap_hook]))


def _args() -> "tuple[str, str, str, str, str, str]":
    sc = curdoc().session_context
    args = sc.request.arguments if (sc and sc.request) else {}

    def _get(key: str, default: str) -> str:
        raw = args.get(key, [default.encode()])
        return raw[0].decode() if raw and isinstance(raw[0], (bytes, bytearray)) else (raw[0] if raw else default)

    lens = _get("lens", "lens/terms")
    root = _get("root", "current")
    return ((lens if lens in D.SIMS else "lens/terms"), root,
            _get("view", ""), _get("onto", "sdg"), _get("ref", ""), _get("target", ""))


from aegir.viz.theme import apply_color_mode, themed  # noqa: E402

_MODE, _K = apply_color_mode()   # org design norm: doc theme follows the UI's data-mode
_lens, _root, _view, _onto, _ref, _target = _args()
_plot = (_aperture_chord(_root, _ref, _target) if _view == "aperture"
         else _pathways(_onto) if _view == "pathways" else _chord(_lens, _root))
curdoc().add_root(themed(hv.render(_plot, backend="bokeh"), _K))

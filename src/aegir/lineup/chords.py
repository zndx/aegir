"""TF-IDF collection-association CHORD diagrams for the lineup lens panels.

Each lens gets a chord of the strongest collection↔collection associations over that lens's axis:
  • Terms   — collections sharing realized ontology terms
  • Content — collections sharing FinePDFs topics
  • Schema  — collections whose tables foreign-key into common downstream tables (the term↔table
              *many-to-many progress* signal: sparse while template→table is 1:1, densifying as
              data-elements come to span tables; falls back to tables≈terms when no FK graph).

Association = **IDF-weighted cosine** (TF is ~binary, so the IDF down-weights ubiquitous shared
features and the ribbons surface *distinctive* affinity — the anti-hairball mechanism). A global
top-N ring fixes each collection's color + angular order; per lens the ring is restricted to the
collections that lens actually wires (a Schema FK-only lens shows fewer than Terms) — same collection
keeps the same color/order across lenses, the consistent-orientation anchor. Each lens shows its own
top-k ribbons. Rendered as a self-contained native-Bokeh ``json_item`` (circular nodes + quadratic-
bezier ribbons — hv.Chord's GraphRenderer doesn't survive ``json_item``→bokehjs-3.9) and inlined onto
the lens note frontmatter, mounted client-side by BokehJS. Deterministic. bokeh/sklearn are imported
lazily — callers guard (the chord is an optional enrichment, not a hard build dep).
"""
from __future__ import annotations

import re
from collections import defaultdict

LENSES = ("lens/terms", "lens/schema", "lens/content")


def _label(cid: str) -> str:
    m = re.search(r"topic-0*(\d+)", cid)
    return f"topic {m.group(1)}" if m else cid.split("/")[-1]


def _features(maps: dict, tid_table: dict | None, fks: list | None) -> tuple[dict, bool]:
    """{lens: {collection_id: set(token)}} for the three axes; + whether real FK signal was used."""
    colls = [f"collection/topic-{t:03d}" for t in maps.get("collections", [])]
    coll_terms = maps.get("coll_terms", {})
    coll_topics = maps.get("coll_topics", {})
    tid_table = tid_table or {}
    # Schema: the tables a collection's tables FK INTO (shared downstream hubs).
    src_by_coll = {c: {tid_table.get(t) for t in coll_terms.get(c, []) if tid_table.get(t)} for c in colls}
    fk_targets: dict[str, set] = defaultdict(set)
    for e in (fks or []):
        for c, srcs in src_by_coll.items():
            if e.src_table in srcs and getattr(e, "dst_table", None):
                fk_targets[c].add(e.dst_table)
    feats = {lens: {} for lens in LENSES}
    for c in colls:
        feats["lens/terms"][c] = set(coll_terms.get(c, []))
        feats["lens/content"][c] = {f"t{t}" for t in coll_topics.get(c, [])}
        feats["lens/schema"][c] = set(fk_targets.get(c, ())) or set(coll_terms.get(c, []))
    return feats, any(fk_targets.values())


def _sim(fmap: dict):
    """(collection_ids, cosine matrix) over IDF-weighted feature tokens, diagonal zeroed."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    cids = [c for c in fmap if fmap[c]]
    if len(cids) < 2:
        return cids, None
    docs = [" ".join(sorted(fmap[c])) for c in cids]
    try:
        x = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=1).fit_transform(docs)
    except ValueError:
        return cids, None
    s = cosine_similarity(x)
    for i in range(len(cids)):
        s[i, i] = 0.0
    return cids, s


def build_lens_chords(maps: dict, *, tid_table: dict | None = None, fks: list | None = None,
                      top_n: int = 30, top_k: int = 50) -> tuple[dict, str]:
    """Return ({lens_id: bokeh json_item}, status). Shared top-N node ring; per-lens top-k ribbons.
    Lenses with no associations are omitted. Raises only on a hard import failure (caller guards)."""
    feats, fk_ok = _features(maps, tid_table, fks)
    sims = {lens: (cids, s) for lens in LENSES for cids, s in [_sim(feats[lens])] if s is not None}
    if not sims:
        return {}, "no associations"

    strength: dict[str, float] = defaultdict(float)
    for cids, s in sims.values():
        tot = s.sum(axis=1)
        for i, c in enumerate(cids):
            strength[c] += float(tot[i])
    ring = [c for c, _ in sorted(strength.items(), key=lambda kv: -kv[1])[:top_n]]
    ring_idx = {c: i for i, c in enumerate(ring)}
    name_by_idx = {ring_idx[c]: _label(c) for c in ring}    # global index → label (stable across lenses)

    out: dict = {}
    for lens, (cids, s) in sims.items():
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
                    edges.append((ring_idx[ring[a]], ring_idx[ring[b]], v))   # keep GLOBAL ring indices
        edges.sort(key=lambda e: -e[2])
        edges = edges[:top_k]
        if not edges:
            continue
        try:
            out[lens] = _render_chord(edges, name_by_idx)
        except Exception as e:  # noqa: BLE001
            print(f"   chord[{lens}] render failed: {type(e).__name__}: {e}")
    return out, ("fk-spanning" if fk_ok else "fallback: tables≈terms (low relational complexity)")


def _render_chord(edges: list, name_by_idx: dict) -> dict:
    """Native-Bokeh chord-like diagram → ``json_item``: ring nodes (circular) + quadratic-bezier
    ribbons. NOT hv.Chord — its GraphRenderer's MultiLine edge X-spec deserializes to ``undefined``
    in bokehjs 3.9 (``embed_item`` throws "reading 'get' of undefined"); native ``multi_line`` +
    ``scatter`` serialize cleanly. ``edges`` carry GLOBAL ring indices; we restrict to the nodes this
    lens actually wires and color each by its global index → a collection keeps its color across
    lenses (the consistent-orientation anchor) while only its present nodes/ribbons are drawn.
    """
    import math

    from bokeh.embed import json_item
    from bokeh.models import ColumnDataSource, HoverTool, LabelSet
    from bokeh.palettes import Category20_20
    from bokeh.plotting import figure

    present = sorted({a for a, _, _ in edges} | {b for _, b, _ in edges})   # global indices wired here
    local = {gi: k for k, gi in enumerate(present)}
    n = len(present)
    ang = [2 * math.pi * k / n for k in range(n)]
    nx = [math.cos(a) for a in ang]
    ny = [math.sin(a) for a in ang]

    vmax = max(v for _, _, v in edges) or 1.0
    xs, ys, lw, strv, pair = [], [], [], [], []
    for ga, gb, v in edges:
        a, b = local[ga], local[gb]
        ts = [i / 24 for i in range(25)]    # quadratic bezier P0=node_a, control=center, P2=node_b
        xs.append([(1 - t) ** 2 * nx[a] + t ** 2 * nx[b] for t in ts])
        ys.append([(1 - t) ** 2 * ny[a] + t ** 2 * ny[b] for t in ts])
        lw.append(0.8 + 4.0 * v / vmax)
        strv.append(round(v, 3))
        pair.append(f"{name_by_idx[ga]} ↔ {name_by_idx[gb]}")
    edge_src = ColumnDataSource(dict(xs=xs, ys=ys, lw=lw, strength=strv, pair=pair))

    node_src = ColumnDataSource(dict(
        x=nx, y=ny, name=[name_by_idx[present[k]] for k in range(n)],
        color=[Category20_20[present[k] % 20] for k in range(n)]))

    p = figure(width=460, height=460, x_range=(-1.45, 1.45), y_range=(-1.45, 1.45),
               match_aspect=True, toolbar_location="above", tools="pan,wheel_zoom,reset,save",
               outline_line_color=None, background_fill_color=None, border_fill_color=None)
    p.axis.visible = False
    p.grid.visible = False
    ribbons = p.multi_line(xs="xs", ys="ys", line_width="lw", line_color="#6b8cff", line_alpha=0.35,
                           hover_line_color="#1f3fae", hover_line_alpha=0.95, source=edge_src)
    nodes = p.scatter("x", "y", size=11, fill_color="color", line_color="white", source=node_src)
    p.add_layout(LabelSet(x="x", y="y", text="name", source=node_src, text_font_size="7pt",
                          x_offset=5, y_offset=2, text_color="#444"))
    p.add_tools(HoverTool(renderers=[nodes], tooltips=[("collection", "@name")]))
    p.add_tools(HoverTool(renderers=[ribbons], line_policy="interp",
                          tooltips=[("assoc", "@pair"), ("strength", "@strength")]))
    return json_item(p)

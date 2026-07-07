"""TF-IDF collection-association substrate for the lineup lens chords.

For each lens, builds per-collection feature tokens and an IDF-weighted cosine similarity over them —
the substrate the *live* chord turns into an ``hv.Chord`` (``aegir.viz.lineup_data`` precomputes the
ring/sims; ``aegir.viz.lineup_app`` renders it, served by the bokeh server):
  • Terms   — collections sharing realized ontology terms
  • Content — collections sharing FinePDFs topics
  • Schema  — collections sharing relational SHAPE: realized-subgraph FK hubs + COLUMN
              VOCABULARY (typed attributes genuinely recur across templates — the
              h_colset/de-canning axis; post family-complex retirement the deterministic
              subgraphs are template-disjoint, so column vocabulary carries the
              discriminating signal). Falls back to tables≈terms when neither is present.

Association = **IDF-weighted cosine** (TF is ~binary, so the IDF down-weights ubiquitous shared
features and surfaces *distinctive* affinity — the anti-hairball mechanism). sklearn is imported
lazily. (The former native-Bokeh ``json_item`` renderer was retired when the lineup moved to live
HoloViews chords served by the bokeh server — see docs/scratch/2026-06-18/195536_live_viz_spine.md.)
"""
from __future__ import annotations

import re
from collections import defaultdict

LENSES = ("lens/terms", "lens/schema", "lens/content")


def _label(cid: str) -> str:
    m = re.search(r"topic-0*(\d+)", cid)
    return f"topic {m.group(1)}" if m else cid.split("/")[-1]


def _features(maps: dict, tid_table: dict | None, fks: list | None,
              col_tokens: dict | None = None) -> tuple[dict, bool]:
    """{lens: {collection_id: set(token)}} for the three axes; + whether real schema signal
    (FK hubs or column vocabulary) was used rather than the tables≈terms fallback."""
    colls = [f"collection/topic-{t:03d}" for t in maps.get("collections", [])]
    coll_terms = maps.get("coll_terms", {})
    coll_topics = maps.get("coll_topics", {})
    tid_table = tid_table or {}
    col_tokens = col_tokens or {}
    # Schema: the tables a collection's tables FK INTO (shared downstream hubs)…
    src_by_coll = {c: {tid_table.get(t) for t in coll_terms.get(c, []) if tid_table.get(t)} for c in colls}
    fk_targets: dict[str, set] = defaultdict(set)
    for e in (fks or []):
        for c, srcs in src_by_coll.items():
            if e.src_table in srcs and getattr(e, "dst_table", None):
                fk_targets[c].add(e.dst_table)
    # …plus the column vocabulary its templates realize (shared typed attributes).
    coll_cols = {c: set().union(*(col_tokens.get(t, set()) for t in coll_terms.get(c, [])))
                 if coll_terms.get(c) else set() for c in colls}
    feats = {lens: {} for lens in LENSES}
    for c in colls:
        feats["lens/terms"][c] = set(coll_terms.get(c, []))
        feats["lens/content"][c] = {f"t{t}" for t in coll_topics.get(c, [])}
        feats["lens/schema"][c] = (set(fk_targets.get(c, ())) | coll_cols.get(c, set())) \
            or set(coll_terms.get(c, []))
    return feats, any(fk_targets.values()) or any(coll_cols.values())


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

"""TF-IDF collection-association substrate for the lineup lens chords.

For each lens, builds per-collection feature tokens and an IDF-weighted cosine similarity over them —
the substrate the *live* chord turns into an ``hv.Chord`` (``aegir.viz.lineup_data`` precomputes the
ring/sims; ``aegir.viz.lineup_app`` renders it, served by the bokeh server):
  • Terms   — collections sharing realized ontology terms
  • Content — collections sharing FinePDFs topics
  • Schema  — collections whose tables foreign-key into common downstream tables (the term↔table
              *many-to-many progress* signal; falls back to tables≈terms when no FK graph).

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

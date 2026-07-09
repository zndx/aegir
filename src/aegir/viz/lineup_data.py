"""Load-once data + TF-IDF chord substrate for the live lineup-chord Panel app.

``panel serve`` re-executes the *served script* (``lineup_app.py``) per browser session, but ``import``
is cached in ``sys.modules`` — so the expensive corpus load + TF-IDF precompute here runs **once per
server process**, not per session. The app module imports these globals and builds a chord per request
(cheap). Mirrors the substrate in ``aegir.lineup.chords`` / ``build`` (kept in lock-step deliberately).
"""
from __future__ import annotations

from collections import defaultdict

from aegir.lineup import build, chords as C, sources as S

_rows = S.load_ontology()
_corpus, _crun = S.corpus_recs()
_coverage, _cov = S.coverage_recs()
build.assign_topic_threads(_corpus, _coverage)   # lock-step with build.run() — organic topics
_notes, MAPS = build.project_collections(_corpus, _coverage)
TID_TABLE, FKS, COL_TOKENS = build._relational_spine(_rows)

_feats, FK_OK = C._features(MAPS, TID_TABLE, FKS, COL_TOKENS)
SIMS = {lens: (cids, s) for lens in C.LENSES for cids, s in [C._sim(_feats[lens])] if s is not None}

_strength: dict[str, float] = defaultdict(float)
for _cids, _s in SIMS.values():
    _tot = _s.sum(axis=1)
    for _i, _c in enumerate(_cids):
        _strength[_c] += float(_tot[_i])
RING = [c for c, _ in sorted(_strength.items(), key=lambda kv: -kv[1])[:30]]
RING_IDX = {c: i for i, c in enumerate(RING)}
NAME = {RING_IDX[c]: C._label(c) for c in RING}
STATUS = ("subgraph + column-vocabulary" if FK_OK
          else "fallback: tables≈terms (no spine run on disk)")

# ── TRUNK (scratch) substrate — the inverted topic layer's lineage, not a fitted pivot:
# term-grounded topics associated by the DOCUMENTS whose items bind to them (margin-gated
# associations; aegir.ontology.topic_layer). Same invariant chord shape, era-true data.
TRUNK_SIMS: dict = {}
TRUNK_RING: list = []
TRUNK_RING_IDX: dict = {}
TRUNK_NAME: dict = {}
try:
    _assoc = S.topic_associations()
    if _assoc:
        _by_topic: dict[str, set] = {}
        for _r in _assoc["records"]:
            if _r.get("assigned"):
                _by_topic.setdefault(_r.get("topic_code", ""), set()).add(_r.get("doc_hash", ""))
        _cids2, _s2 = C._sim({k: v for k, v in _by_topic.items() if k})
        if _s2 is not None:
            for _lens in C.LENSES:
                TRUNK_SIMS[_lens] = (_cids2, _s2)
            _st2: dict = defaultdict(float)
            _tot2 = _s2.sum(axis=1)
            for _i, _c in enumerate(_cids2):
                _st2[_c] += float(_tot2[_i])
            TRUNK_RING = [c for c, _ in sorted(_st2.items(), key=lambda kv: -kv[1])[:30]]
            TRUNK_RING_IDX = {c: i for i, c in enumerate(TRUNK_RING)}
            TRUNK_NAME = {TRUNK_RING_IDX[c]: (c[:22] + "…" if len(c) > 23 else c)
                          for c in TRUNK_RING}
except Exception:  # noqa: BLE001 — the trunk chord is optional enrichment
    pass
N_COLLECTIONS = len(MAPS.get("collections", []))

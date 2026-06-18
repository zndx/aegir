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
_notes, MAPS = build.project_collections(_corpus, _coverage)
TID_TABLE, FKS = build._relational_spine(_rows)

_feats, FK_OK = C._features(MAPS, TID_TABLE, FKS)
SIMS = {lens: (cids, s) for lens in C.LENSES for cids, s in [C._sim(_feats[lens])] if s is not None}

_strength: dict[str, float] = defaultdict(float)
for _cids, _s in SIMS.values():
    _tot = _s.sum(axis=1)
    for _i, _c in enumerate(_cids):
        _strength[_c] += float(_tot[_i])
RING = [c for c, _ in sorted(_strength.items(), key=lambda kv: -kv[1])[:30]]
RING_IDX = {c: i for i, c in enumerate(RING)}
NAME = {RING_IDX[c]: C._label(c) for c in RING}
STATUS = "fk-spanning" if FK_OK else "fallback: tables≈terms (low relational complexity)"
N_COLLECTIONS = len(MAPS.get("collections", []))

"""path — content-addressed lineup paths: the browsing sequence as a computation spec (RH 2026-07-21).

Ward's lineup made the browsing path the computational context ("computations use data found
before it in the lineup"), but FedWiki pages are mutable — a path is not reproducible across
time. Our panels are versioned STATIC projections of certified data products, so a path can be
a durable, federated, cacheable identity:

  * every kasten build emits a PATH MANIFEST: a version id (content hash over the sorted note-id
    universe) + the rank table. Rank-in-sorted-order IS a minimal perfect hash (a bijection
    ids ↔ [0, n)); a CHD/BBHash compaction (~3 bits/key, keys not shipped) is the wire-format
    upgrade for federation transfer — same identity, smaller carrier.
  * a TRAIL encodes as (version, [rank...]) → path_id = sha256(version ∥ ranks)[:16]. Two sites
    holding the same kasten version derive identical path ids with no coordination; a forked
    lineup shares its prefix — prefix hashes are the memoization keys for path-aware Metaflow
    steps (identical prefix ⇒ identical inputs ⇒ shared cached results, federation-wide).
  * decode resolves ranks → note ids → each panel's underlying data-product refs (frontmatter
    carries construct/tag/term provenance) — the LEFT-TO-RIGHT input set for a computation.

    from aegir.lineup.path import build_manifest, encode_trail, decode_path
"""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from pathlib import Path

from aegir.lineup import sources as S

MANIFEST = "path_manifest.json"


def _ids(kb: Path) -> "list[str]":
    idx = json.loads((kb / "index.json").read_text())
    return sorted({n["id"] for n in idx["notes"]})


def build_manifest(kb_dir: "Path | None" = None) -> dict:
    """Emit the kasten's path manifest. The version id is content-derived (sha256 over the
    newline-joined sorted id universe) — same projection bytes ⇒ same version ⇒ compatible
    path ids, on any site."""
    kb = Path(kb_dir or S.kb_dir())
    ids = _ids(kb)
    joined = "\n".join(ids).encode()
    version = hashlib.sha256(joined).hexdigest()[:16]
    man = {"version": version, "n": len(ids),
           "ids_sha256": hashlib.sha256(joined).hexdigest(),
           "mph": "rank-in-sorted-order (CHD compaction = wire-format upgrade)"}
    (kb / MANIFEST).write_text(json.dumps(man, indent=1))
    return man


def encode_trail(trail: "list[str]", kb_dir: "Path | None" = None) -> dict:
    """Trail → {path_id, version, ranks}. Raises KeyError on an id outside the kasten
    (a cross-version trail must be encoded per segment)."""
    kb = Path(kb_dir or S.kb_dir())
    ids = _ids(kb)
    man = json.loads((kb / MANIFEST).read_text()) if (kb / MANIFEST).exists() \
        else build_manifest(kb)
    ranks = []
    for t in trail:
        i = bisect_left(ids, t)
        if i >= len(ids) or ids[i] != t:
            raise KeyError(f"note id not in kasten version {man['version']}: {t!r}")
        ranks.append(i)
    payload = man["version"] + "∥" + ",".join(map(str, ranks))
    path_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    prefixes = [hashlib.sha256((man["version"] + "∥" + ",".join(map(str, ranks[:k + 1])))
                               .encode()).hexdigest()[:16] for k in range(len(ranks))]
    return {"path_id": path_id, "version": man["version"], "ranks": ranks,
            "prefix_ids": prefixes}


def decode_path(ranks: "list[int]", kb_dir: "Path | None" = None) -> "list[dict]":
    """Ranks → the ordered panel set with each panel's data-product provenance (the
    LEFT-OF-ME input universe for a path-aware step)."""
    kb = Path(kb_dir or S.kb_dir())
    ids = _ids(kb)
    idx = {n["id"]: n for n in json.loads((kb / "index.json").read_text())["notes"]}
    out = []
    for r in ranks:
        nid = ids[r]
        n = idx.get(nid, {})
        out.append({"id": nid, "kind": n.get("kind"), "data_product": n.get("data_product"),
                    "root": n.get("root"), "relpath": n.get("relpath")})
    return out

"""relational_collections — collections as connected DDL components, Barabási-calibrated (RH 2026-07-09).

A COLLECTION = the output documents whose tables/views form a connected relational graph
(FK + view-composition edges). The raw graph is scale-free (chapter generation performs
preferential attachment into shared lookups), so a giant hub-glued component absorbs most
docs. Two network-science filters recover domain-aligned components:

  - WEAK-TIES pruning (neighborhood/link overlap): keep an edge only when its endpoints
    share neighbors (the relationship is embedded in a local mesh); a spoke into a global
    hub has overlap ~0 and prunes itself.
  - TARGETED-ATTACK fragmentation: remove top-degree infrastructure hubs; the percolation
    curve S(f) locates the knee. Removed hubs are annotated as shared INFRASTRUCTURE
    (referenced by many collections, constitutive of none).

The threshold is chosen where the RELATIONAL axis agrees with the SEMANTIC axis: minimize
mean within-collection TOPIC ENTROPY (from the inverted topic layer's input associations —
construct pid ≡ harvest doc_hash[:16]) subject to the partition property (docs touch p50
one component) and a bounded giant component. Emits build/relational_collections/
{sweep.json, collections.json}.
"""
from __future__ import annotations

import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CORPUS = Path("/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus")
OUT = REPO / "build" / "relational_collections"
_FROM_JOIN = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.I)


def build_graph() -> "tuple[dict, dict]":
    """(adj: {table: set(table)}, docs_of: {table: set(pid)}) from the constructs web —
    FK edges + view-composition edges (all base tables of a view pairwise-connected
    through it; the view is the join-node, per the lineup's founding graph)."""
    adj: dict[str, set] = defaultdict(set)
    docs_of: dict[str, set] = defaultdict(set)
    for cj in sorted(glob.glob(str(CORPUS / "constructs" / "*.json"))):
        try:
            d = json.loads(Path(cj).read_text())
        except Exception:  # noqa: BLE001
            continue
        pid = Path(cj).stem[:16]
        known = {t.get("name") for t in d.get("tables") or [] if t.get("name")}
        for t in d.get("tables") or []:
            name = t.get("name")
            if not name:
                continue
            docs_of[name].add(pid)
            for fk in t.get("fks") or []:
                ref = fk.get("ref_table")
                if ref:
                    adj[name].add(ref)
                    adj[ref].add(name)
                    docs_of[ref].add(pid)
        for v in d.get("views") or []:
            if not isinstance(v, dict):
                continue
            bases = [b for b in _FROM_JOIN.findall(v.get("sql") or "") if b in known]
            for i in range(len(bases)):
                for j in range(i + 1, len(bases)):
                    adj[bases[i]].add(bases[j])
                    adj[bases[j]].add(bases[i])
    return adj, docs_of


def _components(adj: dict, removed: set, overlap_floor: float = 0.0) -> "list[set]":
    def keep_edge(a: str, b: str) -> bool:
        if overlap_floor <= 0:
            return True
        na = adj[a] - {b} - removed
        nb = adj[b] - {a} - removed
        u = len(na | nb)
        return (len(na & nb) / u if u else 0.0) >= overlap_floor
    seen: set = set()
    comps: list[set] = []
    for start in adj:
        if start in seen or start in removed:
            continue
        comp, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            for y in adj[x]:
                if y not in comp and y not in removed and keep_edge(x, y):
                    stack.append(y)
        seen |= comp
        comps.append(comp)
    return comps


def doc_topics() -> "dict[str, Counter]":
    """pid → topic Counter from the newest input association run (assigned only)."""
    runs = sorted((REPO / "build" / "topic_associations").glob("*/associations.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    out: dict[str, Counter] = defaultdict(Counter)
    if not runs:
        return out
    for ln in runs[0].read_text().splitlines():
        r = json.loads(ln)
        if r.get("assigned"):
            out[(r.get("doc_hash") or "")[:16]][r["topic_code"]] += 1
    return out


def evaluate(comps: "list[set]", docs_of: dict, topics: dict) -> dict:
    n_tables = sum(len(c) for c in comps)
    comp_docs: list[set] = []
    doc_comps: dict[str, int] = defaultdict(int)
    entropies, weights = [], []
    for c in comps:
        pids = set().union(*(docs_of.get(t, set()) for t in c)) if c else set()
        comp_docs.append(pids)
        for p in pids:
            doc_comps[p] += 1
        dist = Counter()
        for p in pids:
            dist.update(topics.get(p, {}))
        tot = sum(dist.values())
        if tot and len(pids) >= 2:
            h = -sum((v / tot) * math.log2(v / tot) for v in dist.values())
            entropies.append(h)
            weights.append(len(pids))
    sizes = sorted((len(c) for c in comps), reverse=True)
    dc = sorted(doc_comps.values())
    wmean_h = (sum(h * w for h, w in zip(entropies, weights)) / sum(weights)) if weights else 0.0
    return {
        "n_components": len(comps), "gc_tables": sizes[0] if sizes else 0,
        "gc_share": round((sizes[0] / n_tables) if n_tables else 0, 3),
        "collections_ge2_docs": sum(1 for p in comp_docs if len(p) >= 2),
        "docs_covered": len(doc_comps),
        "comps_per_doc_p50": dc[len(dc) // 2] if dc else 0,
        "comps_per_doc_max": dc[-1] if dc else 0,
        "mean_topic_entropy": round(wmean_h, 3),
    }


def main() -> int:
    adj, docs_of = build_graph()
    topics = doc_topics()
    deg = sorted(adj, key=lambda t: -len(adj[t]))
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = []
    for k in (0, 5, 10, 20, 40, 80, 160):
        for floor in (0.0, 0.05, 0.1):
            removed = set(deg[:k])
            comps = _components(adj, removed, floor)
            row = {"hubs_removed": k, "overlap_floor": floor,
                   **evaluate(comps, docs_of, topics)}
            sweep.append(row)
            print(json.dumps(row))
    (OUT / "sweep.json").write_text(json.dumps(sweep, indent=1))

    # operating point: bounded giant component + partition property, then min entropy
    ok = [r for r in sweep if r["gc_share"] <= 0.15 and r["comps_per_doc_p50"] <= 1
          and r["docs_covered"] >= 300]
    best = min(ok, key=lambda r: (r["mean_topic_entropy"], -r["collections_ge2_docs"])) if ok else None
    print("\nOPERATING POINT:", json.dumps(best))
    if best:
        removed = set(deg[: best["hubs_removed"]])
        comps = _components(adj, removed, best["overlap_floor"])
        import hashlib
        colls = []
        for c in sorted(comps, key=len, reverse=True):
            pids = sorted(set().union(*(docs_of.get(t, set()) for t in c)))
            if not pids:
                continue
            cid = hashlib.sha256(",".join(sorted(c)).encode()).hexdigest()[:12]
            colls.append({"id": cid, "n_tables": len(c), "tables": sorted(c)[:200],
                          "docs": pids})
        (OUT / "collections.json").write_text(json.dumps(
            {"operating_point": best, "infrastructure_hubs": deg[: best["hubs_removed"]],
             "collections": colls}, indent=1))
        print(f"collections.json: {len(colls)} collections · "
              f"infrastructure hubs: {deg[:min(8, best['hubs_removed'])]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

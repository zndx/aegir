"""Concept-congruence: qdrant-maxsim over OUTPUT chapters vs the INPUT window (RH 2026-07-05).

The original quality driver (BERTopic R_D topic alignment) reborn over the ColBERT
late-interaction index of GEPA-refined SKOS concept entries: classify the OUTPUT text against
the SAME collection the harvest classified the INPUTS against, and quantify how well the corpus
preserves the input window's concept associations.

The artifact is a TRIPARTITE LINEAGE GRAPH:

    input passage ──(maxsim @ harvest)──► concept entry ◄──(maxsim @ congruence)── chapter

Composing the two edge sets recovers passage→chapter concept paths; since the DERIVATION
lineage is known (chapter dir == passage hash), the graph is checkable: a faithful pipeline
has each chapter maxsim-associating with its own source passage's concepts.

Metrics (report-only first, per doctrine):
- top1_recovery — chapter's top concept == source passage's top concept
- profile_congruence — weighted overlap of the two top-k score profiles (0..1)
- corpus aggregates per register + the graph JSON (lineup/Atlas-ready)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_CODE_FENCE = re.compile(r"```.*?```", re.S)


def prose_of(chapter_md: str) -> str:
    """Strip the embedded payload (tables + SQL fences) — congruence scores the TEACHING
    prose against the concept space, not the row data."""
    t = _CODE_FENCE.sub(" ", chapter_md)
    t = _TABLE_ROW.sub(" ", t)
    return re.sub(r"\s+", " ", t)


def profile(text: str, *, top_k: int = 5, collection: "str | None" = None) -> "list[dict]":
    """The concept-association profile: top-k (code, label, score) via ColBERT MaxSim."""
    from aegir.ontology.domain_index import classify_hierarchical
    if collection is None:
        try:
            from aegir.strategy.manifest import lens_binding
            collection = lens_binding().get("vocab_collection")
        except Exception:  # noqa: BLE001
            collection = None
    h = classify_hierarchical(text[:4000], top_k=top_k,
                              **({"collection": collection} if collection else {}))
    return [{"code": x.get("code"), "label": x.get("pref_label"), "score": round(x.get("score", 0), 4)}
            for x in (h.get("hits") or [])]


def _overlap(a: "list[dict]", b: "list[dict]") -> float:
    """Weighted profile congruence: sum of min-normalized shared-concept mass (0..1)."""
    wa = {x["code"]: x["score"] for x in a}
    wb = {x["code"]: x["score"] for x in b}
    ta, tb = sum(wa.values()) or 1.0, sum(wb.values()) or 1.0
    return round(sum(min(wa[c] / ta, wb[c] / tb) for c in set(wa) & set(wb)), 4)


def congruence_report(run_dir: Path, *, harvest_docs: Path | None = None,
                      top_k: int = 5) -> dict:
    """Score every chapter in ``run_dir/chapters`` against the concept collection, join with
    the source-passage profiles, and emit metrics + the tripartite graph."""
    run_dir = Path(run_dir)
    docs = harvest_docs or Path("build/domain_harvest/docs")
    nodes_p, nodes_c, nodes_ch, edges = {}, {}, [], []
    rows = []
    pass_profiles: "dict[str, list[dict]]" = {}
    for cdir in sorted((run_dir / "chapters").iterdir()):
        pid = cdir.name
        if pid not in pass_profiles:
            src = next(iter(docs.glob(pid + "*.txt")), None)
            if src is None:
                continue
            pass_profiles[pid] = profile(src.read_text())
            nodes_p[pid] = {"id": f"passage:{pid}", "kind": "passage"}
            for h in pass_profiles[pid]:
                nodes_c.setdefault(h["code"], {"id": f"concept:{h['code']}",
                                               "kind": "concept", "label": h["label"]})
                edges.append({"src": f"passage:{pid}", "dst": f"concept:{h['code']}",
                              "kind": "harvest_maxsim", "score": h["score"]})
        pp = pass_profiles[pid]
        for reg in ("natural", "semantic"):
            md = cdir / f"{reg}.md"
            if not md.exists():
                continue
            cp = profile(prose_of(md.read_text()))
            ch_id = f"chapter:{pid}/{reg}"
            nodes_ch.append({"id": ch_id, "kind": "chapter", "register": reg})
            for h in cp:
                nodes_c.setdefault(h["code"], {"id": f"concept:{h['code']}",
                                               "kind": "concept", "label": h["label"]})
                edges.append({"src": ch_id, "dst": f"concept:{h['code']}",
                              "kind": "congruence_maxsim", "score": h["score"]})
            rows.append({
                "passage": pid, "register": reg,
                "top1_recovery": bool(pp and cp and pp[0]["code"] == cp[0]["code"]),
                "profile_congruence": _overlap(pp, cp),
                "passage_top": pp[0]["label"] if pp else None,
                "chapter_top": cp[0]["label"] if cp else None,
            })
    by_reg: dict = {}
    for reg in ("natural", "semantic"):
        rs = [r for r in rows if r["register"] == reg]
        if rs:
            by_reg[reg] = {
                "n": len(rs),
                "top1_recovery_rate": round(sum(r["top1_recovery"] for r in rs) / len(rs), 3),
                "mean_profile_congruence": round(
                    sum(r["profile_congruence"] for r in rs) / len(rs), 3),
            }
    report = {"per_register": by_reg, "rows": rows}
    graph = {"nodes": list(nodes_p.values()) + list(nodes_c.values()) + nodes_ch,
             "edges": edges}
    (run_dir / "congruence.json").write_text(json.dumps(report, indent=1))
    (run_dir / "concept_graph.json").write_text(json.dumps(graph, indent=1))
    return report


def cross_congruence(dir_a: "Path", dir_b: "Path", *, main_ref: str = "",
                     shadow_ref: str = "", sample: int = 60) -> dict:
    """The clearinghouse's symmetrized judge: score A's chapters against B's vocab
    collection and vice versa. Reflexive congruence is self-grading; the CROSS cells are
    the comparison. Collections resolve from each side's strategy ref."""
    from aegir.strategy.manifest import lens_binding
    coll_a = lens_binding(main_ref or None).get("vocab_collection")
    coll_b = lens_binding(shadow_ref or None).get("vocab_collection")
    out = {"collections": {"main": coll_a, "shadow": coll_b}}
    for arm, d, coll in (("main_vs_shadow_lens", Path(dir_a), coll_b),
                         ("shadow_vs_main_lens", Path(dir_b), coll_a)):
        scores = []
        mds = sorted((d / "chapters").rglob("natural.md"))[:sample]
        for md in mds:
            try:
                pr = profile(prose_of(md.read_text()), collection=coll)
                if pr:
                    scores.append(pr[0]["score"])
            except Exception:  # noqa: BLE001
                continue
        out[arm] = {"n": len(scores),
                    "mean_top1": round(sum(scores) / len(scores), 4) if scores else None}
    return out

#!/usr/bin/env python
"""audit_reverse_chain — Phase 2: reverse-verification of the core thesis chain
(RH 2026-07-23): Textbooks ← Relational ← Ontology ← Passages ← FinePDFs, every hop by
RECORDED link only (confirmed-links rule; the one-way doctrine — verification READS
backward along recorded lineage, it creates no edges). Full census over the release's
passages, per-segment scores + failure classification.

    uv run python scripts/audit_reverse_chain.py → build/reverse_verification.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = Path("/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus-v06")


def main() -> int:
    pids = sorted(p.stem for p in (RUN / "entities").glob("*.json"))
    chapters = {p.name for p in (RUN / "chapters").iterdir()} if (RUN / "chapters").exists() else set()
    # released-spine table universe (the cross-pathway seam probe)
    spine_tables: set = set()
    try:
        import pyarrow.parquet as pq
        nm = pq.read_table(sorted((REPO / "corpora/ddl").glob("*/naming_map.parquet"))[-1])
        spine_tables = {str(t) for t in nm.column("table_name").to_pylist()}
    except Exception:  # noqa: BLE001
        pass
    # admitting anchors (P→F): manifest hash-prefix → code; code→IRI→pid via released lens
    manifest: dict = {}
    try:
        from aegir.ontology import domain_index as DI
        notat2iri = {str(getattr(c, "code", "")): str(k)
                     for k, c in DI.load_skos(str(DI.DEFAULT_OVERLAY)).items()
                     if getattr(c, "code", "")}
        snap = json.loads((REPO / "strategy/components/lens/aperture.snapshot.json").read_text())
        rows_ = snap if isinstance(snap, list) else snap.get("points") or []
        iri2pid = {r.get("iri"): str(r.get("id")) for r in rows_ if r.get("iri")}
        for line in (REPO / "build/domain_harvest/manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            manifest[str(r.get("hash", ""))[:16]] = iri2pid.get(
                notat2iri.get(str(r.get("code", ""))))
    except Exception:  # noqa: BLE001
        pass

    seg = {k: {"ok": 0, "fail": 0} for k in ("T_R", "R_O", "O_P", "P_F")}
    fails: dict = {"missing_record": [], "in_record_mismatch": [], "cross_pathway_seam": 0}
    per_pid = []
    for pid in pids:
        row = {"pid": pid}
        # T←R: a chapter exists for this pid AND its construct record names its tables
        ch = [c for c in chapters if pid[:12] in c]
        con_p = RUN / "constructs" / f"{pid}.json"
        con = json.loads(con_p.read_text()) if con_p.exists() else None
        t_ok = bool(ch) and bool(con and (con.get("tables") or con.get("views")))
        seg["T_R"]["ok" if t_ok else "fail"] += 1
        if not t_ok:
            fails["missing_record"].append({"pid": pid, "segment": "T_R",
                                            "chapter": bool(ch), "constructs": bool(con)})
        row["T_R"] = t_ok
        # R←O: every construct column carries a recorded concept resolvable IN-RECORD
        r_ok = False
        if con:
            ents = {str(e.get("name") if isinstance(e, dict) else e).split(":", 1)[-1]
                    for e in (con.get("entities") or [])}
            omn = con.get("ontology_omn") or ""
            cols = [c for t in (con.get("tables") or []) for c in (t.get("columns") or [])]
            tnames = {t.get("name") for t in (con.get("tables") or [])}
            # the constructs web's recording conventions (measured, not assumed):
            # entity columns record the CLASS; FK columns record the TARGET TABLE
            # (in-record); created/updated/id are plumbing — no ontology claim made.
            PLUMBING = {"created", "updated", "id"}
            with_c = [c for c in cols
                      if c.get("concept") and c["concept"] not in PLUMBING]
            resolved = [c for c in with_c
                        if c["concept"] in ents or f"sdg:{c['concept']}" in omn
                        or c["concept"] in omn or c["concept"] in tnames]
            row["cols"] = len(cols)
            row["cols_with_concept"] = len(with_c)
            row["concepts_resolved_in_record"] = len(resolved)
            r_ok = bool(cols) and len(with_c) >= max(1, int(0.5 * len(cols))) and \
                len(resolved) >= max(1, int(0.8 * len(with_c)))
            if with_c and len(resolved) < len(with_c):
                fails["in_record_mismatch"].append(
                    {"pid": pid, "segment": "R_O",
                     "unresolved": sorted({c["concept"] for c in with_c
                                           if c not in resolved})[:4]})
            # cross-pathway seam measurement (not a chain failure): construct tables
            # unknown to the RELEASED spine
            tnames = {t.get("name") for t in (con.get("tables") or [])}
            fails["cross_pathway_seam"] += sum(1 for t in tnames if t not in spine_tables)
        seg["R_O"]["ok" if r_ok else "fail"] += 1
        row["R_O"] = r_ok
        # O←P: the derive record exists and is non-empty (the classes' authoring source)
        ent_p = RUN / "entities" / f"{pid}.json"
        e_ok = False
        if ent_p.exists():
            try:
                e_ok = bool(json.loads(ent_p.read_text()).get("entities"))
            except Exception:  # noqa: BLE001
                e_ok = False
        seg["O_P"]["ok" if e_ok else "fail"] += 1
        row["O_P"] = e_ok
        # P←F: the harvest doc exists AND the admitting anchor resolves (manifest record)
        doc = list((REPO / "build/domain_harvest/docs").glob(f"{pid}*.txt"))
        p_ok = bool(doc) and manifest.get(pid) is not None
        seg["P_F"]["ok" if p_ok else "fail"] += 1
        if not p_ok:
            fails["missing_record"].append({"pid": pid, "segment": "P_F",
                                            "doc": bool(doc),
                                            "admitting_anchor": manifest.get(pid)})
        row["P_F"] = p_ok
        per_pid.append(row)

    scores = {k: round(v["ok"] / max(1, v["ok"] + v["fail"]), 4) for k, v in seg.items()}
    full = sum(1 for r in per_pid if all(r.get(k) for k in ("T_R", "R_O", "O_P", "P_F")))
    out = {"doctrine": "recorded links only; verification reads backward, creates no edges",
           "n_passages": len(pids),
           "segment_scores": scores,
           "full_chain_verified": full,
           "full_chain_rate": round(full / max(1, len(pids)), 4),
           "failures": {"missing_record": fails["missing_record"][:24],
                        "n_missing_record": len(fails["missing_record"]),
                        "in_record_mismatch": fails["in_record_mismatch"][:24],
                        "n_in_record_mismatch": len(fails["in_record_mismatch"]),
                        "cross_pathway_seam_tables": fails["cross_pathway_seam"]},
           "per_pid": per_pid}
    (REPO / "build/reverse_verification.json").write_text(json.dumps(out, indent=1))
    print(f"chain: Textbooks←Relational {scores['T_R']:.1%} · Relational←Ontology "
          f"{scores['R_O']:.1%} · Ontology←Passages {scores['O_P']:.1%} · "
          f"Passages←FinePDFs {scores['P_F']:.1%}")
    print(f"FULL CHAIN verified end-to-end: {full}/{len(pids)} ({out['full_chain_rate']:.1%})")
    print(f"failures: {out['failures']['n_missing_record']} missing-record · "
          f"{out['failures']['n_in_record_mismatch']} in-record mismatch · "
          f"{fails['cross_pathway_seam']} cross-pathway seam tables (constructs∉spine)")
    print("→ build/reverse_verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

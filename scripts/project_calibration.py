#!/usr/bin/env python
"""Project a calibration run's provenance into the aegir_hx graph.

Reads ``build/<run>/calibration_log.jsonl`` + the ontology catalog and writes the
chapter-level provenance — Run / Topic / Template / Family / Chapter nodes and
SEEDED_BY / SELECTED / IN_FAMILY / PRODUCED / RE_GROUNDS_TO edges — into ``aegir_hx``
via the governance graph adapter. Idempotent (MERGE). Verifier scores ride as node/
edge properties: the Atlas-classification + OpenLineage-facet surface the UI reads.

    uv run --no-sync python scripts/project_calibration.py --run build/calibration_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.governance import graph as G          # noqa: E402
from aegir.ontology.schema import load_catalog    # noqa: E402


def template_family_map(catalog) -> dict:
    return {t.template_id: (t.provenance or {}).get("family", "?") for t in catalog.templates}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="build/calibration_v1")
    args = ap.parse_args()

    run_dir = REPO / args.run if not Path(args.run).is_absolute() else Path(args.run)
    log = [json.loads(ln) for ln in (run_dir / "calibration_log.jsonl").read_text().splitlines() if ln.strip()]
    catalog = load_catalog(REPO / "src/aegir/ontology/catalog/combined.json")
    fam = template_family_map(catalog)
    scored = [r for r in log if "error" not in r]

    with G.connect() as conn:
        n_ch = 0
        for r in scored:
            run_id = f"cal_{r['episode']}"
            topic = r["target_topic"]
            rw = r.get("rewards", {})
            G.merge_node(conn, "Topic", {"topic_id": topic}, {"qualifiedName": f"topic:{topic}"})
            G.merge_node(conn, "Run", {"run_id": run_id}, {
                "admitted": bool(r.get("admitted")), "latency_s": r.get("latency_s", 0.0),
                "topic_recovery": r.get("topic_recovery", 0.0), "r_axiom": r.get("r_axiom", 0.0)})
            G.merge_edge(conn, "Run", {"run_id": run_id}, "SEEDED_BY", "Topic", {"topic_id": topic})
            for ref in r.get("refs", []):
                f = fam.get(ref, r.get("primary_family", "?"))
                G.merge_node(conn, "Template", {"template_id": ref})
                G.merge_node(conn, "Family", {"name": f})
                G.merge_edge(conn, "Template", {"template_id": ref}, "IN_FAMILY", "Family", {"name": f})
                G.merge_edge(conn, "Run", {"run_id": run_id}, "SELECTED", "Template", {"template_id": ref})
            if r.get("admitted"):
                G.merge_node(conn, "Chapter", {"chapter_id": run_id}, {
                    "qualifiedName": f"chapter:{run_id}", "primary_family": r.get("primary_family", "?"),
                    "r_axiom": r.get("r_axiom", 0.0), "claim_grounding": rw.get("claim_grounding", 0.0),
                    "cross_modal": rw.get("cross_modal", 0.0)})
                G.merge_edge(conn, "Run", {"run_id": run_id}, "PRODUCED", "Chapter", {"chapter_id": run_id})
                G.merge_edge(conn, "Chapter", {"chapter_id": run_id}, "RE_GROUNDS_TO", "Topic", {"topic_id": topic}, {
                    "cosine": r.get("topic_cosine", 0.0), "rank": r.get("topic_rank", -1),
                    "hit_at_1": bool(r.get("topic_recovery", 0) >= 1.0)})
                n_ch += 1
        print(f"projected {len(scored)} runs, {n_ch} chapters into aegir_hx")

        # ── first points on the board: the graph answers the questions that mattered ──
        print("\n=== graph census ===")
        for lbl in ("Run", "Chapter", "Topic", "Template", "Family"):
            print(f"  {lbl}: {G.scalar(conn, f'MATCH (n:{lbl}) RETURN count(n)')}")

        print("\n=== loop closure (the FinePDFs↔corpus cycle, made queryable) ===")
        edges = G.scalar(conn, "MATCH (:Chapter)-[r:RE_GROUNDS_TO]->(:Topic) RETURN count(r)")
        hit1 = G.scalar(conn, "MATCH (:Chapter)-[r:RE_GROUNDS_TO]->(:Topic) WHERE r.rank = 0 RETURN count(r)")
        print(f"  RE_GROUNDS_TO edges: {edges}   hit@1 (rank=0): {hit1}")

        print("\n=== cockpit: admitted chapters by family ===")
        for f, c in Counter(G.run(conn, "MATCH (c:Chapter) RETURN c.primary_family")).most_common():
            print(f"  {f}: {c}")

        print("\n=== invariant audit: admitted chapters re-grounding worse than hit@5 (refine queue) ===")
        print(f"  rank>=5: {G.scalar(conn, 'MATCH (:Chapter)-[r:RE_GROUNDS_TO]->(:Topic) WHERE r.rank >= 5 RETURN count(r)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

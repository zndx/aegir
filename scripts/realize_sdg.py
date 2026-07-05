#!/usr/bin/env python
"""Realize the sdg entity-derived ontology: merge → HermiT certify → DDL/shapes → score.

The realize boundary for the entity-centric (post-template) derivation. Inputs are per-passage
entity JSONs (``derive_loop`` output); outputs are the SHIPPED artifacts — the trust doctrine
holds: kvasir fast-refutes in-loop, **HermiT signs the certificate** here.

  build/sdg/<run>/entities/*.json
    → merge_entities → sdg-ontology.omn        (the real ontology, HermiT/DeepOnto-feedable)
    → HermiT                → certificate.json    (consistent, n_classes, unsat)
    → kvasir ddl --sql      → ddl.sql             (proof-carrying, sqlparser-self-checked)
    → kvasir shapes         → shapes.ttl          (SHACL Core — the fixpoint/constraint view)
    → score_ontology_ddl    → structure.json      (width dist + shape EMD vs SchemaPile)

Run: LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/realize_sdg.py \
       --entities-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KVASIR = REPO / "components/kvasir/target/release/kvasir"
if str(REPO) not in sys.path:  # `scripts.` imports when exec'd as a file (sys.path[0] = scripts/)
    sys.path.insert(0, str(REPO))


def realize(entities_dir: Path, output_dir: Path, *, skip_hermit: bool = False) -> dict:
    from aegir.ontology.derive_loop import merge_entities
    from aegir.ontology.entities import from_json, to_manchester

    sets = []
    for p in sorted(entities_dir.glob("*.json")):
        obj = json.loads(p.read_text())
        sets.append(from_json(obj if "entities" in obj else {"entities": obj}))
    merged = merge_entities(sets)
    omn = to_manchester(merged)
    output_dir.mkdir(parents=True, exist_ok=True)
    omn_path = output_dir / "sdg-ontology.omn"
    omn_path.write_text(omn)
    print(f"merged {sum(len(s) for s in sets)} entities from {len(sets)} passages "
          f"→ {len(merged)} classes → {omn_path}")

    cert: dict = {"reasoner": "HermiT", "skipped": skip_hermit}
    if not skip_hermit:
        from aegir.ontology.deeponto_harness import ensure_jvm
        ensure_jvm()  # MUST precede any deeponto import (click.prompt hangs non-interactively)
        from scripts.build_realized_ontology import _reason
        _onto, _tmp, consistent, n_classes, unsat, why = _reason(omn, explain=bool(0))
        if int(n_classes) == 0 and merged:
            # VACUOUS-PARSE TRIPWIRE: OWLAPI silently loaded zero frames (missing declarations,
            # illegal punning, …) — "consistent" would certify an EMPTY doc. Fail loud.
            raise RuntimeError(
                f"vacuous HermiT parse: {len(merged)} merged classes but OWLAPI loaded 0 — "
                "the certificate would be meaningless; inspect the omn for punning/declaration "
                "issues (OWLAPI warnings name the entities)")
        cert.update({"isConsistent": bool(consistent), "n_classes": int(n_classes),
                     "unsat": list(unsat or [])})
        print(f"HermiT: consistent={consistent} classes={n_classes} unsat={len(unsat or [])}")
        if unsat:
            cert["why"] = why or {}
    (output_dir / "certificate.json").write_text(json.dumps(cert, indent=2))

    for sub, out in (("ddl", "ddl.sql"), ("shapes", "shapes.ttl")):
        args = [str(KVASIR), sub, str(omn_path)] + (["--sql"] if sub == "ddl" else [])
        r = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            (output_dir / out).write_text(r.stdout)
        else:
            print(f"WARN kvasir {sub} failed: {r.stderr[:200]}", file=sys.stderr)

    from scripts.score_ontology_ddl import score
    structure = score(omn_path)
    (output_dir / "structure.json").write_text(json.dumps(structure, indent=2, default=str))
    print(f"structure: {structure['n_elected']} elected | {structure['total_fks']} FKs | "
          f"{structure['n_junctions']} junctions | {structure['n_lookups']} lookups | "
          f"median {structure['width']['median']} | EMD {structure['shape_emd']}")
    return {"n_classes": len(merged), "certificate": cert, "structure": structure,
            "omn": str(omn_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--skip-hermit", action="store_true",
                    help="fast pre-flight only (kvasir); the certificate step needs the JVM")
    a = ap.parse_args()
    res = realize(a.entities_dir, a.output_dir, skip_hermit=a.skip_hermit)
    ok = res["certificate"].get("isConsistent", True) is not False
    return 0 if ok else 2


if __name__ == "__main__":
    rc = main()
    from aegir.utils.clean_exit import clean_exit
    clean_exit(rc)

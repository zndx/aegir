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
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KVASIR = REPO / "components/kvasir/target/release/kvasir"
if str(REPO) not in sys.path:  # `scripts.` imports when exec'd as a file (sys.path[0] = scripts/)
    sys.path.insert(0, str(REPO))


def _splice_ontology(base_omn: str, extra_path: Path) -> str:
    """GENERATION UNIFICATION (RH 2026-07-21): OMN-text union of an additional realized
    ontology (the catalog realization) into the entity-derived doc. The extra doc's prefix
    header + Ontology: frame are stripped; missing prefixes are added to the base header;
    duplicate bare property/annotation declarations dedup; repeated Class: frames are LEGAL
    OMN (axiom union) — shared-IRI collisions surface as HermiT signals, which is the point."""
    extra = extra_path.read_text()
    base_pfx = dict(re.findall(r"^Prefix:\s*(\w*):\s*<([^>]+)>", base_omn, re.M))
    add_pfx = []
    for k, v in re.findall(r"^Prefix:\s*(\w*):\s*<([^>]+)>", extra, re.M):
        if k not in base_pfx:
            add_pfx.append(f"Prefix: {k}: <{v}>")
            base_pfx[k] = v
    body = re.sub(r"^Prefix:[^\n]*\n", "", extra, flags=re.M)
    body = re.sub(r"^Ontology:[^\n]*\n", "", body, flags=re.M)
    base_kind = {m.group(2): m.group(1) for m in
                 re.finditer(r"^(Object|Data|Annotation)Property:\s*(\S+)$", base_omn, re.M)}
    # cross-generation PROPERTY PUNS (same sdg: name, different kind) are illegal in OWL 2 DL
    # and collapse reasoners globally — shim the SPLICED side to <name>__cat (loud, worklisted;
    # the semantic merge of each pun is explicit review work, never a silent rewrite)
    conflicts = []
    for m in re.finditer(r"^(Object|Data|Annotation)Property:\s*(\S+)$", body, re.M):
        k, nm = m.group(1), m.group(2)
        if nm in base_kind and base_kind[nm] != k:
            conflicts.append({"property": nm, "base_kind": base_kind[nm], "spliced_kind": k})
    for c in conflicts:
        nm = c["property"]
        body = re.sub(rf"(?<![\w:]){re.escape(nm)}(?![\w])", nm + "__cat", body)
        print(f"  PUN SHIM: {nm} ({c['base_kind']} in base vs {c['spliced_kind']} in spliced) "
              f"→ spliced side renamed {nm}__cat (review worklist)")
    if conflicts:
        import json as _json
        Path("build/unification_worklist.json").write_text(_json.dumps(
            {"property_kind_conflicts": conflicts}, indent=1))
    base_decls = set(re.findall(r"^((?:Object|Data|Annotation)Property: \S+)$", base_omn, re.M))
    body_lines = []
    for ln in body.split("\n"):
        if ln.strip() and re.fullmatch(r"(?:Object|Data|Annotation)Property: \S+", ln.strip()) \
                and ln.strip() in base_decls:
            continue
        body_lines.append(ln)
    if add_pfx:
        base_omn = base_omn.replace("Ontology:", "\n".join(add_pfx) + "\nOntology:", 1)
    return base_omn + "\n\n" + "\n".join(body_lines)


def realize(entities_dir: Path, output_dir: Path, *, skip_hermit: bool = False,
            merge_ontology: "Path | None" = None, ground_entity_props: bool = False) -> dict:
    from aegir.ontology.derive_loop import merge_entities
    from aegir.ontology.entities import from_json, to_manchester

    sets = []
    for p in sorted(entities_dir.glob("*.json")):
        obj = json.loads(p.read_text())
        sets.append(from_json(obj if "entities" in obj else {"entities": obj}))
    merged = merge_entities(sets)
    omn = to_manchester(merged)
    if merge_ontology:
        omn = _splice_ontology(omn, merge_ontology)
        print(f"unified: merged ontology {merge_ontology} spliced (prefixes reconciled)")
    if ground_entity_props:
        # phase-2 signal harvest: ground the entity-side bare sdg: properties by stem so the
        # armed signatures (and staged domains) BITE at entity scale; unsat = worklist
        from aegir.ontology.relation_signatures import grounding_frames
        props = set(re.findall(r"^ObjectProperty:\s*sdg:(\w+)$", omn, re.M))
        g_omn, loose = grounding_frames(props)
        if g_omn:
            omn += "\n\n" + g_omn
        print(f"grounded {len(props) - len(loose)}/{len(props)} entity properties by stem")
    output_dir.mkdir(parents=True, exist_ok=True)
    omn_path = output_dir / "sdg-ontology.omn"
    omn_path.write_text(omn)
    print(f"merged {sum(len(s) for s in sets)} entities from {len(sets)} passages "
          f"→ {len(merged)} classes → {omn_path}")

    # ARTIFACTS FIRST (RH shakedown doctrine): shapes/ddl/structure land even when the
    # certificate refuses — exit-3-before-kvasir would blind the unification shakedown.
    for sub, out in (("ddl", "ddl.sql"), ("shapes", "shapes.ttl")):
        args = [str(KVASIR), sub, str(omn_path)] + (["--sql"] if sub == "ddl" else [])
        r = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            (output_dir / out).write_text(r.stdout)
        else:
            print(f"WARN kvasir {sub} failed: {r.stderr[:200]}", file=sys.stderr)

    structure: dict = {}
    try:
        from scripts.score_ontology_ddl import score
        structure = score(omn_path)
        (output_dir / "structure.json").write_text(json.dumps(structure, indent=2, default=str))
        print(f"structure: {structure['n_elected']} elected | {structure['total_fks']} FKs | "
              f"{structure['n_junctions']} junctions | {structure['n_lookups']} lookups | "
              f"median {structure['width']['median']} | EMD {structure['shape_emd']}")
    except Exception as e:  # noqa: BLE001 — shakedown doctrine: a refused score never blinds
        print(f"WARN structure scoring unavailable: {str(e)[:200]}", file=sys.stderr)

    cert: dict = {"reasoner": "HermiT", "skipped": skip_hermit}
    if not skip_hermit:
        from aegir.ontology.deeponto_harness import ensure_jvm
        ensure_jvm()  # MUST precede any deeponto import (click.prompt hangs non-interactively)
        from scripts.build_realized_ontology import _reason
        _onto, _tmp, consistent, n_classes, unsat, why = _reason(omn, explain=bool(0))
        if int(n_classes) == 0 and merged:
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
            names = [u.rsplit('#', 1)[-1] for u in list(unsat)[:5]]
            raise SystemExit(
                f"REFUSED: {len(unsat)} unsatisfiable classes (first: {names}) — a sick TBox "
                "is not scored or shipped; certificate.json carries the full list (exit 3)")
    (output_dir / "certificate.json").write_text(json.dumps(cert, indent=2))

    return {"n_classes": len(merged), "certificate": cert, "structure": structure,
            "omn": str(omn_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--skip-hermit", action="store_true",
                    help="fast pre-flight only (kvasir); the certificate step needs the JVM")
    ap.add_argument("--merge-ontology", type=Path, default=None,
                    help="GENERATION UNIFICATION: splice an additional realized OMN (the "
                         "catalog realization) into the union before certify/emit")
    ap.add_argument("--ground-entity-props", action="store_true",
                    help="ground bare entity sdg: properties by stem (signatures/domains bite; "
                         "phase-2 signal harvest)")
    a = ap.parse_args()
    res = realize(a.entities_dir, a.output_dir, skip_hermit=a.skip_hermit,
                  merge_ontology=a.merge_ontology, ground_entity_props=a.ground_entity_props)
    ok = res["certificate"].get("isConsistent", True) is not False
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        print(e, file=sys.stderr)
        rc = 3
    from aegir.utils.clean_exit import clean_exit
    clean_exit(rc)

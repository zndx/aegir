#!/usr/bin/env python
"""Promote derived ontology candidates into the live catalog — close the loop at the system boundary.

The content-first deriver stages admitted primitives in ``08_derived.candidate.json``, but EVERY downstream
consumer (``generate_chapter``, ``build_ddl_spine``, ``build_verbalization_frames``) skips ``.candidate``
files and nothing assembled ``combined.json`` — so "grow the ontology" was append-to-a-file-nobody-loads
(derivation audit 2026-06-20, the Phase-0 must-fix). This promotes candidates through the deep gate into a
clean family file consumers actually read, and rebuilds ``combined.json``:

  candidate → re-gate (DeepOnto parse, G1) → HermiT consistency (BFO/CCO ∪ primitive; the inc-2a oracle)
            → strip ``_``-prefixed staging fields → clean ``08_derived.json`` (valid CatalogTemplates)
            → rebuild ``combined.json`` (glob ``0*.json`` minus candidate/combined, merge null_stats).

After promotion, ``build_ddl_spine`` / ``generate_chapter`` (which glob ``0*.json``) see the derived family
automatically; run ``build_verbalization_frames`` (glob widened to ``0*``) to attach verbalization frames.

    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/promote_candidates.py            # full (HermiT)
    uv run --no-sync python scripts/promote_candidates.py --no-hermit                                       # CPU gates only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.ontology import derivation_membrane as DM  # noqa: E402
from aegir.ontology.schema import Catalog, CatalogTemplate, load_catalog, save_catalog  # noqa: E402

_CATALOG_DIR = REPO / "src" / "aegir" / "ontology" / "catalog"
_DERIVED_VERSION = "0.3.0-derived"


def _to_template(d: dict) -> CatalogTemplate:
    return CatalogTemplate(
        template_id=d["template_id"], manchester_template=d["manchester_template"],
        slot_types=d.get("slot_types") or {}, is_complex=bool(d.get("is_complex")),
        verbal_template=d.get("verbal_template", ""), bfo_anchor_path=d.get("bfo_anchor_path") or [])


def build_combined() -> dict:
    """Assemble ``combined.json`` from all family files (``0*.json`` minus candidate/combined), merging
    null_stats. This is the assembler the pipeline was missing."""
    fams = sorted(p for p in _CATALOG_DIR.glob("0*.json")
                  if ".candidate" not in p.name and "combined" not in p.name)
    templates: list[CatalogTemplate] = []
    null_stats: dict = {}
    version = "0.3.0"
    for f in fams:
        cat = load_catalog(f)
        if f.name.startswith("01"):
            version = cat.version
        templates.extend(cat.templates)
        null_stats.update(cat.null_stats or {})
    save_catalog(Catalog(version=version, templates=templates, null_stats=null_stats),
                 _CATALOG_DIR / "combined.json")
    return {"families": [f.name for f in fams], "n_templates": len(templates)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="08_derived.candidate.json")
    ap.add_argument("--out", default="08_derived.json")
    ap.add_argument("--no-hermit", dest="hermit", action="store_false", help="skip HermiT consistency (CPU gates only)")
    ap.add_argument("--no-jvm", dest="jvm", action="store_false", help="skip the DeepOnto re-gate (G1 parse)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cand_path = _CATALOG_DIR / args.candidate
    if not cand_path.exists():
        print(f"no candidate file {cand_path.relative_to(REPO)} — nothing to promote", file=sys.stderr)
        return 1
    cands = json.loads(cand_path.read_text()).get("templates", [])
    print(f"promoting from {cand_path.name}: {len(cands)} candidates")

    coherence = None
    if args.hermit:
        try:
            from mediate_consistency import coherence as _coh
            from aegir.ontology.deeponto_harness import ensure_jvm
            ensure_jvm()
            coherence = _coh
        except Exception as e:  # noqa: BLE001
            print(f"  HermiT unavailable ({type(e).__name__}); promoting on CPU/parse gates only", file=sys.stderr)

    promoted: list[CatalogTemplate] = []
    funnel = {"in": len(cands), "regate": 0, "hermit": 0, "promoted": 0}
    for d in cands:
        ct = _to_template(d)
        g = DM.content_membrane(ct, source_span=d.get("_source_span", ""), jvm=args.jvm)
        g1_ok = g["g1_well_formed"] and g["g1_deeponto_parses"] is not False
        if not (g1_ok and g["g2_is_complex_class"] and g["clean_room"] and g["anchor_valid"]):
            print(f"  ✘ re-gate {ct.template_id[:36]:36s} g1={g1_ok} g2={g['g2_is_complex_class']} {g['g1_reason']}")
            continue
        funnel["regate"] += 1
        if coherence is not None:
            res = coherence(ct)
            if not res.get("consistent"):
                print(f"  ✘ HermiT {ct.template_id[:36]:36s} {res.get('error') or res.get('unsat')}")
                continue
            funnel["hermit"] += 1
        promoted.append(ct)
        funnel["promoted"] += 1
        print(f"  ✓ {ct.template_id}")

    print(f"\nFUNNEL: {funnel}")
    if args.dry_run:
        print("(dry run — no files written)")
        return 0
    if not promoted:
        print("nothing promoted; combined.json unchanged")
        return 0

    out_path = _CATALOG_DIR / args.out
    existing_cat = load_catalog(out_path) if out_path.exists() else None
    existing = list(existing_cat.templates) if existing_cat else []
    version = existing_cat.version if existing_cat else _DERIVED_VERSION
    seen = {t.template_id for t in existing}
    merged = existing + [t for t in promoted if t.template_id not in seen]
    save_catalog(Catalog(version=version, templates=merged, null_stats={}), out_path)
    print(f"wrote {len(merged)} templates ({len(merged) - len(existing)} new) → {out_path.relative_to(REPO)}")

    comb = build_combined()
    print(f"rebuilt combined.json: {comb['n_templates']} templates across {len(comb['families'])} families "
          f"({comb['families']})")
    print("  → derived family is now in combined.json + globbed by build_ddl_spine/generate_chapter. "
          "Run build_verbalization_frames (glob widened to 0*) to attach verbalization frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

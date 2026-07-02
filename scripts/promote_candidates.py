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
    # The candidate's derivation SIGNALS (_pattern/_grounds_ddl/_domain/_tier) must CROSS this boundary into
    # provenance — grounds_ddl drives the deterministic realize profile (lowering-by-theorem), domain is the
    # cross-domain tag for emergence. Dropping them here severed the ontology↔DDL loop once (0/433 promoted
    # templates retained grounds_ddl → 100% dice-rolled spine); do not re-sever.
    prov = dict(d.get("provenance") or {})
    for key, src in (("pattern", "_pattern"), ("tier", "_tier"),
                     ("grounds_ddl", "_grounds_ddl"), ("domain", "_domain")):
        if d.get(src) and key not in prov:
            prov[key] = d[src]
    return CatalogTemplate(
        template_id=d["template_id"], manchester_template=d["manchester_template"],
        slot_types=d.get("slot_types") or {}, is_complex=bool(d.get("is_complex")),
        verbal_template=d.get("verbal_template", ""), bfo_anchor_path=d.get("bfo_anchor_path") or [],
        provenance=prov)


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
    ap.add_argument("--align-min", type=float, default=0.35,
                    help="LINK-1 gate: min cosine(candidate verbalization, its FinePDFs _source_span) to admit — "
                         "the membrane condition for 'the ontology is informed by the inputs' (0 = off)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cand_path = _CATALOG_DIR / args.candidate
    if not cand_path.exists():
        print(f"no candidate file {cand_path.relative_to(REPO)} — nothing to promote", file=sys.stderr)
        return 1
    cands = json.loads(cand_path.read_text()).get("templates", [])
    print(f"promoting from {cand_path.name}: {len(cands)} candidates")

    align: dict = {}                              # LINK-1: cos(verbalization, the candidate's FinePDFs source span)
    if args.align_min > 0 and cands:
        try:
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer("all-MiniLM-L6-v2")
            V = enc.encode([(d.get("verbal_template") or "") for d in cands], normalize_embeddings=True, show_progress_bar=False)
            Sx = enc.encode([(d.get("_source_span") or "") for d in cands], normalize_embeddings=True, show_progress_bar=False)
            align = {d["template_id"]: float((V[i] * Sx[i]).sum()) for i, d in enumerate(cands)}
        except Exception as e:  # noqa: BLE001
            print(f"  link-1 gate unavailable ({type(e).__name__}); promoting without it", file=sys.stderr)

    RG = None
    if args.hermit:
        try:
            from aegir.ontology import reasoning_gates as RG
            from aegir.ontology.deeponto_harness import ensure_jvm
            ensure_jvm()
        except Exception as e:  # noqa: BLE001
            print(f"  HermiT unavailable ({type(e).__name__}); promoting on CPU/parse gates only", file=sys.stderr)
            RG = None

    # Stage 1 — per-candidate CPU re-gate (G1 well-formed/parse, G2 complex, clean-room, anchor)
    funnel = {"in": len(cands), "regate": 0, "link1_drop": 0, "reasoned": 0, "promoted": 0}
    regated: list[CatalogTemplate] = []
    for d in cands:
        ct = _to_template(d)
        g = DM.content_membrane(ct, source_span=d.get("_source_span", ""), jvm=args.jvm)
        g1_ok = g["g1_well_formed"] and g["g1_deeponto_parses"] is not False
        if not (g1_ok and g["g2_is_complex_class"] and g["clean_room"] and g["anchor_valid"]):
            print(f"  ✘ re-gate {ct.template_id[:36]:36s} g1={g1_ok} g2={g['g2_is_complex_class']} {g['g1_reason']}")
            continue
        a = align.get(ct.template_id)
        if a is not None and a < args.align_min:                 # LINK-1: verbalization orthogonal to its source
            print(f"  ✘ link-1  {ct.template_id[:36]:36s} align={a:.3f} < {args.align_min} (orthogonal to FinePDFs source)")
            funnel["link1_drop"] += 1
            continue
        funnel["regate"] += 1
        regated.append(ct)

    # Stage 2 — AGGREGATE reasoner gate (Gate 4 consistency + unsat=0, Gate 5a equivalence dedup), ONE HermiT pass
    promoted: list[CatalogTemplate] = regated
    if RG is not None and regated:
        bg = RG.batch_gate(regated)
        if not bg["consistent"]:
            print(f"  ✘ batch globally inconsistent ({bg['reason'].get('error')}) — nothing promoted", file=sys.stderr)
            promoted = []
        else:
            promoted = bg["survivors"]
            for tid, why in bg["dropped"].items():
                print(f"  ✘ reasoner {tid[:36]:36s} {why}")
            print(f"  reasoner: {len(promoted)}/{len(regated)} satisfiable+non-redundant · "
                  f"mean inferred supers {bg['mean_inferred_supers']} · {bg['n_classes']} classes")
        funnel["reasoned"] = len(promoted)
    funnel["promoted"] = len(promoted)
    for t in promoted:
        print(f"  ✓ {t.template_id}")

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
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

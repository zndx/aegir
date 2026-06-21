#!/usr/bin/env python
"""Content-first ontology derivation — the RIGHT-ARROW flow: FinePDFs text → Qwen3.6 DERIVES primitives →
the pattern library constrains structure → the membrane verifies → **the ontology grows from the content**.

The flaw this fixes (RH, 2026-06-20): the prototype 540 templates became the reference frame. The coverage
audit *scored* FinePDFs topics against the fixed templates; ``generate_chapter`` *sampled* from them. So
streaming the bulk of FinePDFs (and other text) expanded the ontology's semantic coverage **not at all** —
the arrow pointed the wrong way (content measured against ontology, never mining ontology from content).

This inverts it. For each unit of real-world text:

  1. DERIVE   — Qwen3.6 READS the passage and derives the domain primitives it implies (entities, relations,
                attributes, value-sets, processes), choosing the best-fit axiom PATTERN per primitive from the
                structural vocabulary (``patterns.py``: FHIR-minimum / OWL2 / ODP / SysMLv2). The LLM supplies
                SEMANTICS; the pattern enforces STRUCTURE.
  2. VERIFY    — the same membrane as ``generate_ontology_primitives`` (structural → clean-room → novelty →
                realize-grounding → utility; ``--verify-jvm`` adds DeepOnto-verbalize + HermiT). Nothing
                enters the catalog unverified.
  3. GROW      — admitted primitives expand the ontology; ``08_derived.candidate.json`` accumulates. The
                forward-cursor (``--skip-docs``) streams forward through FinePDFs so coverage compounds.

The templates demote to a SEED + regression baseline; FinePDFs becomes the driver; the LLM does the lifting.
This is "leverage the LLM to its fullest" applied to the most upstream, highest-leverage stage (see memory
ontology_realization + aegir-convergence-loop). Clean-room unchanged: FHIR CC0 structure only, no proprietary
terminology content; every IRI is sdg:-coined original expression. Full thinking traces retained.

    just engine-serve   # (Qwen3.6 up)
    uv run --no-sync python scripts/derive_ontology.py --n-docs 12 --max-primitives 4         # derive + gate
    uv run --no-sync python scripts/derive_ontology.py --n-docs 200 --skip-docs 0 \
        --write-candidates --verify-jvm --advance-cursor                                       # streaming growth
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.ontology import patterns  # noqa: E402
from aegir.ontology import derivation_membrane as DM  # noqa: E402 — the three-gate content membrane
import generate_ontology_primitives as G  # noqa: E402 — reuse G.assemble (pattern → CatalogTemplate)

_TRACES = REPO / "build" / "ontology_derivation_traces.jsonl"
_CANDIDATES = REPO / "src" / "aegir" / "ontology" / "catalog" / "08_derived.candidate.json"
_CURSOR = REPO / "build" / "derive_ontology_cursor.json"
_ARR_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.S)
# Local FinePDFs corpus candidates (\x03-delimited docs). Override with --corpus.
_CORPUS_DEFAULTS = [
    "/raid/datasets/aegir-corpus-v1/finepdfs-lab/finepdfs_lab_train.txt",
    "/raid/datasets/aegir-corpus-v1/finepdfs-lab/finepdfs_lab.txt",
]

_SYSTEM = (
    "You are an ontology engineer growing a BFO 2020 / CCO-anchored ontology FROM real-world text. You will "
    "READ a passage and DERIVE the domain ontology primitives it implies: the salient entities, relationships, "
    "attributes, value-sets, and processes a rigorous ontology should capture to represent THIS domain. "
    "Express EACH derived primitive using exactly ONE of the provided axiom PATTERNS (the structural "
    "vocabulary) — choose the best-fit pattern per primitive. Hard rules: (1) keep each pattern's class slots "
    "{X:Class}/{Y:Class} as slots; fill ONLY property/data slots and $DESIGN params, keying property_fillers "
    "by the EXACT slot name shown before the colon (e.g. for {p:ObjectProperty} use key \"p\", value "
    "\"sdg:hasAnalyte\"; never key by a class slot, never invent keys). Coin camelCase sdg: IRIs. (2) anchor "
    "each to a real BFO/CCO class. (3) invent NO proprietary terminology codes (no SNOMED/LOINC, no numeric "
    "code systems). (4) FAITHFULLY and FULLY represent the domain — use as many distinct structural patterns "
    "as it genuinely requires (rich domains need disjoint partitions, cardinality, value-partitions, reified "
    "relations, composites). Repeating a pattern across primitives is FINE; do NOT avoid one because you used "
    "it. Each primitive must be a genuinely COMPLEX asserted class (multiple constraints / disjointness / "
    "cardinality), never a bare 'X is a Y'. (5) ground each primitive in a real quote (source_span); skip the "
    "passage if it implies no complex domain classes. OUTPUT CONTRACT: emit exactly one fenced ```json "
    "block holding {\"primitives\": [ ... ]}, each primitive an object with keys: pattern (a pattern name from "
    "the catalog), template_id (snake_case), property_fillers (slot-name -> sdg:IRI), new_properties (list of "
    "{iri,label,domain,range}), bfo_anchor_path (list of prefixed BFO/CCO IRIs, leaf last), verbal_template "
    "(one sentence using the {slots}), source_span (a short quote from the passage that motivated it), "
    "rationale. Put all reasoning OUTSIDE the json block."
)


def _corpus_path(arg: str | None) -> Path | None:
    for cand in ([arg] if arg else []) + _CORPUS_DEFAULTS:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def sample_content(corpus: Path, n: int, skip: int, min_chars: int, max_chars: int):
    """Forward-cursor document sampler over the \\x03-delimited local corpus. Returns (docs, scanned)."""
    docs: list[str] = []
    skipped = scanned = 0
    buf = ""
    with corpus.open(encoding="utf-8", errors="ignore") as fh:
        while len(docs) < n:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            parts = buf.split("\x03")
            buf = parts.pop()
            for d in parts:
                d = d.strip()
                if len(d) < min_chars:
                    continue
                scanned += 1
                if skipped < skip:
                    skipped += 1
                    continue
                docs.append(d[:max_chars])
                if len(docs) >= n:
                    break
    return docs, skip + scanned


def _fallback_passages() -> list[str]:
    """Three original domain passages (clean-room; NOT FinePDFs) so the deriver is testable with no corpus."""
    return [
        ("In a regional water authority, each monitoring station records turbidity, dissolved-oxygen, and pH "
         "samples on a fixed schedule. A sample is collected by a field technician, transported under a "
         "chain-of-custody record, and analysed in an accredited laboratory. Each analysis yields a measured "
         "value with a unit and an uncertainty, and is compared against a regulatory threshold to flag "
         "exceedances. A station belongs to a catchment, and catchments roll up into a river basin."),
        ("A precision-machining shop tracks work orders through routing operations on numerically-controlled "
         "machines. Each operation consumes tooling, produces an inspected feature with a measured dimension "
         "and a tolerance, and is performed at a workcenter. A part has a bill of materials referencing "
         "component parts; a lot of finished parts carries a heat-treatment certificate and a material "
         "traceability record back to the supplier."),
        ("In a clinical biobank, a participant consents to a study under a protocol. A specimen is derived from "
         "a collection event, partitioned into aliquots, and stored at a freezer location at a controlled "
         "temperature. An aliquot may be assigned to an assay that produces a result interpreted against a "
         "reference range; deviations are recorded as protocol exceptions linked to the responsible site."),
    ]


def _apply_domain_filter(docs: list[str], args) -> "tuple[list[str], list[dict | None], dict]":
    """Aim the aperture: classify each doc against the SKOS domain index (ColBERT/Qdrant MaxSim) and keep
    only those whose top concept falls in the requested subtree above the belief floor. Grounds the input
    filter in the ontology — to widen coverage you widen the SKOS hierarchy first."""
    from aegir.ontology import domain_index as DI
    codes = DI.subtree_codes(DI.load_skos(), args.domain)
    if not codes:
        print(f"  domain '{args.domain}' not found in the SKOS hierarchy — add it first (expand-hierarchy-first)", file=sys.stderr)
        return [], [], {"scanned": len(docs), "kept": 0, "subtree": 0}
    kept, tags = [], []
    for d in docs:
        h = DI.classify_hierarchical(d, top_k=5, url=args.domain_url, collection=args.domain_collection)
        top = h.get("top") or {}
        # gate on the relative margin (confidence), not just top-1-in-subtree — the rich domain concepts
        # are strong attractors so off-domain docs still land in-subtree but at a much lower rel_margin.
        if DI.in_subtree(top, codes) and h["rel_margin"] >= args.domain_tau:
            kept.append(d)
            tags.append({"code": top.get("code"), "label": top.get("pref_label"),
                         "rel_margin": h["rel_margin"], "belief": h["belief"]})
    return kept, tags, {"scanned": len(docs), "kept": len(kept), "subtree": len(codes)}


def _load_harvest(store_path: str, n: int, max_chars: int) -> "tuple[list[str], list[dict | None], str]":
    """Load pre-matched in-domain docs from the content-addressed harvest cache (``harvest_domain_docs``).
    Already domain-filtered + deduped — derive directly, tagging each with its harvested SKOS domain."""
    store = Path(store_path)
    manifest: dict[str, dict] = {}
    mf = store / "manifest.jsonl"
    if mf.exists():
        for line in mf.read_text().splitlines():
            try:
                r = json.loads(line)
                manifest[r["hash"]] = r
            except (ValueError, KeyError):
                continue
    docs, tags = [], []
    for f in sorted((store / "docs").glob("*.txt"))[:n]:
        docs.append(f.read_text(encoding="utf-8", errors="ignore")[:max_chars])
        m = manifest.get(f.stem, {})
        tags.append({"code": m.get("code"), "label": m.get("label"),
                     "rel_margin": m.get("rel_margin"), "hash": f.stem})
    return docs, tags, f"{store.name} ({len(docs)} docs)"


def _pattern_catalog() -> tuple[str, dict]:
    """Compact catalog text (class_template patterns only) the LLM picks from, + the name→pattern map."""
    pats = [p for p in patterns.library().values() if p.pattern_kind == "class_template"]
    pats.sort(key=lambda p: (("fhir_minimum", "owl2_core", "odp", "sysmlv2").index(p.tier), p.name))
    lines = []
    for p in pats:
        lines.append(f"- {p.name} [{p.tier}, grounds {p.grounds_ddl}]: {p.manchester_skeleton}")
    return "\n".join(lines), {p.name: p for p in pats}


def derive(passage: str, catalog_text: str, *, capability: str,
           temperature: float, max_primitives: int) -> dict:
    """Ask Qwen3.6 to derive pattern-conforming primitives FROM the passage. Returns proposals + trace.
    No anti-repetition instruction: structural diversity emerges from faithful representation (gates 2/3
    enforce richness, not the prompt)."""
    from aegir.engine.client import complete_detailed
    prompt = (
        f"AXIOM PATTERN CATALOG (the structural vocabulary — choose the best-fit pattern per primitive by its "
        f"exact name; reuse patterns as often as the domain needs):\n{catalog_text}\n\n"
        f"PASSAGE (derive the COMPLEX domain classes it implies — entities, relations, attributes, value-sets, "
        f"processes):\n\"\"\"\n{passage}\n\"\"\"\n\n"
        f"Derive up to {max_primitives} high-value COMPLEX primitives (skip the passage if it implies none) "
        f"as one ```json block {{\"primitives\": [...]}}.")
    out = complete_detailed(prompt, capability=capability, system_prompt=_SYSTEM,
                            max_tokens=8192, temperature=temperature)
    m = _ARR_FENCE.search(out["text"])
    primitives: list[dict] = []
    if m:
        try:
            blob = json.loads(m.group(1))
            primitives = blob.get("primitives", []) if isinstance(blob, dict) else (blob if isinstance(blob, list) else [])
        except ValueError:
            primitives = []
    return {"primitives": primitives, "raw": out["text"], "reasoning": out.get("reasoning_content", "")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None, help="path to \\x03-delimited local corpus (FinePDFs); default auto")
    ap.add_argument("--n-docs", type=int, default=12, help="documents (content units) to derive from")
    ap.add_argument("--skip-docs", type=int, default=None, help="forward cursor; default reads build/derive_ontology_cursor.json")
    ap.add_argument("--advance-cursor", action="store_true", help="persist the forward cursor after the run")
    ap.add_argument("--max-primitives", type=int, default=4, help="max primitives to derive per document")
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=4649)
    ap.add_argument("--write-candidates", action="store_true", help="write admitted primitives → 08_derived.candidate.json")
    ap.add_argument("--no-jvm", dest="jvm", action="store_false", help="skip the DeepOnto JVM gates (G1-parse/G3); G2/clean/anchor/faithful still run")
    ap.add_argument("--k-verbal", type=int, default=3, help="G3 floor: min distinct procedural verbalization skeletons")
    # ── semantic-domain aperture (ColBERT/Qdrant over the SKOS hierarchy) ──
    ap.add_argument("--domain", default=None, help="SKOS subtree (notation code or prefLabel) to AIM the aperture at, e.g. 'Process' (LIMS once added)")
    ap.add_argument("--domain-tau", type=float, default=0.10, help="min rank1−rank2 relative margin for a doc to pass the domain gate (≈0.10 separates on-domain from off-domain)")
    ap.add_argument("--domain-oversample", type=int, default=5, help="sample this many × n-docs, then filter to in-domain (so the gate doesn't starve n)")
    ap.add_argument("--domain-url", default="http://localhost:6355")
    ap.add_argument("--domain-collection", default="sdg_domains")
    ap.add_argument("--from-harvest", nargs="?", const=str(REPO / "build" / "domain_harvest"), default=None,
                    help="derive from the content-addressed harvest cache (pre-matched in-domain docs) instead of streaming; optional path")
    args = ap.parse_args()

    corpus = None
    new_cursor = 0
    if args.from_harvest:
        # consume the harvest cache: docs are already domain-matched + deduped (no re-filtering)
        docs, domain_tags, src = _load_harvest(args.from_harvest, args.n_docs, args.max_chars)
        if not docs:
            print(f"no harvested docs in {args.from_harvest}", file=sys.stderr)
            return 1
        print(f"DERIVE FROM HARVEST: {len(docs)} pre-matched docs from {src}")
    else:
        corpus = _corpus_path(args.corpus)
        skip = args.skip_docs
        if skip is None:
            skip = json.loads(_CURSOR.read_text()).get("cursor", 0) if _CURSOR.exists() else 0
        n_sample = args.n_docs * args.domain_oversample if args.domain else args.n_docs
        if corpus:
            docs, new_cursor = sample_content(corpus, n_sample, skip, args.min_chars, args.max_chars)
            src = f"{corpus.name} (skip={skip})"
        else:
            docs, new_cursor = _fallback_passages()[: n_sample], skip
            src = "fallback passages (no local corpus found — clean-room samples)"
        if not docs:
            print(f"no documents sampled from {src}", file=sys.stderr)
            return 1
        # semantic-domain aperture: keep only docs whose top SKOS concept is in the requested subtree
        domain_tags = [None] * len(docs)
        if args.domain:
            docs, domain_tags, fstats = _apply_domain_filter(docs, args)
            print(f"DOMAIN APERTURE '{args.domain}': {fstats['kept']}/{fstats['scanned']} docs in-subtree "
                  f"({fstats['subtree']} SKOS concepts, tau={args.domain_tau})")
            docs, domain_tags = docs[: args.n_docs], domain_tags[: args.n_docs]
            if not docs:
                print("  no in-domain docs in this window — widen the window or the SKOS subtree", file=sys.stderr)
                return 1

    if args.jvm:
        try:
            from aegir.ontology.deeponto_harness import ensure_jvm
            ensure_jvm()
        except Exception as e:  # noqa: BLE001
            print(f"  JVM unavailable ({type(e).__name__}); G1-parse/G3 deferred. Run with LD_LIBRARY_PATH=build/jvm-libs.", file=sys.stderr)
            args.jvm = False

    catalog_text, name_map = _pattern_catalog()
    print(f"source: {src} · docs={len(docs)} · patterns(class_template)={len(name_map)} · "
          f"membrane=G1(parse{'+jvm' if args.jvm else ''})·G2(complex)·G3(verbal≥{args.k_verbal})·clean·anchor·faithful  [NO novelty gate]")

    _TRACES.parent.mkdir(parents=True, exist_ok=True)
    trace_fh = _TRACES.open("a")
    admitted: list[dict] = []
    funnel = {"docs": 0, "derived": 0, "valid_pattern": 0, "assembled": 0, "g1_parse": 0, "g2_complex": 0,
              "g3_verbal": 0, "clean_room": 0, "anchor": 0, "faithful": 0, "admit": 0}
    seen_anchors: set[str] = set()
    used_groundings: set[str] = set()
    used_tiers: set[str] = set()
    for di, passage in enumerate(docs):
        funnel["docs"] += 1
        try:
            r = derive(passage, catalog_text, capability=args.capability,
                       temperature=args.temperature, max_primitives=args.max_primitives)
        except Exception as e:  # noqa: BLE001 — engine down / RPC: report, continue
            print(f"  [doc {di}] engine error: {e}", file=sys.stderr)
            continue
        for prim in r["primitives"]:
            funnel["derived"] += 1
            pat = name_map.get(prim.get("pattern", ""))
            if not pat:
                continue
            funnel["valid_pattern"] += 1
            tmpl = G.assemble(pat, prim)
            if not tmpl:
                continue
            funnel["assembled"] += 1
            g = DM.content_membrane(tmpl, source_span=prim.get("source_span", ""),
                                    jvm=args.jvm, k_verbal=args.k_verbal)
            funnel["g1_parse"] += int(bool(g["g1_well_formed"] and g["g1_deeponto_parses"] is not False))
            funnel["g2_complex"] += int(bool(g["g2_is_complex_class"]))
            funnel["g3_verbal"] += int(bool(g["g3_verbal_diversity_ok"]))
            funnel["clean_room"] += int(bool(g["clean_room"]))
            funnel["anchor"] += int(bool(g["anchor_valid"]))
            funnel["faithful"] += int(bool(g["faithful"]))
            trace_fh.write(json.dumps({
                "doc": di, "source": src, "domain": domain_tags[di], "pattern": pat.name, "tier": pat.tier,
                "template_id": tmpl.template_id, "manchester": tmpl.manchester_template,
                "source_span": prim.get("source_span", ""),
                "gates": {k: v for k, v in g.items() if not k.startswith("_")},
                "proposal": prim, "reasoning": r["reasoning"][:4000]}) + "\n")
            if g["admit"]:
                funnel["admit"] += 1
                used_groundings.add(pat.grounds_ddl)
                used_tiers.add(pat.tier)
                for a in (tmpl.bfo_anchor_path or []):
                    seen_anchors.add(a)
                admitted.append({"template_id": tmpl.template_id, "manchester_template": tmpl.manchester_template,
                                 "slot_types": tmpl.slot_types, "is_complex": tmpl.is_complex,
                                 "verbal_template": tmpl.verbal_template, "bfo_anchor_path": tmpl.bfo_anchor_path,
                                 "_pattern": pat.name, "_tier": pat.tier, "_grounds_ddl": pat.grounds_ddl,
                                 "_source_span": prim.get("source_span", ""), "_complexity": g["g2_complexity"],
                                 "_domain": domain_tags[di],
                                 "_new_properties": prim.get("new_properties") or [], "_utility": g["utility"]})
                print(f"  ✓ [doc {di}|{pat.name}] {tmpl.template_id} (utility {g['utility']}, "
                      f"cx {g['g2_complexity']['score']}, {g['g3_verbal'].get('n_distinct', '-')} verbal)")
    trace_fh.close()

    # G2 at the output level + canonical merge (genuinely identical concepts only — NOT structural repeats)
    bg = DM.batch_gate(admitted, min_complex=2)
    admitted = bg["merged"]

    print(f"\nFUNNEL: {funnel}")
    yield_rate = round(funnel["admit"] / max(1, funnel["docs"]), 2)
    print(f"DERIVATION YIELD: {funnel['admit']} admitted from {funnel['docs']} docs = {yield_rate}/doc "
          f"(after canonical merge: {bg['n_merged']}; {bg['n_dup_dropped']} dup concepts merged)")
    print(f"G2 BATCH: {bg['n_complex_classes']} complex asserted classes "
          f"({'PASS' if bg['multiple_complex_ok'] else 'FAIL'} — needs ≥2) · tiers {bg['tiers']} · groundings {bg['groundings']}")
    print(f"ONTOLOGY GROWTH: +{bg['n_merged']} primitives · {len(seen_anchors)} distinct BFO/CCO anchors touched")
    print("  → richness gates (parse·complex·diverse-verbalization) admit semantic coverage without an anti-repetition heuristic.")

    if args.write_candidates and admitted:
        existing_cand = []
        if _CANDIDATES.exists():
            try:
                existing_cand = json.loads(_CANDIDATES.read_text()).get("templates", [])
            except ValueError:
                existing_cand = []
        seen_ids = {t.get("template_id") for t in existing_cand}
        merged = existing_cand + [a for a in admitted if a["template_id"] not in seen_ids]
        _CANDIDATES.write_text(json.dumps({"family": "08_derived", "templates": merged}, indent=1) + "\n")
        print(f"wrote {len(merged)} candidate primitives ({len(merged) - len(existing_cand)} new) → {_CANDIDATES.relative_to(REPO)}")
    else:
        print(f"(dry run — {len(admitted)} admitted; pass --write-candidates)")

    if args.advance_cursor and corpus:
        _CURSOR.write_text(json.dumps({"cursor": new_cursor, "corpus": str(corpus)}, indent=1))
        print(f"cursor advanced → {new_cursor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

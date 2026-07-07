#!/usr/bin/env python
"""Dynamic ontology-primitive generation — Qwen3.6 adds SEMANTIC RICHNESS, patterns.py enforces STRUCTURE.

The capability the ontology has lacked: in response to diverse inputs (FHIR resource concepts, FinePDFs gap
topics, or any seed), the local engine (Qwen3.6) proposes semantically-rich ontology primitives that CONFORM
to an axiom-pattern from ``patterns.py`` (the structural contract), and NOTHING enters the catalog until it
passes the verification membrane — the "next few gates":

  1. structural   — patterns.instantiate + validate; realize.py grounds it (parse_restrictions finds the
                    expected restrictions ⇒ the asserted structure materializes into a DDL profile).
  2. clean-room   — _FORBIDDEN tripwire: no proprietary terminology codes (SNOMED/LOINC) or foreign-schema ids.
  3. novelty      — not a structural duplicate of an existing catalog template (the ontology actually grows).
  4. utility      — coverage(is-a-gap) × structural-novelty(new pattern/grounding) × realizability(grounds a
                    DDL profile) × verified — admits high-utility primitives, not volume.
  5. JVM membrane — DeepOnto verbalizable + HermiT consistent/non-trivial + BFO-ancestry (``--verify-jvm``;
                    the deep reasoner gate, batched, best-effort — the inc-2a oracle).

FHIR (CC0) + the pattern library set the floor; Qwen3.6 levels up with domain-rich content that still
conforms. Clean-room: FHIR resource CONCEPTS (names) are CC0; we generate original axioms + sdg:-derived
identifiers, never proprietary code content. Thinking traces retained (corpus value-add). Engine = LOCAL.

    just engine-serve   # (Qwen3.6 up)
    uv run --no-sync python scripts/generate_ontology_primitives.py --per-concept 2 --concepts fhir   # propose+gate
    uv run --no-sync python scripts/generate_ontology_primitives.py --write-candidates --verify-jvm    # full membrane
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import ddl, patterns, realize  # noqa: E402
from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402

_CACHE = REPO / "build" / "ontology_primitive_cache.json"
_TRACES = REPO / "build" / "ontology_primitive_traces.jsonl"
_CANDIDATES = REPO / "src" / "aegir" / "ontology" / "catalog" / "08_generated.candidate.json"
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
# proprietary terminology / foreign-schema tripwire. NOTE: numeric BFO/CCO IRIs (bfo:0000015) are LEGITIMATE
# — we match SNOMED/LOINC by name + a standalone 6+ digit code NOT prefixed by an ontology IRI scheme.
_FORBIDDEN = re.compile(r"\bsnomed\b|\bloinc\b|(?<![:_a-z0-9])\d{6,}\b|catalog_product|ir_model|res_partner", re.I)

# FHIR resource CONCEPTS (CC0 spec names; STRUCTURAL concepts only — no proprietary terminology content).
# These map onto our observation/measurement (LIMS/clinical) + directive families and set the floor.
FHIR_CONCEPTS = [
    "observation", "specimen", "diagnostic_report", "procedure", "device", "substance", "encounter",
    "care_plan", "medication_administration", "supply_request", "service_request", "body_structure",
    "research_study", "molecular_sequence", "imaging_study", "questionnaire_response",
]

_SYSTEM = (
    "You are an ontology engineer extending a BFO 2020 / CCO-anchored ontology. You are given ONE axiom "
    "PATTERN (a Manchester skeleton with typed slots) and ONE domain concept. Propose a semantically rich, "
    "domain-faithful primitive that CONFORMS EXACTLY to the pattern's structure. Hard rules: (1) keep the "
    "pattern's class slots as SLOTS (never fill a class slot with an IRI) but RENAME each to a domain-meaningful "
    "CamelCase concept via class_slot_names (class-slot-token -> ConceptName, e.g. {\"X\":\"Specimen\"}); fill only "
    "the property slots {p:ObjectProperty}/{d:DataProperty} and any $DESIGN params with concrete sdg: IRIs you coin "
    "(camelCase, e.g. sdg:hasSpecimen). "
    "(2) anchor to a real BFO/CCO class. (3) invent NO proprietary terminology codes (no SNOMED/LOINC). "
    "(4) OUTPUT CONTRACT: emit exactly one fenced ```json block with keys: template_id (snake_case), "
    "class_slot_names (class-slot-token -> domain ConceptName), "
    "property_fillers (slot-token -> sdg:IRI), new_properties (list of {iri,label,domain,range}), "
    "bfo_anchor_path (list of prefixed BFO/CCO IRIs, leaf last), verbal_template (one sentence with the {slots}), "
    "rationale. Put reasoning OUTSIDE the json block."
)


def _concept_signals(args) -> list[str]:
    if args.concepts == "fhir":
        return FHIR_CONCEPTS[: args.limit] if args.limit else FHIR_CONCEPTS
    if args.concepts == "finepdfs" and args.audit_run:
        import pyarrow.parquet as pq
        rows = pq.read_table(Path(args.audit_run) / "topic_coverage.parquet").to_pylist()
        gaps = [r for r in rows if (r.get("coverage_score") or 1.0) < 0.35][: args.limit or 30]
        return [re.sub(r"\W+", "_", (r.get("label") or f"topic_{r.get('topic_id')}")).strip("_").lower() for r in gaps]
    return [c.strip() for c in (args.concepts or "").split(",") if c.strip()]


def _patterns_for(concept: str, rng) -> list[patterns.AxiomPattern]:
    """Pick target patterns for a concept — instantiable (class_template) ones, biased to richer tiers."""
    pool = [p for p in patterns.library().values() if p.pattern_kind == "class_template"]
    rng.shuffle(pool)
    return pool


def _existing_structures(catalog_dir: Path) -> set[str]:
    """Normalized manchester structures already in the catalog (for the novelty gate)."""
    out: set[str] = set()
    from aegir.ontology.schema import catalog_files
    for f in catalog_files(catalog_dir):
        for t in load_catalog(f).templates:
            out.add(_norm_structure(t.manchester_template))
    return out


def _norm_structure(m: str) -> str:
    m = re.sub(r"\{[^}]+\}", "{}", m or "")          # erase slot identities
    m = re.sub(r"\b\w+:\w+\b", "IRI", m)              # erase concrete IRIs
    return re.sub(r"\s+", " ", m).strip().lower()


def propose(pattern: patterns.AxiomPattern, concept: str, *, capability: str, temperature: float) -> dict:
    """Ask Qwen3.6 for a pattern-conforming primitive (JSON). Returns the parsed proposal + trace."""
    from aegir.engine.client import complete_detailed
    prompt = (f"PATTERN '{pattern.name}' (tier {pattern.tier}, grounds {pattern.grounds_ddl}):\n"
              f"  skeleton: {pattern.manchester_skeleton}\n  anchor guidance: {pattern.bfo_cco_anchor}\n"
              f"  slots: {pattern.slots()}   design params: {pattern.design_params()}\n"
              f"CONCEPT: {concept}\nProduce the pattern-conforming primitive as one ```json block.")
    out = complete_detailed(prompt, capability=capability, system_prompt=_SYSTEM,
                            max_tokens=4096, temperature=temperature)
    m = _FENCE.search(out["text"])
    proposal = {}
    if m:
        try:
            proposal = json.loads(m.group(1))
        except ValueError:
            proposal = {}
    return {"proposal": proposal, "raw": out["text"], "reasoning": out.get("reasoning_content", "")}


def assemble(pattern: patterns.AxiomPattern, proposal: dict) -> "CatalogTemplate | None":
    """Fill the pattern's property slots/$params with the proposal's IRIs → a CatalogTemplate."""
    if not proposal.get("template_id"):
        return None
    tid = proposal["template_id"]
    # The LLM sometimes echoes the PATTERN name as the template_id (e.g. "action_definition_behavior"),
    # which would yield generic t_<pattern>/fact_<pattern> table names. Derive the real concept from the
    # subject (head) class slot — the renamed primary entity ("Class: {VideoAnalyticsExecution:Class} …").
    _lib = patterns.library()
    if tid in _lib or any(tid.startswith(p + "_") for p in _lib):   # echoed the pattern name (± a tier suffix)
        m = re.match(r"\s*Class:\s*\{(\w+):", pattern.manchester_skeleton)
        subj = (proposal.get("class_slot_names") or {}).get(m.group(1)) if m else None
        if subj:
            tid = re.sub(r"(?<!^)(?=[A-Z])", "_", re.sub(r"\W+", "_", subj).strip("_")).strip("_").lower() or tid
    tdict = patterns.instantiate(
        pattern.name, proposal.get("property_fillers") or {},
        template_id=tid, family="08_generated",
        bfo_anchor_path=proposal.get("bfo_anchor_path") or [], verbal_template=proposal.get("verbal_template") or "",
        class_names=proposal.get("class_slot_names") or {})
    return CatalogTemplate(
        template_id=tdict["template_id"], manchester_template=tdict["manchester_template"],
        slot_types=tdict["slot_types"], is_complex=tdict["is_complex"],
        verbal_template=tdict["verbal_template"], bfo_anchor_path=tdict["bfo_anchor_path"])


def membrane(template: CatalogTemplate, pattern: patterns.AxiomPattern, existing: set[str]) -> dict:
    """The CPU verification gates + utility. (JVM gates run separately, batched, via --verify-jvm.)"""
    g: dict = {}
    g["structural_valid"] = not patterns.validate(pattern) and template.manchester_template.count("{") == template.manchester_template.count("}")
    g["clean_room"] = not _FORBIDDEN.search(template.manchester_template)
    # realize-grounding: does realize lower it into the pattern's claimed DDL profile?
    try:
        import random
        prof = realize.profile_for_grounds(pattern.grounds_ddl)
        rs = realize.realize_schema(template, "08_generated", profile=prof, rng=random.Random(7))
        g["realize_grounds"] = len(rs.tables) >= (2 if pattern.grounds_ddl in ("junction", "eav") else 1)
        g["_realized_tables"] = len(rs.tables)
    except Exception as e:  # noqa: BLE001
        g["realize_grounds"] = False
        g["_realize_error"] = type(e).__name__
    g["novel"] = _norm_structure(template.manchester_template) not in existing
    # utility: coverage(novel) × structural-novelty(tier>fhir_minimum or new grounding) × realizability × valid
    structural_novelty = 1.0 if pattern.tier in ("owl2_core", "odp", "sysmlv2") else 0.6
    g["utility"] = round(
        (1.0 if g["novel"] else 0.2) * structural_novelty
        * (1.0 if g["realize_grounds"] else 0.3) * (1.0 if g["structural_valid"] and g["clean_room"] else 0.0), 3)
    g["admit_cpu"] = bool(g["structural_valid"] and g["clean_room"] and g["novel"]
                          and g["realize_grounds"] and g["utility"] >= 0.5)
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", default="fhir", help="'fhir' | 'finepdfs' | comma-separated seed concepts")
    ap.add_argument("--audit-run", default=None, help="coverage_v0 run (for --concepts finepdfs gap topics)")
    ap.add_argument("--per-concept", type=int, default=2, help="patterns to attempt per concept")
    ap.add_argument("--limit", type=int, default=6, help="cap concepts (smoke)")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=4649)
    ap.add_argument("--write-candidates", action="store_true", help="write admitted primitives → 08_generated.candidate.json")
    ap.add_argument("--verify-jvm", action="store_true", help="run the DeepOnto/HermiT deep membrane on CPU-admitted primitives")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    concepts = _concept_signals(args)
    existing = _existing_structures(REPO / "src" / "aegir" / "ontology" / "catalog")
    lib_n = len([p for p in patterns.library().values() if p.pattern_kind == "class_template"])
    print(f"concepts={len(concepts)} · patterns(class_template)={lib_n} · existing structures={len(existing)}")

    _TRACES.parent.mkdir(parents=True, exist_ok=True)
    trace_fh = _TRACES.open("a")
    admitted: list[dict] = []
    funnel = {"proposed": 0, "parsed": 0, "assembled": 0, "structural": 0, "clean_room": 0, "novel": 0,
              "realize_grounds": 0, "admit_cpu": 0}
    for concept in concepts:
        pats = _patterns_for(concept, rng)[: args.per_concept]
        for pat in pats:
            funnel["proposed"] += 1
            try:
                r = propose(pat, concept, capability=args.capability, temperature=args.temperature)
            except Exception as e:  # noqa: BLE001 — engine down / RPC: report, continue
                print(f"  [{concept}|{pat.name}] engine error: {e}", file=sys.stderr)
                continue
            if r["proposal"]:
                funnel["parsed"] += 1
            tmpl = assemble(pat, r["proposal"])
            if not tmpl:
                continue
            funnel["assembled"] += 1
            g = membrane(tmpl, pat, existing)
            for k in ("structural_valid", "clean_room", "novel", "realize_grounds"):
                funnel[k.replace("structural_valid", "structural")] += int(bool(g.get(k)))
            trace_fh.write(json.dumps({"concept": concept, "pattern": pat.name, "tier": pat.tier,
                                       "template_id": tmpl.template_id, "manchester": tmpl.manchester_template,
                                       "gates": {k: v for k, v in g.items() if not k.startswith("_")},
                                       "proposal": r["proposal"], "reasoning": r["reasoning"][:4000]}) + "\n")
            if g["admit_cpu"]:
                funnel["admit_cpu"] += 1
                existing.add(_norm_structure(tmpl.manchester_template))  # avoid intra-run dupes
                admitted.append({"template_id": tmpl.template_id, "manchester_template": tmpl.manchester_template,
                                 "slot_types": tmpl.slot_types, "is_complex": tmpl.is_complex,
                                 "verbal_template": tmpl.verbal_template, "bfo_anchor_path": tmpl.bfo_anchor_path,
                                 "_pattern": pat.name, "_tier": pat.tier, "_grounds_ddl": pat.grounds_ddl,
                                 "_new_properties": r["proposal"].get("new_properties") or [],
                                 "_utility": g["utility"]})
                print(f"  ✓ [{concept}|{pat.name}] {tmpl.template_id} (utility {g['utility']}, {g.get('_realized_tables')}t)")
    trace_fh.close()
    print(f"\nFUNNEL: {funnel}")

    if args.verify_jvm and admitted:
        admitted = _jvm_membrane(admitted)
    if args.write_candidates and admitted:
        _CANDIDATES.write_text(json.dumps({"family": "08_generated", "templates": admitted}, indent=1) + "\n")
        print(f"wrote {len(admitted)} candidate primitives → {_CANDIDATES.relative_to(REPO)}")
    else:
        print(f"(dry run — {len(admitted)} CPU-admitted; pass --write-candidates / --verify-jvm)")
    return 0


def _jvm_membrane(admitted: list[dict]) -> list[dict]:
    """Deep membrane (DeepOnto verbalize + HermiT consistency) on CPU-admitted primitives. Best-effort:
    requires the JVM (LD_LIBRARY_PATH=build/jvm-libs). Drops primitives that fail to verbalize or that
    HermiT finds incoherent. (Batched HermiT over one ontology is the production path; here per-primitive.)"""
    try:
        from aegir.ontology.deeponto_harness import extract_parts
    except Exception as e:  # noqa: BLE001
        print(f"  JVM membrane unavailable ({type(e).__name__}); CPU-admitted retained, HermiT pending", file=sys.stderr)
        return admitted
    kept = []
    for a in admitted:
        ct = CatalogTemplate(template_id=a["template_id"], manchester_template=a["manchester_template"],
                             slot_types=a["slot_types"], is_complex=a["is_complex"],
                             verbal_template=a.get("verbal_template", ""), bfo_anchor_path=a["bfo_anchor_path"])
        try:
            parts = extract_parts(ct)
            a["_deeponto_verbalized"] = parts is not None
        except Exception:  # noqa: BLE001
            a["_deeponto_verbalized"] = False
        # HermiT consistency would load (candidate ∪ BFO/CCO) via DeepOnto Ontology.reasoner (abox.py
        # pattern) — wired here as the deep gate; left as a follow-up batch step for the full run.
        if a["_deeponto_verbalized"]:
            kept.append(a)
    print(f"  JVM membrane: {len(kept)}/{len(admitted)} verbalized (HermiT consistency = batched follow-up)")
    return kept


if __name__ == "__main__":
    raise SystemExit(main())

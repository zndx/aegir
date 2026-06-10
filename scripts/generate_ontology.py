#!/usr/bin/env python
"""FinePDFs→ontology generator (Phase 2) — LLM-generate-and-verify.

Grows the ontology from filtered FinePDFs: targets coverage GAPS (topics the seed catalog does
not cover), conditions an LLM on seed exemplars + the target topic's real FinePDFs text, and asks
it to author NEW OWL Manchester axiom templates in the slot DSL. Each candidate runs a GATE chain
(the "crystallization conditions") — structural slot-DSL consistency, BFO-anchoring, and
schema-realism (does it lower to a non-trivial DDL table?). DeepOnto verbalize + topic-alignment
are the deeper gates (added next). Admitted templates land in a growing `.candidate` catalog for
editorial review (per the charter — no auto-merge to the published vocab).

The catalog is a seed crystal: it imparts shape + BFO/CCO grounding; generation grows from it,
conditioned on today's FinePDFs (and a different FinePDFs tomorrow).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402
from aegir.ontology.ddl import render_ddl, template_to_table  # noqa: E402

ANCHORS = {"bfo:Process", "bfo:IndependentContinuant", "cco:Artifact",
           "cco:InformationContentEntity", "cco:DescriptiveICE", "cco:DirectiveICE",
           "cco:DesignativeICE"}
SLOT_RE = re.compile(r"\{(?P<name>\w+):(?P<type>Class|ObjectProperty|DataProperty|Individual)(?::[^}]+)?\}")
JSON_RE = re.compile(r"```json\s*(.*?)```", re.S)

SLOT_DSL = (
    "Slot DSL: write each axiom as OWL Manchester syntax with typed slots {name:Type} where Type "
    "is Class | ObjectProperty | DataProperty | Individual. Use BFO/CCO/sdg IRIs (bfo:Process, "
    "cco:Artifact, cco:DescriptiveICE, sdg:producesMeasurement, …) for fixed terms and {slots} for "
    "the variable concepts. e.g. 'Class: {X:Class} SubClassOf: cco:Artifact, {p:ObjectProperty} some {Y:Class}'."
)


def get_lm(model: str, max_tokens: int, temperature: float):
    import dspy
    provider = model.split("/", 1)[0]
    key = {"cerebras": "CEREBRAS_API_KEY", "xai": "XAI_API_KEY"}.get(provider, "")
    return dspy.LM(model=model, api_key=os.environ.get(key), max_tokens=max_tokens,
                   temperature=temperature, cache=False)


def load_gap_topics(coverage_run: str, limit: int) -> list[dict]:
    import pyarrow.parquet as pq
    rows = pq.read_table(Path(coverage_run) / "topic_coverage.parquet").to_pylist()
    gaps = [r for r in rows if r.get("status") in ("gap", "borderline")]
    gaps.sort(key=lambda r: r.get("coverage_score", 0.0))  # worst-covered first
    return gaps[:limit]


def build_prompt(topic: dict, exemplars: list[CatalogTemplate]) -> str:
    ex_txt = "\n".join(
        f"- template_id: {t.template_id}\n  manchester: {t.manchester_template}\n"
        f"  slot_types: {json.dumps(t.slot_types)}\n  bfo_anchor_path: {json.dumps(t.bfo_anchor_path)}"
        for t in exemplars)
    src = (topic.get("topic_repr_text") or "")[:2500]
    return (
        "You are an ontology engineer extending a BFO 2020 / CCO grounded ontology.\n\n"
        f"{SLOT_DSL}\n\n"
        f"Existing templates in this family (imitate their shape + grounding):\n{ex_txt}\n\n"
        "Author 3-5 NEW Manchester axiom templates capturing the domain concepts in the SOURCE TEXT, "
        "each grounded in BFO/CCO (the subject SubClassOf-chains to a BFO/CCO anchor). Prefer templates "
        "with typed DataProperty slots (values/dates/counts) AND ObjectProperty relations.\n\n"
        f'SOURCE TEXT (real document from the target topic):\n"""{src}"""\n\n'
        "Output ONLY a fenced ```json block: a JSON list of objects "
        '{"template_id": str, "manchester_template": str, "slot_types": {slot: type}, "bfo_anchor_path": [iri]}. '
        f"template_id: lowercase_snake, unique, descriptive. bfo_anchor_path must end at one of {sorted(ANCHORS)}."
    )


def parse_templates(text: str) -> list[CatalogTemplate]:
    m = JSON_RE.search(text or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except Exception:
        return []
    out = []
    for d in (data if isinstance(data, list) else []):
        try:
            out.append(CatalogTemplate(
                template_id=str(d["template_id"]),
                manchester_template=str(d["manchester_template"]),
                slot_types={str(k): str(v) for k, v in d["slot_types"].items()},
                bfo_anchor_path=[str(x) for x in d.get("bfo_anchor_path", [])]))
        except Exception:
            pass
    return out


def gate_structural(t: CatalogTemplate) -> tuple[bool, str]:
    used = {(m.group("name"), m.group("type")) for m in SLOT_RE.finditer(t.manchester_template)}
    used_names = {n for n, _ in used}
    if used_names != set(t.slot_types):
        return False, f"slots used {sorted(used_names)} != declared {sorted(t.slot_types)}"
    for n, ty in used:
        if t.slot_types.get(n) != ty:
            return False, f"slot {n} type mismatch"
    return True, ""


def gate_bfo(t: CatalogTemplate) -> tuple[bool, str]:
    if not t.bfo_anchor_path:
        return False, "no bfo_anchor_path"
    if t.bfo_anchor_path[-1] not in ANCHORS:
        return False, f"anchor {t.bfo_anchor_path[-1]} not a known BFO/CCO anchor"
    return True, ""


def gate_schema_realism(t: CatalogTemplate, family: str) -> tuple[bool, str]:
    try:
        st = template_to_table(t, family)
        ncols = len(st.table.columns)
        if ncols < 3:
            return False, f"too narrow ({ncols} cols)"
        render_ddl(st, [])
        return True, f"{ncols} cols"
    except Exception as e:
        return False, f"lowering failed: {type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage-run", required=True, help="coverage_v0/<run>/ dir")
    ap.add_argument("--n-topics", type=int, default=5)
    ap.add_argument("--model", default="cerebras/zai-glm-4.7")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--out", default=str(REPO / "src/aegir/ontology/catalog/08_generated.candidate.json"))
    args = ap.parse_args()

    cat_by_family: dict[str, list[CatalogTemplate]] = {}
    for f in sorted((REPO / "src/aegir/ontology/catalog").glob("0[1-7]_*.json")):
        if "candidate" in f.name or "combined" in f.name:
            continue
        cat_by_family[f.stem] = load_catalog(str(f)).templates

    topics = load_gap_topics(args.coverage_run, args.n_topics)
    lm = get_lm(args.model, args.max_tokens, args.temperature)
    admitted: list[CatalogTemplate] = []
    stats = {"topics": 0, "proposed": 0, "admitted": 0, "rejects": {}}
    cum_cost = 0.0

    for topic in topics:
        family = topic.get("top_family") or "07_long_tail"
        prompt = build_prompt(topic, cat_by_family.get(family, [])[:3])
        try:
            r = lm(messages=[{"role": "user", "content": prompt}])
            item = r[0] if isinstance(r, list) and r else r
            txt = (item.get("text") if isinstance(item, dict) else str(item)) or ""
        except Exception as e:
            print(f"topic {topic.get('topic_id')}: gen failed ({type(e).__name__})")
            continue
        stats["topics"] += 1
        cands = parse_templates(txt)
        if not cands:
            print(f"  [topic {topic.get('topic_id')}: resp {len(txt)} chars, "
                  f"json-block={'yes' if JSON_RE.search(txt) else 'no'}]")
        stats["proposed"] += len(cands)
        n_adm = 0
        for t in cands:
            gates = [("structural", gate_structural(t)), ("bfo", gate_bfo(t)),
                     ("schema_realism", gate_schema_realism(t, family))]
            failed = [g for g, (ok, _) in gates if not ok]
            if failed:
                for g in failed:
                    stats["rejects"][g] = stats["rejects"].get(g, 0) + 1
                continue
            t.provenance = {"generated_from_topic": str(topic.get("topic_id")),
                            "family": family, "model": args.model}
            admitted.append(t)
            n_adm += 1
        stats["admitted"] += n_adm
        try:
            cum_cost += float(lm.history[-1].get("cost") or 0.0)
        except Exception:
            pass
        print(f"topic {topic.get('topic_id')} ({family}): proposed {len(cands)}, admitted {n_adm} | ${cum_cost:.3f}")
        if args.budget_usd and cum_cost >= args.budget_usd:
            print(f"budget ${cum_cost:.2f} reached — stopping")
            break

    out = {"version": "0.1.0-generated-candidate",
           "templates": [asdict(t) for t in admitted], "null_stats": {}}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n=== {stats['admitted']}/{stats['proposed']} admitted from {stats['topics']} topics "
          f"(${cum_cost:.2f}) → {args.out} ===")
    print(f"rejects by gate: {stats['rejects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

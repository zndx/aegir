"""ground_primitives — SubClassOf grounding for the residual reasoner-ungrounded classes.

Some referenced concepts are PRIMITIVE natural kinds (Age, Chemical, StateAction) that cannot be
given a necessary-AND-sufficient genus+differentia ≡ — the define loop correctly set keep=true.
But the mandate ([[bfo_cco_grounding_mandate]]) requires a PATH to BFO/CCO, which a SubClassOf
genus provides (INDIRECT grounding — the reasoner still entails X ⊑ BFO/CCO). This pass proposes a
single grounding parent per concept, disposed through the same parse + HermiT/CCO membranes.

Also handles heads whose ≡ genus is a property restriction (no named class): a SubClassOf grounding
parent is ADDED alongside the existing ≡ (grounds it without disturbing the definition).

    just engine-serve
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/ground_primitives.py [--rounds 3]
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

from aegir.engine.client import complete_detailed  # noqa: E402
from aegir.ontology.grounding import load_certificate  # noqa: E402
from aegir.ontology.schema import CatalogTemplate, load_catalog, save_catalog  # noqa: E402
from evolve_rigor import parse_slots, validate_detailed  # noqa: E402
from grounding_anchors import Retriever  # noqa: E402
from build_realized_ontology import consistency_check  # noqa: E402

_RESTR = re.compile(r"(?:sdg|bfo|cco):(\w+)\s+(?:some|only|exactly\s+\d+|min\s+\d+|max\s+\d+)\s+sdg:(\w+)")
_SLOT = re.compile(r"\{(\w+):(\w+)\}")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _decamel(s: str) -> str:
    return _CAMEL.sub(" ", s).replace("_", " ").lower()


_SYS = (
    "You GROUND primitive domain classes of a BFO 2020 / CCO ontology. Each is a natural kind that may "
    "resist a full genus-differentia definition — so give it just a GROUNDING PARENT: the single most-specific "
    "class it IS-A, as ONE Manchester axiom `Class: {Name:Class} SubClassOf: <genus>`. The genus must be a "
    "provided ANCHOR (cco:/fhir:/sdg:) OR a BFO category (bfo:0000040 material entity / bfo:0000015 process / "
    "bfo:0000031 generically dependent continuant / bfo:0000019 quality / bfo:0000023 role / bfo:0000020 "
    "specifically dependent continuant). PREFER a specific cco:/sdg: anchor over a bare bfo: category — a deep "
    "path to BFO is BETTER than a shallow one. The genus must be BROADER than the class (a parent kind), NEVER "
    "the class itself or a synonym.\n"
    "SYNTAX: lowercase prefixes only (cco: bfo: sdg:); use the EXACT cco: IRIs from the anchors (e.g. "
    "cco:ont00000958) — do NOT invent cco:/bfo: names.\n"
    "OUTPUT exactly one ```json block {\"groundings\":[{\"manchester_template\":\"Class: {Name:Class} "
    "SubClassOf: <genus>\",\"verbal\":\"one-sentence gloss\"}]}. Reasoning OUTSIDE the json."
)


def _parse(text: str) -> list:
    m = re.search(r"```json\s*(.+?)```", text, re.S) or re.search(r"\{[\s\S]*\"groundings\"[\s\S]*\}", text)
    if not m:
        return []
    try:
        return json.loads(m.group(1) if m.lastindex else m.group(0)).get("groundings", [])
    except (ValueError, AttributeError):
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4, help="concepts per engine call (small: reasoning-token headroom)")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    cat = load_catalog(str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    cert = load_certificate(REPO / "corpora/ontology/grounding_certificate.json")
    if not cert or not cert.get("ungrounded"):
        print("no grounding certificate / nothing ungrounded")
        return 0
    targets = list(cert["ungrounded"])
    # context: where each target is referenced (head + property)
    contexts: dict[str, list] = {}
    for t in cat.templates:
        man = t.manchester_template or ""
        hm = re.search(r"Class:\s*\{(\w+)", man)
        head = hm.group(1) if hm else "?"
        for prop, cls in _RESTR.findall(man):
            if cls in targets:
                contexts.setdefault(cls, []).append(f"{head} (the target of {prop})")
    print(f"ground_primitives: {len(targets)} residual concepts · SubClassOf grounding · ≤{args.rounds} rounds")
    retriever = Retriever()

    accepted: dict[str, CatalogTemplate] = {}
    feedback: dict[str, str] = {}
    for rnd in range(args.rounds):
        pending = [t for t in targets if t not in accepted]
        if not pending:
            break
        # BATCH small: Qwen3.8-27B spends ~4-9k reasoning tokens PER concept, so a big batch overflows max_tokens
        # and truncates before the content json emits (silently → 0 proposed). 4/call keeps content in budget.
        proposed: dict[str, tuple] = {}
        for bi in range(0, len(pending), args.batch):
            chunk = pending[bi:bi + args.batch]
            lines = []
            for f in chunk:
                ctx = ", ".join(sorted(set(contexts.get(f, ["(a referenced domain concept)"])))[:4])
                ex = {_decamel(f), f.lower()}
                alist = " · ".join(f"{a['curie']} ({a['label']})" for a in retriever.retrieve(_decamel(f), k=8, exclude=ex))
                block = f"- {f}  (referenced by: {ctx})\n  anchors: {alist}"
                if f in feedback:
                    block += f"\n  PRIOR REJECTED — {feedback[f]} → fix and re-emit."
                lines.append(block)
            out = complete_detailed("Ground these primitive classes:\n\n" + "\n".join(lines),
                                    capability=args.capability, system_prompt=_SYS, max_tokens=24000,
                                    temperature=args.temperature)
            for d in _parse(out["text"]):
                man = d.get("manchester_template") or ""
                hm = re.search(r"Class:\s*\{(\w+)", man)
                m = next((f for f in chunk if hm and f.lower() == hm.group(1).lower()), None)
                if m and "SubClassOf" in man:
                    proposed[m] = (man, d.get("verbal") or f"A {m}.")
        prov = {f"prim_{f.lower()}": CatalogTemplate(
            template_id=f"prim_{f.lower()}", manchester_template=man, slot_types=parse_slots(man),
            is_complex=False, verbal_template=verbal, verbal_templates=[], mean_verbal_length=0.0,
            bfo_anchor_path=[], broader=[], provenance={"source": "ground_primitives"})
            for f, (man, verbal) in proposed.items()}
        detailed = validate_detailed([(tid, t.manchester_template) for tid, t in prov.items()], prov) if prov else {}
        n_new = 0
        for f, (man, _v) in proposed.items():
            ok, reason = detailed.get(f"prim_{f.lower()}", (False, "not validated"))
            if ok and f not in accepted:
                accepted[f] = prov[f"prim_{f.lower()}"]
                feedback.pop(f, None)
                n_new += 1
            elif not ok:
                feedback[f] = f"`{man}` → {reason}"
        n_demoted = 0
        if accepted:
            _c, unsat = consistency_check(list(cat.templates) + list(accepted.values()))
            unsat_local = {u.rsplit("#", 1)[-1].rsplit("/", 1)[-1].lower() for u in unsat}
            for f in list(accepted):
                if f.lower() in unsat_local:
                    feedback[f] = f"HermiT: '{f}' UNSATISFIABLE under its genus — re-ground to a compatible parent."
                    del accepted[f]
                    n_demoted += 1
        print(f"  round {rnd + 1}/{args.rounds}: proposed {len(proposed)} · +{n_new} · {n_demoted} demoted · "
              f"{len(accepted)}/{len(targets)} grounded · {len(feedback)} pending")

    cat.templates.extend(accepted.values())
    save_catalog(cat, str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    print(f"\nAPPLIED: {len(accepted)} SubClassOf groundings → catalog.json")
    if len(accepted) < len(targets):
        print(f"  still ungrounded ({len(targets) - len(accepted)}): {sorted(set(targets) - set(accepted))}")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

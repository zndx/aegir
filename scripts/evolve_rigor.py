"""GEPA-style rigor-evolution — the engine co-evolves the ontology toward definitional rigor.

The OntoClean membrane (`aegir.ontology.ontoclean`) FLAGS each derived primitive's rigorous target
(≡-candidate kind / BFO role / disposition); the engine REFLECTS on that natural-language feedback and
AUTHORS the rigorous axiom — deciding sufficiency, emitting `EquivalentTo` for genuine genus-differentia
definitions, BFO roles for anti-rigid relational kinds, refining the differentia, or keeping a primitive when
it genuinely cannot be defined. HermiT disposes (the realize step). This is NOT slot-fill: the model may
refine the axiom, distinguish itself, or say a flag is wrong.

    just engine-serve   # the gRPC engine must be up
    uv run --no-sync python scripts/evolve_rigor.py --batch 12 [--limit N]
    LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py --strict-grounding
    uv run --no-sync python scripts/ontology_oquare.py corpora/ontology/sdg-ontology.owl --certificate corpora/ontology/HERMIT_CERTIFICATE.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.engine.client import complete_detailed  # noqa: E402
from aegir.ontology import ontoclean  # noqa: E402
from aegir.ontology.schema import load_catalog, save_catalog  # noqa: E402

_SYS = (
    "You are an ontology engineer hardening a BFO 2020 / CCO ontology toward DEFINITIONAL RIGOR (the OntoClean "
    "+ IOF discipline). You receive primitives currently asserted as bare `Class: {Head} SubClassOf: <genus>, "
    "<restrictions>`, each with a meta-property flag. For EACH primitive, decide and re-author its Manchester "
    "axiom:\n"
    "• equivalent_class candidate — if the genus PLUS the stated restrictions are NECESSARY AND SUFFICIENT to "
    "define the class (anything that is the genus and satisfies the restrictions IS an instance), re-emit the "
    "SAME axiom but with **`EquivalentTo:`** in place of `SubClassOf:`. If one distinguishing condition is "
    "missing for sufficiency, ADD it (refine the differentia) — you may improve the axiom. If the kind is "
    "genuinely PRIMITIVE (no sufficient definition exists, e.g. a natural-kind whose essence isn't captured by "
    "the stated relations), KEEP `SubClassOf:` and say why.\n"
    "• role — if the concept is anti-rigid AND relational (supplier / operator / reviewer / participant: its "
    "bearer could stop being it and still exist), re-model it as a BFO ROLE: `Class: {Head} SubClassOf: "
    "bfo:0000023, bfo:0000052 some {Bearer:Class}` (inheres-in a bearer) and, where a process is implied, "
    "`bfo:0000054 some {Process:Class}` (realized-in) — NOT a rigid subclass of an Agent/Continuant kind.\n"
    "Hard rules: PRESERVE the `{Name:Class}` slot DSL and the existing properties/anchors (keep the same slot "
    "names); change ONLY the operator (SubClassOf→EquivalentTo) and, if you refine, ADD slots/restrictions — "
    "never delete the domain content. Prefer EquivalentTo wherever the differentia are genuinely sufficient "
    "(the IOF defines ~half its terms this way). OUTPUT exactly one ```json block: "
    "{\"upgrades\":[{\"template_id\":\"…\",\"decision\":\"equivalent|role|keep\",\"manchester_template\":\"…\","
    "\"rationale\":\"one line\"}]}. Put reasoning OUTSIDE the json."
)


def _batch_prompt(items: list[dict]) -> str:
    blocks = [
        f"- template_id: {it['template_id']}\n  flag: {it['flag']} — {it['rationale']}\n"
        f"  current: {it['manchester']}\n  gloss: {it['gloss']}"
        for it in items
    ]
    return "Harden these primitives (decide equivalent / role / keep per the rules):\n\n" + "\n\n".join(blocks)


def to_equivalent(manchester: str) -> "str | None":
    """Deterministic SubClassOf → EquivalentTo rewrite of the ORIGINAL axiom (guaranteed-valid syntax):
    `Class: {H} SubClassOf: A, B some C` → `Class: {H} EquivalentTo: A and B some C`. Top-level conjuncts are
    comma-separated (Manchester restrictions carry no top-level commas), so the split is safe. The engine
    DECIDES sufficiency; this guarantees the emitted syntax parses (the original already did)."""
    m = re.match(r"(Class:\s*\{[^}]+\})\s+SubClassOf:\s*(.+)", manchester.strip(), re.S)
    if not m:
        return None
    conjuncts = [c.strip() for c in m.group(2).split(",") if c.strip()]
    return f"{m.group(1)} EquivalentTo: " + " and ".join(conjuncts) if conjuncts else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/08_derived.json"))
    ap.add_argument("--out", default=str(REPO / "src/aegir/ontology/catalog/08_derived.json"))
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="cap targets processed (0 = all)")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    cat = load_catalog(args.catalog)
    by_id = {t.template_id: t for t in cat.templates}
    verdicts = [ontoclean.classify_template(t) for t in cat.templates]
    targets = [v for v in verdicts if v["suggested"] in ("equivalent_class", "role")]
    if args.limit:
        targets = targets[:args.limit]
    print(f"rigor-evolution: distribution {ontoclean.summarize(verdicts)} · {len(targets)} targets to harden")

    upgrades: dict[str, dict] = {}
    n_batches = (len(targets) + args.batch - 1) // args.batch
    for bi in range(n_batches):
        chunk = targets[bi * args.batch:(bi + 1) * args.batch]
        items = [{
            "template_id": v["template_id"], "flag": v["suggested"], "rationale": v["rationale"],
            "manchester": by_id[v["template_id"]].manchester_template,
            "gloss": by_id[v["template_id"]].verbal_template or "",
        } for v in chunk]
        try:
            out = complete_detailed(_batch_prompt(items), capability=args.capability, system_prompt=_SYS,
                                    max_tokens=8192, temperature=args.temperature)
        except Exception as e:  # noqa: BLE001
            print(f"  batch {bi + 1}/{n_batches}: engine error {type(e).__name__}: {str(e)[:80]}")
            continue
        m = re.search(r"```json\s*(.+?)```", out["text"], re.S)
        if not m:
            print(f"  batch {bi + 1}/{n_batches}: no json block")
            continue
        try:
            ups = json.loads(m.group(1)).get("upgrades", [])
        except ValueError:
            print(f"  batch {bi + 1}/{n_batches}: malformed json")
            continue
        ok = 0
        for u in ups:
            tid, dec, man = u.get("template_id"), u.get("decision"), u.get("manchester_template") or ""
            if tid not in by_id:
                continue
            if dec == "equivalent":  # syntax rewritten deterministically from the original; engine DECIDES sufficiency
                upgrades[tid] = u; ok += 1
            elif dec == "role" and "0000023" in man and "{" in man:
                upgrades[tid] = u; ok += 1
        print(f"  batch {bi + 1}/{n_batches}: {ok}/{len(chunk)} hardened "
              f"(eq {sum(1 for u in ups if u.get('decision') == 'equivalent')} · "
              f"role {sum(1 for u in ups if u.get('decision') == 'role')} · "
              f"keep {sum(1 for u in ups if u.get('decision') == 'keep')})")

    n_eq = n_role = 0
    for tid, u in upgrades.items():
        if u["decision"] == "equivalent":
            new_man = to_equivalent(by_id[tid].manchester_template)
            if new_man:
                by_id[tid].manchester_template = new_man
                n_eq += 1
        elif u["decision"] == "role":
            by_id[tid].manchester_template = u["manchester_template"]
            n_role += 1
    save_catalog(cat, args.out)
    print(f"\nAPPLIED: {n_eq} EquivalentTo definitions + {n_role} BFO-role re-models → {Path(args.out).name}")
    print("DISPOSE next: build_realized_ontology.py --strict-grounding (HermiT) → ontology_metrology / ontology_oquare")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

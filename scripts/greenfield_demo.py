#!/usr/bin/env python
"""Greenfield end-to-end demo (#141): FinePDFs passage → domain ontology → DDL/views →
two-register prose SIDE BY SIDE via parallel local-Qwen ACP harnesses.

  FinePDFs passage
    → derive_harness.propose      (subject-matter entities, not the bibliographic wrapper)
    → entities.to_manchester      (real OWL, HermiT/DeepOnto-feedable)
    → score_ontology_ddl          (the substrate check: what DDL, how close to SchemaPile)
    → entities.to_construct       (RI-true tables for the prose)
    → generate.harness.run_queue  ((natural,local) + (semantic,local); semantic tool-equipped)
    → two prose outputs side by side

Run (engine + :8100 vLLM must be up): uv run --no-sync python scripts/greenfield_demo.py <passage.txt>
"""
from __future__ import annotations

import sys
from pathlib import Path

from aegir.ontology.derive_harness import propose
from aegir.ontology.entities import to_construct, to_manchester
from aegir.generate.harness import generate, logic_table


def main() -> int:
    passage_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        sorted(Path("build/domain_harvest/docs").glob("*.txt"))[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "/tmp/claude-1001/-home-rch-local-src-zndx-aegir/"
        "c1a82f70-af52-44be-8741-a01509b778d2/scratchpad/greenfield_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    passage = passage_path.read_text()[:6000]
    print(f"[1/4] deriving domain entities from {passage_path.name} …")
    entities, meta = propose(passage, temperature=0.3)
    print(f"      {meta['n_entities']} entities: {', '.join(e.name for e in entities)}")

    omn = to_manchester(entities)
    (out_dir / "ontology.omn").write_text(omn)

    print("[2/4] scoring the substrate DDL vs SchemaPile …")
    import subprocess
    subprocess.run(["uv", "run", "--no-sync", "python", "scripts/score_ontology_ddl.py",
                    str(out_dir / "ontology.omn")], check=False)

    style = ""  # no same-domain anchor: the semantic register writes ontology-discourse about the DATA
    construct = to_construct(entities, n_rows=4, style_anchor=style)
    # ground the semantic register in kvasir-VERIFIED schema facts (deterministic; the pipeline
    # owns the tools, the agent can't fake them — reliable form of "the harness leverages kvasir")
    from aegir.generate.harness import kvasir_facts
    construct["kvasir_facts"] = kvasir_facts(omn)
    print(f"      kvasir-verified facts grounding the semantic register:\n        "
          + construct["kvasir_facts"].replace("\n", "\n        "))

    print("[3/4] generating BOTH registers on local Qwen (parallel ACP harnesses) …")
    results = generate({"demo": construct},
                       logic_table(["natural", "semantic"], ["local"]),
                       {"local": 2})

    print("[4/4] two prose outputs SIDE BY SIDE:\n" + "=" * 78)
    for r in sorted(results, key=lambda r: r.register):
        (out_dir / f"prose.{r.register}.md").write_text(r.prose or f"(error: {r.error})")
        tag = f"{r.register.upper()} register  [{r.provider or r.backend}]"
        tools = f"  tool_calls={r.exchange.get('tool_calls')}" if r.exchange else ""
        print(f"\n### {tag}{tools}\n")
        print((r.prose or f"(ERROR: {r.error})")[:1400])
        print("\n" + "-" * 78)
    print(f"\nartifacts → {out_dir}")
    from aegir.utils.clean_exit import clean_exit
    clean_exit(0)
    return 0


if __name__ == "__main__":
    main()

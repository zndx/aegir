"""evolve_rigor_to_green.py — the idempotent def-rigor GATE-REALIZER.

Drives the ontology to OQuaRE-GREEN (the FunctionalAdequacy leg: definitional_completeness + realizable_
machinery) via the proven tools, re-realizing + re-gating each round:

    round = define_intermediate_classes (≡ genus+differentia, CCO/HermiT-gated)
          → evolve_rigor (OntoClean ≡ + free-authored BFO roles)
          → build_realized_ontology --strict-grounding (HermiT-validated OWL)
          → ontology_oquare (the gate)

IDEMPOTENT: if the gate is already green, no-op (fixpoint). FAIL-LOUD: if it cannot converge in --max-rounds,
exit non-zero with the gradient — a degraded pipeline never masquerades as a met milestone. This is the first
gate-realizer stage of the end-to-end pipeline (memory end_to_end_pipeline_vision); ``just evolve-rigor`` wraps it.

  LD_LIBRARY_PATH=$(pwd)/build/jvm-libs:$(pwd)/build/cuda-driver-libs \
    uv run --no-sync python scripts/evolve_rigor_to_green.py [--max-rounds 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CERT = REPO / "corpora" / "ontology" / "HERMIT_CERTIFICATE.md"
PY = ["uv", "run", "--no-sync", "python"]


def _run(cmd: "list[str]", desc: str) -> None:
    print(f"\n→ {desc}", flush=True)
    if subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
        print(f"✘ FAILED (fail-loud): {desc}", file=sys.stderr)
        sys.exit(2)


def gate() -> dict:
    """Run ontology_oquare → the authoritative gate + the FunctionalAdequacy drivers."""
    p = subprocess.run(PY + ["scripts/ontology_oquare.py", "--certificate", str(CERT), "--json"],
                       cwd=str(REPO), capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        print("✘ could not read OQuaRE gate:\n" + (p.stdout[-300:] + p.stderr[-300:]), file=sys.stderr)
        sys.exit(2)
    m = d.get("metrics_raw", {})
    return {"green": bool(d.get("gate_green")), "agg": d.get("aggregate"),
            "fa": d.get("characteristics", {}).get("FunctionalAdequacy"),
            "def_c": m.get("definitional_completeness"), "rm": m.get("realizable_machinery"),
            "classes": m.get("n_domain_classes")}


def _fmt(g: dict) -> str:
    return (f"OQuaRE={g['agg']}  FunctionalAdequacy={g['fa']}  def_complete={g['def_c']}  "
            f"realizable={g['rm']}  classes={g['classes']}  green={g['green']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rounds", type=int, default=3, help="outer define→evolve→realize→gate rounds")
    ap.add_argument("--define-rounds", type=int, default=4, help="inner per-filler feedback rounds")
    a = ap.parse_args()

    g = gate()
    print(f"\n=== evolve-rigor · start · {_fmt(g)} ===")
    if g["green"]:
        print("🟢 already GREEN — no-op (idempotent).")
        return 0

    for rnd in range(1, a.max_rounds + 1):
        print(f"\n════════ ROUND {rnd}/{a.max_rounds} ════════")
        _run(PY + ["scripts/define_intermediate_classes.py", "--rounds", str(a.define_rounds)],
             "define intermediate classes (≡ genus+differentia, CCO/HermiT-gated)")
        _run(PY + ["scripts/evolve_rigor.py"], "evolve rigor (OntoClean ≡ + free-authored BFO roles)")
        _run(PY + ["scripts/build_realized_ontology.py", "--strict-grounding"],
             "re-realize → HermiT-validated OWL")
        g = gate()
        print(f"\n── after round {rnd}: {_fmt(g)}")
        if g["green"]:
            print(f"\n🟢 GREEN — converged in {rnd} round(s).")
            return 0

    print(f"\n🔴 could not converge to green in {a.max_rounds} rounds · {_fmt(g)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

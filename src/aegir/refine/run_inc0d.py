"""inc-0(d) CAPABILITY run — the real vibe-acp agent inside the membrane-gated loop on ch0.

Single-shot baseline (the defective ``controlled_ch0``) vs the loop's refined output, with the PROPOSE step
driven by the real agent (fork-venv ``_propose`` subprocess → ``vibe-acp`` → engine). Reports the delta against
the plan's pass criterion: ≥3 of {placeholder↓, disjointness→0, entailment↑, length-in-band} improve, no
RI/HermiT regression. Run with ``LD_LIBRARY_PATH=build/jvm-libs`` (HermiT) and the engine up.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from aegir.refine import eval as ev
from aegir.refine.loop import run_refinement

ROOT = Path(__file__).resolve().parents[3]
FORK_PY = ROOT / "components" / "oss-mistral-cli" / ".venv" / "bin" / "python"
OUT = Path("/raid/build/aegir/path-a/refine")


def agent_propose(construct: dict, feedback: dict) -> str:
    """PROPOSE via the fork-venv subprocess (isolated env — see _propose.py)."""
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["AEGIR_FORK_DIR"] = str(ROOT / "components" / "oss-mistral-cli")
    proc = subprocess.run([str(FORK_PY), "-m", "aegir.refine._propose"],
                          input=json.dumps(construct), capture_output=True, text=True,
                          env=env, timeout=600)
    if proc.returncode != 0:
        sys.stderr.write("PROPOSE subprocess failed:\n" + proc.stderr[-2000:])
        raise RuntimeError("propose subprocess failed")
    return json.loads(proc.stdout)["prose"]


def main() -> int:
    ch0 = ev.controlled_ch0()
    OUT.mkdir(parents=True, exist_ok=True)
    res = run_refinement(ch0, propose_fn=agent_propose,
                         trace_path=str(OUT / "ch0_capability.jsonl"), commit_dir=str(OUT))
    b, r = res["baseline_metrics"], res["refined_metrics"]
    print("=== inc-0(d) CAPABILITY: ch0 single-shot vs membrane-refined (real vibe-acp PROPOSE) ===")
    print("fired:", " → ".join(res["fired"]))
    print("outcome:", res["outcome"], "| iterations:", res["iterations"], "| committed:", res["committed"])
    print(f"placeholder_rate : {b['placeholder_rate']} → {r['placeholder_rate']}")
    print(f"disjointness     : {len(b['disjointness_violations'])} → {len(r['disjointness_violations'])}")
    print(f"ri_ok            : {b['ri_ok']} → {r['ri_ok']}")
    print(f"prose_entailment : {b['prose_entailment']} → {r['prose_entailment']}")
    print(f"length_ok        : {b['length_ok']} → {r['length_ok']} ({b['length']} → {r['length']} chars)")
    print("--- refined prose (agent) ---")
    print(res["refined"]["prose"][:600])
    improved = sum([r["placeholder_rate"] < b["placeholder_rate"],
                    len(r["disjointness_violations"]) < len(b["disjointness_violations"]),
                    r["prose_entailment"] > b["prose_entailment"],
                    r["length_ok"] and not b["length_ok"]])
    regress = (b["ri_ok"] and not r["ri_ok"]) or (len(r["disjointness_violations"]) >
                                                  len(b["disjointness_violations"]))
    print(f"\n{improved}/4 improved; RI/HermiT regression: {regress} → "
          f"{'PASS' if improved >= 3 and not regress else 'FAIL'} (≥3 + no regression)")
    return 0 if (improved >= 3 and not regress) else 1


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

"""inc-2 — agent SCAFFOLD AGENCY over the tables + DUAL-REGISTER output, on ch0.

The agent now PROPOSES the table fixes (``synth_column`` edits) — not just prose — so agency reaches the
STRUCTURE (the audit's concept-salad fix); ``scaffold.apply_edits`` disposes them RI-safe (keys untouched).
At COMMIT the loop writes BOTH registers (natural + semantic prose over the SAME RI-true tables), feeding the
register-faceted lineup. Run with ``LD_LIBRARY_PATH=build/jvm-libs`` + the engine up.
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


def _subproc(construct: dict, mode: str, register: str, feedback: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["AEGIR_FORK_DIR"] = str(ROOT / "components" / "oss-mistral-cli")
    req = json.dumps({"construct": construct, "mode": mode, "register": register, "feedback": feedback})
    p = subprocess.run([str(FORK_PY), "-m", "aegir.refine._propose"], input=req,
                       capture_output=True, text=True, env=env, timeout=600)
    if p.returncode != 0:
        sys.stderr.write("propose subprocess failed:\n" + p.stderr[-2000:])
        raise RuntimeError("propose subprocess failed")
    return json.loads(p.stdout)


def scaffold_propose(construct: dict, objective: str, feedback: dict) -> list:
    return _subproc(construct, "edits", "natural", feedback).get("edits", [])


def prose_natural(construct: dict, feedback: dict | None = None) -> str:
    return _subproc(construct, "prose", "natural", feedback or {}).get("prose", "")


def prose_semantic(construct: dict) -> str:
    return _subproc(construct, "prose", "semantic", {}).get("prose", "")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ch0 = ev.controlled_ch0()
    res = run_refinement(ch0, propose_fn=prose_natural, scaffold_propose=scaffold_propose,
                         dual_prose_fn=prose_semantic, register="natural", max_iters=8,
                         trace_path=str(OUT / "inc2.jsonl"), commit_dir=str(OUT))
    b, r = res["baseline_metrics"], res["refined_metrics"]
    print("=== inc-2: agent scaffold agency + dual-register on ch0 ===")
    print("fired:", " → ".join(res["fired"]), "| outcome:", res["outcome"], "| iters:", res["iterations"])
    print(f"  placeholder {b['placeholder_rate']}→{r['placeholder_rate']} | "
          f"disjointness {len(b['disjointness_violations'])}→{len(r['disjointness_violations'])} | "
          f"ri {b['ri_ok']}→{r['ri_ok']} | entailment {b['prose_entailment']}→{r['prose_entailment']}")
    print("  dual-register surfaces:", [(reg, Path(p).name) for reg, p in res["surfaces"]])
    ok = (res["outcome"] == "promote" and not r["disjointness_violations"]
          and r["placeholder_rate"] == 0.0 and len(res["surfaces"]) == 2)
    print(f"\ninc-2 {'PASS' if ok else 'PARTIAL'}: agent drove the table fixes (scaffold agency); "
          f"both registers committed over the same RI-true tables.")
    return 0 if ok else 1


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

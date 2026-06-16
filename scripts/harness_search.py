"""harness_search — the candidate filesystem + interface validation + eval (inc-2b).

The substrate the Meta-Harness outer loop (inc-2c) navigates. Each candidate harness
is a self-contained `harness.py` living in its own dir alongside its full traces and
scores — the UNCOMPRESSED feedback channel the paper (Lee et al. 2026) shows beats
scalar-only feedback:

    <workspace>/candidates/<NNN>/
        harness.py            # the single-file harness program (seed copy or proposer-mutated)
        meta.json             # {origin, parent, note}
        traces/topic_<id>.jsonl   # full per-topic run trace (every mint+gate+objective)
        scores.json           # {reward (batch on-vs-shuffled R1), cost, per_topic}

This module provides: init_search (seed → 000/), load_harness (dynamic import),
validate_harness (cheap FIXTURE smoke — no Grok/JVM), evaluate_harness (live run over
a search set → traces + scores). The driver (mediate_h0.py) wires the FROZEN executors.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import coverage_r1 as cr  # noqa: E402  (pure; no acp/JVM)

SEED_HARNESS = REPO / "src/aegir/meta_harness/harness_h0.py"


# ── candidate filesystem ──────────────────────────────────────────────────
def candidate_dir(workspace: str | Path, nnn: int) -> Path:
    return Path(workspace) / "candidates" / f"{nnn:03d}"


def init_search(workspace: str | Path, seed: str | Path = SEED_HARNESS) -> Path:
    """Create candidates/000/ from the committed seed harness. Returns 000/'s path."""
    d = candidate_dir(workspace, 0)
    (d / "traces").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed, d / "harness.py")
    (d / "meta.json").write_text(json.dumps(
        {"origin": "seed", "parent": None, "seed_src": str(seed),
         "note": "H₀ — flat transcription of fsm_rete.seed_rules"}, indent=2) + "\n")
    return d


def load_harness(cdir: str | Path):
    """Dynamic-import a candidate's harness.py as an isolated module."""
    cdir = Path(cdir)
    path = cdir / "harness.py"
    name = f"harness_cand_{cdir.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load harness at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── interface validation (cheap; fixtures, not capability) ─────────────────
_FIXTURE_RESP = (
    "```json\n[{"
    '"template_id": "fixture_probe_construct", '
    '"manchester_template": "Class: {X:Class} SubClassOf: cco:DescriptiveICE, {p:ObjectProperty} some {Y:Class}", '
    '"slot_types": {"X": "Class", "p": "ObjectProperty", "Y": "Class"}, '
    '"bfo_anchor_path": ["cco:DescriptiveICE"]}]\n```'
)


def validate_harness(cdir: str | Path) -> tuple[bool, str]:
    """Smoke-test the harness INTERFACE with fixture executors (no Grok/JVM/network).
    A fixture `complete` returns a parseable construct; a fixture `gate` returns
    fail-then-pass signals so run() must thread mint→gate→reorient→promote. This
    asserts the harness is RUNNABLE and honors the contract — it claims NOTHING about
    capability (that is evaluate_harness on real topics). Mirrors the role of
    fsm_rete.FixtureEffector."""
    try:
        h = load_harness(cdir)
    except Exception as e:
        return False, f"import failed: {type(e).__name__}: {e}"
    if not callable(getattr(h, "run", None)):
        return False, "no callable run()"

    calls = {"complete": 0, "gate": 0}

    def fix_complete(prompt: str) -> str:
        calls["complete"] += 1
        return _FIXTURE_RESP

    base = {"deeponto_ok": True, "deeponto_complex": True, "consistent": True,
            "polyglot_ok": True, "novelty_ok": True, "schema_ok": True, "r1_on": 0.4, "n_cols": 5}

    def fix_gate(construct, topic):
        calls["gate"] += 1
        # 1st construct: R1 not specific (→ enrich); 2nd: full contract (→ promote)
        return {**base, "r1_ci_low": (0.02 if calls["gate"] >= 2 else -0.01)}

    topic = {"topic_id": -1, "top_family": "01_foundation", "topic_repr_text": "fixture source text",
             "_top_terms": ["alpha", "beta", "gamma"]}
    try:
        rec = h.run(topic, fix_gate, fix_complete, exemplars={}, max_iters=4)
    except Exception as e:
        return False, f"run() raised: {type(e).__name__}: {e}"
    for k in ("outcome", "iters", "construct", "signals", "history"):
        if k not in rec:
            return False, f"run() record missing '{k}'"
    if rec["outcome"] != "promote":
        return False, f"fixture should promote, got {rec['outcome']!r} (objectives={rec.get('objectives')})"
    if rec["iters"] != 2:
        return False, f"fixture should take 2 iters (enrich→promote), got {rec['iters']}"
    return True, f"OK (promote in {rec['iters']} iters; complete×{calls['complete']}, gate×{calls['gate']})"


# ── reward / eval over a search set ────────────────────────────────────────
def _batch_r1(admitted: list[dict], Vt, rng, n_shuffled: int = 40) -> dict:
    """The G-cov verdict: pooled on-vs-shuffled R1 across promoted constructs."""
    deltas, ons = [], []
    for a in admitted:
        t, tid = a["template"], int(a["topic_id"])
        vc = cr.construct_terms(t)
        if not vc or tid >= len(Vt) or not Vt[tid]:
            continue
        on = cr.r1(vc, Vt[tid], use_embed=False)
        others = [j for j in range(len(Vt)) if j != tid and Vt[j]]
        if not others:
            continue
        sh = np.mean([cr.r1(vc, Vt[j], use_embed=False)
                      for j in rng.choice(others, size=min(n_shuffled, len(others)), replace=False)])
        deltas.append(on - float(sh))
        ons.append(float(on))
    if not deltas:
        return {"n": 0}
    m, lo, hi = cr.boot_ci(np.array(deltas), rng)
    return {"n": len(deltas), "delta": float(m), "ci": [float(lo), float(hi)],
            "ci_clean": bool(lo > 0), "on_mean": float(np.mean(ons))}


def evaluate_harness(cdir: str | Path, topics: list[dict], gate, complete,
                     exemplars: dict, Vt, rng, max_iters: int = 6, tag: str = "") -> dict:
    """Run the candidate harness over a topic set; write per-topic traces +
    scores[_tag].json. reward = batch on-vs-shuffled R1 across promoted constructs (the
    G-cov gate verdict); cost = total mint calls. `tag` namespaces the artifacts so the
    same candidate can be scored on a SEARCH set and a HELD-OUT set without clobbering
    (the outer loop, inc-2c). Returns the scores dict."""
    cdir = Path(cdir)
    tdir = cdir / "traces" / tag if tag else cdir / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    h = load_harness(cdir)

    per_topic, admitted, mint_calls = [], [], 0
    for topic in topics:
        tid = int(topic["topic_id"])
        trace: list[dict] = []
        rec = h.run(topic, gate, complete, exemplars, max_iters=max_iters, log=trace.append)
        (tdir / f"topic_{tid}.jsonl").write_text(
            "\n".join(json.dumps(e) for e in trace) + ("\n" if trace else ""))
        mint_calls += rec["iters"]
        c = rec.get("construct") or {}
        sig = rec.get("signals") or {}
        per_topic.append({"topic_id": tid, "outcome": rec["outcome"], "iters": rec["iters"],
                          "construct_id": c.get("template_id"), "objectives": rec.get("objectives"),
                          "r1_on": sig.get("r1_on"), "r1_ci_low": sig.get("r1_ci_low")})
        if rec["outcome"] == "promote" and c.get("template") is not None:
            admitted.append({"template": c["template"], "topic_id": tid})

    reward = _batch_r1(admitted, Vt, rng)
    scores = {"reward": reward,
              "cost": {"mint_calls": mint_calls, "promoted": len(admitted), "attempted": len(topics)},
              "per_topic": per_topic}
    (cdir / (f"scores_{tag}.json" if tag else "scores.json")).write_text(json.dumps(scores, indent=2) + "\n")
    return scores


if __name__ == "__main__":
    # Cheap self-test: init the search workspace from the committed seed, then
    # validate the H₀ interface with fixtures (no Grok/JVM). Capability is
    # evaluate_harness on real topics (mediate_h0.py).
    import tempfile
    ws = Path(tempfile.mkdtemp(prefix="harness_search_"))
    d0 = init_search(ws)
    ok, msg = validate_harness(d0)
    print(f"init_search → {d0}")
    print(f"validate_harness(000): {'PASS' if ok else 'FAIL'} — {msg}")
    assert ok, msg
    print("\nH₀ interface validated (fixtures). candidate filesystem stood up.")

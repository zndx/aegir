#!/usr/bin/env python
"""harness_outer_loop — the Meta-Harness OUTER LOOP (inc-2c). Algorithm 1.

The actual FORM (Lee, Khattab, Finn 2026): a coding-agent PROPOSER searches over
HARNESS CODE, reading the candidate filesystem (`candidates/{NNN}/` — full harness.py
+ traces + scores, the uncompressed feedback channel), proposing improved single-file
harnesses; each is interface-validated then evaluated on a SEARCH set; the
reward-vs-cost Pareto frontier is carried; the frontier is finally judged on a
HELD-OUT set the proposer never saw. A discovered harness is adopted iff it beats H₀
on held-out coverage-close.

  reward (coverage-close) = mean over ATTEMPTED topics of the per-construct on-vs-
        shuffled R1 delta, scoring unpromoted topics 0 (no construct admitted = no
        coverage closed). = batch Δ × promoted / attempted.
  cost  = total mint calls (Grok completions) across the set.

Proposer is pluggable (Proposer protocol). `GrokCoderProposer` = Grok-as-coder over
ACP (programmatic/repeatable). A stronger proposer (Opus via the Agent tool) plugs into
the same seam — it writes candidates/{NNN}/harness.py and this module validates/evaluates/
ranks it identically.

Run (JVM libs; bounded live search):
  LD_LIBRARY_PATH=$(pwd)/build/jvm-libs:$(pwd)/build/cuda-driver-libs \
    uv run --no-sync python scripts/harness_outer_loop.py \
    --coverage-run /raid/.../coverage_v1/043d7dcc185245c8 \
    --n-search 3 --n-held 3 --iters 2 --k 2 --max-iters 4
Self-test (pure mechanics, no Grok/JVM):  uv run --no-sync python scripts/harness_outer_loop.py --self-test
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import coverage_r1 as cr  # noqa: E402
import harness_search as hs  # noqa: E402

log = logging.getLogger("outer_loop")


# ── reward / Pareto (pure) ─────────────────────────────────────────────────
def cov_reward(scores: dict) -> tuple[float, int]:
    """(coverage-close reward, cost) from an evaluate_harness scores dict. Reward =
    batch Δ × promoted / attempted (unpromoted topics contribute 0 closure)."""
    r = scores.get("reward", {})
    c = scores.get("cost", {})
    attempted = max(1, c.get("attempted", 1))
    delta = r.get("delta", 0.0) if r.get("n") else 0.0
    promoted = r.get("n", 0)
    return float(delta * promoted / attempted), int(c.get("mint_calls", 0))


def pareto_frontier(items: list[dict]) -> list[dict]:
    """Non-dominated set: maximize reward, minimize cost. items: [{id, reward, cost, ...}]."""
    front = []
    for a in items:
        dominated = any(b is not a and b["reward"] >= a["reward"] and b["cost"] <= a["cost"]
                        and (b["reward"] > a["reward"] or b["cost"] < a["cost"]) for b in items)
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda x: (-x["reward"], x["cost"]))


# ── topic split ────────────────────────────────────────────────────────────
def sample_gap_topics(coverage_run: str | Path, n_search: int, n_held: int,
                      seed: int, search=None, held=None) -> tuple[list[int], list[int]]:
    """Disjoint search / held-out gap-topic ids (seeded). Explicit lists override."""
    import pyarrow.parquet as pq
    rows = pq.read_table(Path(coverage_run) / "topic_coverage.parquet").to_pylist()
    gaps = sorted(r["topic_id"] for r in rows if r.get("status") == "gap")
    if search and held:
        return [int(x) for x in search], [int(x) for x in held]
    rng = np.random.default_rng(seed)
    pick = rng.choice(gaps, size=min(n_search + n_held, len(gaps)), replace=False)
    pick = [int(x) for x in pick]
    return pick[:n_search], pick[n_search:n_search + n_held]


# ── proposer ───────────────────────────────────────────────────────────────
class Proposer(Protocol):
    def propose(self, parent_dir: Path, k: int) -> list[str]: ...   # -> k harness.py sources


_PY_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _extract_py(resp: str) -> str | None:
    m = _PY_RE.findall(resp or "")
    # the longest fenced block is the harness (avoid grabbing a tiny snippet)
    return max((b for b in m), key=len).strip() if m else None


def _trace_excerpt(parent_dir: Path, tag: str = "search", max_topics: int = 3) -> str:
    tdir = parent_dir / "traces" / tag
    if not tdir.exists():
        tdir = parent_dir / "traces"
    out = []
    for f in sorted(tdir.glob("topic_*.jsonl"))[:max_topics]:
        out.append(f"# {f.name}\n" + f.read_text().strip())
    return "\n".join(out)[:4000]


class GrokCoderProposer:
    """Grok-as-coder over ACP: reads the parent harness + its scores + trace excerpts
    and proposes k improved single-file harnesses. `complete` is an injected ACP
    completion seam (a SEPARATE grok session from the mint executor)."""

    def __init__(self, complete):
        self.complete = complete

    def _prompt(self, parent_dir: Path, variant: int) -> str:
        src = (parent_dir / "harness.py").read_text()
        scores = ""
        sp = parent_dir / "scores_search.json"
        if sp.exists():
            scores = sp.read_text()
        excerpt = _trace_excerpt(parent_dir)
        focus = ["Focus on build_prompt: make the agent mint MORE topic-specific compound "
                 "terms (combine 2-3 salient terms into class/property names) and exploit the "
                 "topic's nearest catalog templates as scaffolding.",
                 "Focus on next_objective and the loop: spend fewer mints — reorder/condense "
                 "objectives so a promote is reached sooner; tighten the enrich hint."][variant % 2]
        return (
            "You are improving a Meta-Harness HARNESS PROGRAM (Lee et al. 2026). A harness is a "
            "single Python file that wraps two FROZEN executors — `complete(prompt:str)->str` "
            "(Grok) and `gate(construct,topic)->signals` — and drives an iterate loop to mint an "
            "ontology construct that passes a conjunctive contract, maximizing coverage-close "
            "reward (more topics promoted with higher on-vs-shuffled R1) at low cost (fewer "
            "`complete` calls).\n\n"
            "HARD CONSTRAINTS (interface — do NOT break):\n"
            "  • keep `def run(topic, gate, complete, exemplars, max_iters=6, log=None) -> dict` "
            "returning {outcome,iters,construct,signals,history,objectives}.\n"
            "  • call the FROZEN executors only as `complete(prompt)` and `gate(construct, topic)`; "
            "do NOT reimplement them or change their signatures.\n"
            "  • parse Grok's fenced-JSON construct via the existing `go.parse_templates`; keep the "
            "`generate_ontology as go` import. Promote only when the contract is fully satisfied "
            "(the gate's `consistent`, deeponto_ok/complex, polyglot_ok, novelty_ok, schema_ok, "
            "r1_ci_low>0). You MAY rewrite build_prompt / next_objective / the loop body.\n\n"
            f"IMPROVEMENT FOCUS (variant {variant}): {focus}\n\n"
            f"CURRENT HARNESS (candidates/{parent_dir.name}/harness.py):\n```python\n{src}\n```\n\n"
            f"ITS SEARCH-SET SCORES:\n{scores}\n\nTRACE EXCERPTS (where mints were spent):\n{excerpt}\n\n"
            "Return ONLY the complete improved harness as ONE ```python fenced block.")

    def propose(self, parent_dir: Path, k: int) -> list[str]:
        out = []
        for i in range(k):
            resp = self.complete(self._prompt(parent_dir, i))
            (parent_dir / f"proposer_raw_{i}.txt").write_text(resp or "")   # full response, for diagnosis
            src = _extract_py(resp)
            if src:                       # accept any fenced block; validate_harness rejects partials (logged INVALID)
                out.append(src)
            else:
                log.warning("proposer variant %d: no fenced code block (%d chars)", i, len(resp or ""))
        return out


class StaticProposer:
    """A pre-authored harness proposer — the seam for an EXTERNAL coding agent (the
    Meta-Harness proposer is Claude Code/Opus). The agent reads the candidate
    filesystem and writes harness.py files; this returns their source so the SAME
    loop validates/evaluates/ranks them. Source files are consumed one per iteration."""

    def __init__(self, source_paths: list[str]):
        self.paths = [Path(p) for p in source_paths]
        self._i = 0

    def propose(self, parent_dir: Path, k: int) -> list[str]:
        out = []
        for _ in range(k):
            if self._i >= len(self.paths):
                break
            out.append(self.paths[self._i].read_text())
            self._i += 1
        return out


# ── the loop (Algorithm 1) ─────────────────────────────────────────────────
def run_outer_loop(workspace, proposer: Proposer, search_topics, held_topics, gate, complete,
                   exemplars, Vt, rng, iters: int, k: int, max_iters: int) -> dict:
    ws = Path(workspace)
    d0 = hs.init_search(ws)
    ok, msg = hs.validate_harness(d0)
    log.info("H₀ interface: %s — %s", "PASS" if ok else "FAIL", msg)
    if not ok:
        return {"error": f"H₀ invalid: {msg}"}

    def _eval(cdir, topics, tag):
        s = hs.evaluate_harness(cdir, topics, gate, complete, exemplars, Vt, rng,
                                max_iters=max_iters, tag=tag)
        rew, cost = cov_reward(s)
        return {"id": Path(cdir).name, "dir": str(cdir), "reward": rew, "cost": cost,
                "promoted": s["cost"]["promoted"], "attempted": s["cost"]["attempted"]}

    pool = [_eval(d0, search_topics, "search")]
    log.info("H₀ search: reward=%.4f cost=%d (%d/%d promoted)",
             pool[0]["reward"], pool[0]["cost"], pool[0]["promoted"], pool[0]["attempted"])
    nnn = 1
    for it in range(iters):
        parent = pareto_frontier(pool)[0]                 # best-reward frontier point
        log.info("iter %d: proposing %d from parent %s (reward=%.4f)", it, k, parent["id"], parent["reward"])
        for src in proposer.propose(Path(parent["dir"]), k):
            cdir = hs.candidate_dir(ws, nnn)
            (cdir / "traces").mkdir(parents=True, exist_ok=True)
            (cdir / "harness.py").write_text(src)
            (cdir / "meta.json").write_text(json.dumps(
                {"origin": "proposer", "parent": parent["id"], "iter": it}, indent=2) + "\n")
            ok, msg = hs.validate_harness(cdir)
            if not ok:
                log.warning("  cand %03d INVALID — %s", nnn, msg)
                (cdir / "INVALID.txt").write_text(msg + "\n")
                nnn += 1
                continue
            rec = _eval(cdir, search_topics, "search")
            log.info("  cand %03d search: reward=%.4f cost=%d (%d/%d)", nnn, rec["reward"],
                     rec["cost"], rec["promoted"], rec["attempted"])
            pool.append(rec)
            nnn += 1

    front = pareto_frontier(pool)
    log.info("Pareto frontier (search): %s", [(c["id"], round(c["reward"], 4), c["cost"]) for c in front])

    # held-out: judge H₀ and every frontier candidate the proposer never saw
    held = {}
    for c in [pool[0]] + [f for f in front if f["id"] != "000"]:
        held[c["id"]] = _eval(c["dir"], held_topics, "held")
    h0_held = held["000"]["reward"]
    winners = [(cid, h) for cid, h in held.items() if cid != "000" and h["reward"] > h0_held]
    verdict = {
        "h0_held_reward": h0_held, "h0_search_reward": pool[0]["reward"],
        "frontier_search": [{k2: c[k2] for k2 in ("id", "reward", "cost", "promoted", "attempted")} for c in front],
        "held": {cid: {k2: h[k2] for k2 in ("reward", "cost", "promoted", "attempted")} for cid, h in held.items()},
        "beats_h0_on_held": [{"id": cid, "held_reward": h["reward"], "delta_vs_h0": h["reward"] - h0_held}
                             for cid, h in sorted(winners, key=lambda x: -x[1]["reward"])],
        "discovered_winner": (sorted(winners, key=lambda x: -x[1]["reward"])[0][0] if winners else None),
    }
    (ws / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def _self_test() -> int:
    # pure-mechanics check: cov_reward + pareto_frontier (no Grok/JVM).
    s = {"reward": {"n": 2, "delta": 0.30}, "cost": {"mint_calls": 5, "promoted": 2, "attempted": 4}}
    rew, cost = cov_reward(s)
    assert abs(rew - 0.15) < 1e-9 and cost == 5, (rew, cost)          # 0.30 * 2/4
    empty = cov_reward({"reward": {"n": 0}, "cost": {"mint_calls": 3, "attempted": 4}})
    assert empty == (0.0, 3), empty
    items = [{"id": "000", "reward": 0.15, "cost": 5}, {"id": "001", "reward": 0.20, "cost": 5},
             {"id": "002", "reward": 0.20, "cost": 8}, {"id": "003", "reward": 0.10, "cost": 2}]
    front = [c["id"] for c in pareto_frontier(items)]
    assert front == ["001", "003"], front     # 001 dominates 000&002; 003 cheapest non-dominated
    print("cov_reward + pareto_frontier OK:", front)
    print("\nouter-loop mechanics verified (pure). Live search = the JVM/Grok run.")
    return 0


def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--coverage-run")
    ap.add_argument("--n-search", type=int, default=3)
    ap.add_argument("--n-held", type=int, default=3)
    ap.add_argument("--search", help="explicit search topic_ids (comma-sep; overrides sampling)")
    ap.add_argument("--held", help="explicit held-out topic_ids (comma-sep)")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--propose-files", default=None,
                    help="comma-sep harness.py paths → StaticProposer (Opus-authored candidates) "
                         "instead of Grok-as-coder")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--seed", type=int, default=20260616)
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.coverage_run:
        ap.error("--coverage-run required for a live run")

    rng = np.random.default_rng(args.seed)
    sids, hids = sample_gap_topics(args.coverage_run, args.n_search, args.n_held, args.seed,
                                   search=args.search.split(",") if args.search else None,
                                   held=args.held.split(",") if args.held else None)
    log.info("search=%s  held-out=%s", sids, hids)
    ws = Path(args.workspace or f"/raid/checkpoints/aegir-artifacts/evidence/meta_harness/outer_{sids[0]}")
    ws.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    rows = pq.read_table(Path(args.coverage_run) / "topic_coverage.parquet").to_pylist()
    by = {r["topic_id"]: r for r in rows}
    reprs = [(by.get(i, {}).get("topic_repr_text") or "") for i in range(len(rows))]
    Vt = cr.topic_term_signatures(reprs)
    mk = lambda ids: [dict(by[i], _top_terms=sorted(Vt[i])) for i in ids]  # noqa: E731

    from mediate_gate import ContractGate
    from mediate_acp import ACPMintEffector
    gate = ContractGate(args.coverage_run)
    mint_eff = ACPMintEffector()            # mint executor (frozen Grok, session A)
    coder_eff = None
    if args.propose_files:
        proposer = StaticProposer(args.propose_files.split(","))   # Opus-authored candidates
        log.info("proposer: StaticProposer over %s", args.propose_files.split(","))
    else:
        coder_eff = ACPMintEffector()       # coder/proposer (frozen Grok, session B)
        proposer = GrokCoderProposer(coder_eff.complete)
    try:
        verdict = run_outer_loop(ws, proposer, mk(sids), mk(hids),
                                 gate.gate, mint_eff.complete, mint_eff.exemplars, Vt, rng,
                                 iters=args.iters, k=args.k, max_iters=args.max_iters)
    finally:
        mint_eff.close()
        if coder_eff is not None:
            coder_eff.close()

    print("\n=== OUTER-LOOP VERDICT ===")
    print(f"  H₀ held-out reward: {verdict.get('h0_held_reward'):.4f}")
    w = verdict.get("discovered_winner")
    if w:
        bh = verdict["beats_h0_on_held"][0]
        print(f"  DISCOVERED WINNER: {w} — held reward {bh['held_reward']:.4f} "
              f"(Δ {bh['delta_vs_h0']:+.4f} vs H₀). A discovered harness beats H₀ on held-out.")
    else:
        print("  no discovered harness beat H₀ on held-out (machinery validated; proposer underperformed).")
    print(f"  artifacts → {ws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

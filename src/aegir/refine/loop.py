"""inc-0(d) refinement loop — the meta-harness FSM orchestrating PROPOSE → gates → COMMIT.

The membrane-oracle invariant is structural and inherited from the spine (``fsm_rete.MetaHarness``): the
agent only acts through ``Effector.mint`` (PROPOSE); the deterministic gates live in ``Effector.gate`` and the
salience-ordered rules route on their signals. The agent can request a refinement but never runs, fakes, or
bypasses a gate — and the cascade can only terminate ``promote`` when every gate passes.

Stages (rules below): PROPOSE (mint) → VALUE_GATE (HermiT) → PLACEHOLDER → RI_GATE → PROSE_GATE → COMMIT.
Table-level remediations (value-disjointness, placeholders) are DETERMINISTIC membrane effectors here; the
agent owns PROSE. inc-2 moves table remediation into the agent's hands via scaffold tools (so agency reaches
the structure, per the audit's concept-salad risk). ``propose_fn`` is injected: a fork-venv subprocess for the
real ``vibe-acp`` agent (capability), or a deterministic function (control-plane logic test).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from aegir.meta_harness.fsm_rete import Effect, MetaHarness, Rule
from aegir.refine import eval as ev

# objective → handled in mint(); thresholds for the gate booleans
PLACEHOLDER_MAX = 0.05
ENTAILMENT_MIN = 0.50


def refinement_rules() -> list[Rule]:
    g = lambda s, k, d=False: s.get(k, d)
    return [
        Rule("contract_satisfied", 120,
             lambda s, c: (g(s, "has_construct") and g(s, "value_ok") and g(s, "placeholder_ok")
                           and g(s, "ri_ok") and g(s, "prose_ok")),
             Effect("terminate", "promote", "all gates pass (value+placeholder+RI+prose)"), specificity=8),
        Rule("budget_exhausted", 110,
             lambda s, c: c["iterations"] >= c["max_iters"],
             Effect("terminate", "give_up", "max iterations")),
        Rule("no_construct", 100,
             lambda s, c: not g(s, "has_construct"),
             Effect("objective", "draft_initial")),
        Rule("value_fail", 90,
             lambda s, c: g(s, "has_construct") and not g(s, "value_ok"),
             Effect("objective", "fix_value"), specificity=3),
        Rule("placeholder_fail", 80,
             lambda s, c: g(s, "has_construct") and g(s, "value_ok") and not g(s, "placeholder_ok"),
             Effect("objective", "fix_placeholders"), specificity=2),
        Rule("ri_fail", 75,
             lambda s, c: g(s, "has_construct") and not g(s, "ri_ok"),
             Effect("objective", "fix_ri")),
        Rule("prose_fail", 70,
             lambda s, c: (g(s, "has_construct") and g(s, "value_ok") and g(s, "placeholder_ok")
                           and g(s, "ri_ok") and not g(s, "prose_ok")),
             Effect("objective", "fix_prose")),
    ]


def _load_value_pools():
    for p in (Path("src/aegir/ontology/entity_value_pools.json"),
              Path(__file__).resolve().parents[1] / "ontology" / "entity_value_pools.json"):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                return {}
    return {}


class RefinementEffector:
    def __init__(self, ch0: dict, propose_fn: Callable[[dict, dict], str],
                 scaffold_propose: "Callable | None" = None) -> None:
        self.ch0 = ch0
        self.propose_fn = propose_fn
        self.scaffold_propose = scaffold_propose   # inc-2: agent proposes structured table edits
        self.pools = _load_value_pools()

    # ── gates → signals ──────────────────────────────────────────────────────
    def gate(self, construct: dict, topic) -> dict:
        m = ev.measure(construct)
        return {
            "value_ok": len(m["disjointness_violations"]) == 0,
            "placeholder_ok": m["placeholder_rate"] <= PLACEHOLDER_MAX,
            "ri_ok": m["ri_ok"],
            "prose_ok": m["prose_entailment"] >= ENTAILMENT_MIN and m["length_ok"],
            **m,
        }

    # ── PROPOSE / remediations ───────────────────────────────────────────────
    def mint(self, topic, objective: str, feedback: dict, prev: dict | None) -> dict:
        if objective == "draft_initial":
            return json.loads(json.dumps(self.ch0))   # fresh deep copy of the single-shot baseline
        c = json.loads(json.dumps(prev))               # copy-on-write per remediation
        if objective == "fix_value":
            if not self._agent_edit(c, "fix_value", feedback):
                self._fix_value(c)                     # deterministic membrane fallback
        elif objective == "fix_placeholders":
            if not self._agent_edit(c, "fix_placeholders", feedback):
                self._fix_placeholders(c)
        elif objective == "fix_prose":
            c["prose"] = self.propose_fn(c, feedback)
        return c

    def _agent_edit(self, c: dict, objective: str, feedback: dict) -> bool:
        """inc-2 scaffold agency: the agent PROPOSES structured edits; ``scaffold.apply_edits`` DISPOSES them
        RI-safe (keys untouched, values from the ontology pools). Returns whether any edit applied — else mint
        falls back to deterministic remediation, so the membrane always closes the objective."""
        if not self.scaffold_propose:
            return False
        from aegir.refine.scaffold import apply_edits
        try:
            edits = self.scaffold_propose(c, objective, feedback) or []
        except Exception:  # noqa: BLE001 — an agent/transport failure must not break the loop
            return False
        applied, _ = apply_edits(c, edits)
        return bool(applied)

    def _fix_value(self, c: dict) -> None:
        """Deterministic membrane remediation: replace each disjoint-contaminant cell with a clean value of the
        column's own concept (sampled from the column's non-violating cells)."""
        from aegir.refine.value_gate import check_column
        onto = c.get("value_onto", {})
        kw = dict(classes=onto.get("classes", ()), subclass_of=onto.get("subclass_of", ()),
                  disjoint=onto.get("disjoint", ()))
        for t in c.get("tables", []):
            for col in t["columns"]:
                concept = col.get("concept")
                vs = [(str(cell["value"]), cell.get("source", concept)) for cell in col["cells"]]
                if not concept or not vs:
                    continue
                r = check_column(concept, vs, **kw)
                if r["consistent"]:
                    continue
                clean = next((cell["value"] for cell in col["cells"]
                              if str(cell["value"]) not in r["violations"]), concept)
                for cell in col["cells"]:
                    if str(cell["value"]) in r["violations"]:
                        cell["value"], cell["source"] = clean, concept

    def _fix_placeholders(self, c: dict) -> None:
        """Deterministic value-enrichment: replace placeholder cells with a domain value for the column's
        concept (entity_value_pools — the real Comp-4 artifact — else a concept-derived exemplar)."""
        for t in c.get("tables", []):
            for col in t["columns"]:
                concept = col.get("concept", "")
                pool = self.pools.get(concept) or self.pools.get(col["name"]) or []
                used = {str(cell["value"]) for cell in col["cells"]}
                k = 0
                for cell in col["cells"]:
                    val = str(cell.get("value", "")).strip()
                    is_ph = (ev.re.sub(r"[\s_]", "", val.lower()) in ev.PLACEHOLDER_TOKENS
                             or ev.re.fullmatch(r"\{[^}]*\}", val or "x"))
                    if is_ph:
                        repl = next((p for p in pool[k:] if str(p) not in used), None) \
                            or f"{concept}_{k + 1}"
                        cell["value"], used = str(repl), used | {str(repl)}
                        k += 1


def run_refinement(ch0: dict, *, propose_fn: Callable[[dict, dict], str],
                   scaffold_propose: "Callable | None" = None, dual_prose_fn: "Callable | None" = None,
                   register: str = "natural", lineage: "object | None" = None, max_iters: int = 8,
                   trace_path: str | Path | None = None, commit_dir: str | Path | None = None) -> dict:
    """Run the membrane-gated loop on one chapter. With ``scaffold_propose`` the agent gets agency over the
    TABLES (inc-2); with ``dual_prose_fn`` COMMIT writes BOTH registers (the other register's prose over the
    same RI-true tables). Returns {…, refined, surfaces:[(register, path)], committed}."""
    eff = RefinementEffector(ch0, propose_fn, scaffold_propose=scaffold_propose)
    h = MetaHarness(refinement_rules(), eff, max_iters=max_iters, trace_path=trace_path)
    ctx = h.run(ch0.get("template_id", "ch0"))
    refined = ctx.construct or ch0
    fired = [e["fired"] for e in h.trace if e["event"] == "agenda"]
    out = {
        "outcome": ctx.terminate, "iterations": ctx.iterations, "fired": fired,
        "baseline_metrics": ev.measure(ch0), "refined_metrics": ev.measure(refined),
        "refined": refined, "committed": None, "surfaces": [],
    }
    if ctx.terminate == "promote" and commit_dir is not None:   # COMMIT: scratch → current (dual-register)
        cur = Path(commit_dir) / "current"
        cur.mkdir(parents=True, exist_ok=True)
        tid = ch0.get("template_id", "ch0")
        surfaces = [(register, refined)]
        if dual_prose_fn is not None:           # the OTHER register's prose over the SAME RI-true tables
            other = "semantic" if register == "natural" else "natural"
            alt = json.loads(json.dumps(refined))
            try:
                alt["prose"] = dual_prose_fn(refined)
                surfaces.append((other, alt))
            except Exception:  # noqa: BLE001
                pass
        for reg, surf in surfaces:
            p = cur / f"{tid}.{reg}.json"
            p.write_text(json.dumps({**surf, "register": reg}, indent=2))
            out["surfaces"].append((reg, str(p)))
        out["committed"] = out["surfaces"][0][1]
        (cur / f"{tid}.metrics.json").write_text(json.dumps(out["refined_metrics"], indent=2))
        if lineage is not None:   # hx/OL: the surfaces GROUNDS_TO the exchanges + the gate verdicts
            out["lineage"] = lineage.emit(template_id=tid, surfaces=out["surfaces"],
                                          gate_verdicts=out["refined_metrics"])
    return out


def deterministic_prose(construct: dict, feedback: dict) -> str:
    """Control-plane PROPOSE stand-in (no engine): emits prose that mentions the table entities and clears the
    length band, so the loop's routing + gates + measurement can be exercised fast. NOT a capability test."""
    parts = ["This chapter examines the imaging-study dataset and its referential structure."]
    for t in construct.get("tables", []):
        cols = ", ".join(c["name"].replace("_", " ") for c in t["columns"])
        parts.append(f"The {t['name'].replace('_', ' ')} table records {cols}.")
        for col in t["columns"]:
            vals = [str(c["value"]).replace("_", " ") for c in col["cells"][:3]
                    if str(c["value"]).lower() not in ev.PLACEHOLDER_TOKENS]
            if vals and col["name"] != t.get("pk"):
                parts.append(f"Representative {col['name'].replace('_', ' ')} values include "
                             f"{', '.join(vals)}.")
        for fk in t.get("fks", []):
            parts.append(f"Each {fk['col'].replace('_', ' ')} references "
                         f"{fk['ref_table'].replace('_', ' ')} via {fk['ref_col'].replace('_', ' ')}, "
                         f"enforcing referential integrity between the two tables.")
    return " ".join(parts)


if __name__ == "__main__":
    # Control-plane logic test (deterministic PROPOSE — no engine). Asserts the cascade routes
    # draft → value → placeholder → prose → satisfied and the metrics move. Capability (real agent) is the
    # separate run in run_inc0d.py.
    import sys
    ch0 = ev.controlled_ch0()
    res = run_refinement(ch0, propose_fn=deterministic_prose,
                         trace_path="/tmp/refine_ch0_controlplane.jsonl")
    print("fired:", res["fired"])
    print("outcome:", res["outcome"], "| iters:", res["iterations"])
    b, r = res["baseline_metrics"], res["refined_metrics"]
    print(f"placeholder_rate : {b['placeholder_rate']} → {r['placeholder_rate']}")
    print(f"disjointness     : {len(b['disjointness_violations'])} → {len(r['disjointness_violations'])} "
          f"{b['disjointness_violations']}")
    print(f"ri_ok            : {b['ri_ok']} → {r['ri_ok']}")
    print(f"prose_entailment : {b['prose_entailment']} → {r['prose_entailment']}")
    print(f"length_ok        : {b['length_ok']} → {r['length_ok']} ({b['length']} → {r['length']} chars)")
    expected = ["no_construct", "value_fail", "placeholder_fail", "prose_fail", "contract_satisfied"]
    assert res["fired"] == expected, f"cascade route mismatch: {res['fired']}"
    assert res["outcome"] == "promote"
    improved = sum([r["placeholder_rate"] < b["placeholder_rate"],
                    len(r["disjointness_violations"]) < len(b["disjointness_violations"]),
                    r["prose_entailment"] > b["prose_entailment"],
                    r["length_ok"] and not b["length_ok"]])
    assert not (r["ri_ok"] is False and b["ri_ok"] is True), "RI regressed"
    print(f"\n{improved}/4 metrics improved, no RI/HermiT regression — CONTROL-PLANE PASS (≥3 required).")
    sys.exit(0 if improved >= 3 else 1)

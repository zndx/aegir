"""Agent-mediated meta-harness — the RETE/FSM control spine.

A signal-responsive navigator (Holland, *Signals and Boundaries*) that drives an
agent toward a construct passing the ontology contract. See
``docs/current/src/meta_harness_boundary.md`` for the design of record.

This module is the SPINE (the part that cannot be retrofitted): working memory of
facts, salience-ordered production rules, an agenda with logged conflict
resolution, the 5-state FSM, scored objectives, and an append-only trace. The
*matcher* is a naive correct scan behind a stable API — a RETE discrimination
network drops in here later without touching the spine. The *effectors* (the LLM
``mint`` and the deterministic contract ``gate``) are EXTERNAL and injected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

# ── FSM states ────────────────────────────────────────────────────────────
OBSERVE_CONTRACT = "ObserveContract"
SELECT_OBJECTIVE = "SelectObjective"
ACT_AND_SIGNAL = "ActAndSignal"
EVALUATE_FEEDBACK = "EvaluateFeedback"
TERMINATE_OR_ITERATE = "TerminateOrIterate"
DONE = "Done"


# ── Facts / working memory ────────────────────────────────────────────────
@dataclass
class Fact:
    kind: str
    payload: dict
    id: int = -1
    rev: int = 0


class WorkingMemory:
    """Assert / update / retract / query facts; global revision counter.

    The matcher reads facts through ``signals()`` (the alpha-test convenience
    view). A RETE network would compile rule conditions against this same store.
    """

    def __init__(self) -> None:
        self._facts: dict[int, Fact] = {}
        self._by_kind: dict[str, int] = {}
        self._next = 0
        self.revision = 0

    def put(self, kind: str, payload: dict) -> Fact:
        """Assert or replace the singleton fact of ``kind`` (this spine keeps one
        fact per kind: ``topic``, ``signals``, ``state``, ``objective``)."""
        self.revision += 1
        if kind in self._by_kind:
            f = self._facts[self._by_kind[kind]]
            f.payload = dict(payload)
            f.rev = self.revision
            return f
        f = Fact(kind=kind, payload=dict(payload), id=self._next, rev=self.revision)
        self._facts[self._next] = f
        self._by_kind[kind] = self._next
        self._next += 1
        return f

    def get(self, kind: str) -> dict:
        fid = self._by_kind.get(kind)
        return dict(self._facts[fid].payload) if fid is not None else {}

    def signals(self) -> dict:
        """Flattened view the rules match against."""
        return self.get("signals")


# ── Rules / agenda ────────────────────────────────────────────────────────
@dataclass
class Effect:
    kind: str          # "objective" | "terminate"
    value: str         # objective name | termination outcome ("promote"/"give_up")
    reason: str = ""


@dataclass
class Rule:
    name: str
    salience: int
    when: Callable[[dict, dict], bool]   # (signals, ctx_view) -> bool   [alpha-test layer]
    then: Effect
    specificity: int = 0                 # tie-break after salience (author-declared)
    order: int = 0                       # final tie-break (declaration recency)


class Agenda:
    """Conflict resolution: collect the matched rules, resolve by
    (salience ↓, specificity ↓, declaration order ↑), fire ONE per cycle,
    and hand back the full conflict set for logging."""

    @staticmethod
    def resolve(matched: list[Rule]) -> tuple[Rule | None, list[str]]:
        if not matched:
            return None, []
        ordered = sorted(matched, key=lambda r: (-r.salience, -r.specificity, r.order))
        return ordered[0], [r.name for r in ordered]


# ── Objectives ────────────────────────────────────────────────────────────
@dataclass
class Objective:
    name: str
    score: Callable[[dict], float]          # attainability from signals
    satisfied: Callable[[dict], bool] = lambda s: False


# ── Effector protocol (EXTERNAL to the spine; injected) ───────────────────
class Effector(Protocol):
    def mint(self, topic, objective: str, feedback: dict, prev: dict | None) -> dict: ...
    def gate(self, construct: dict, topic) -> dict: ...   # -> signal dict


# ── Run context + trace ───────────────────────────────────────────────────
@dataclass
class Ctx:
    topic: object
    iterations: int = 0
    construct: dict | None = None
    objective: str | None = None
    terminate: str | None = None
    terminate_reason: str = ""
    history: list = field(default_factory=list)   # r1_on per iteration


# ── The spine ─────────────────────────────────────────────────────────────
class MetaHarness:
    def __init__(self, rules: list[Rule], effector: Effector, max_iters: int = 8,
                 trace_path: str | Path | None = None) -> None:
        for i, r in enumerate(rules):
            r.order = i
        self.rules = rules
        self.effector = effector
        self.max_iters = max_iters
        self.trace: list[dict] = []
        self.trace_path = Path(trace_path) if trace_path else None
        self.wm = WorkingMemory()

    def _emit(self, event: str, data: dict) -> None:
        self.trace.append({"event": event, "rev": self.wm.revision, **data})

    def _set_signals(self, sig: dict) -> None:
        self.wm.put("signals", sig)
        self._emit("signals", {"signals": sig})

    def _decide(self, ctx: Ctx) -> None:
        """One rule cycle: match → resolve → fire one. Sets ctx.objective or
        ctx.terminate. Logs the full conflict set + the chosen rule."""
        sig = self.wm.signals()
        ctx_view = {"iterations": ctx.iterations, "max_iters": self.max_iters}
        matched = [r for r in self.rules if r.when(sig, ctx_view)]
        chosen, conflict = Agenda.resolve(matched)
        self._emit("agenda", {"conflict_set": conflict, "fired": chosen.name if chosen else None})
        if chosen is None:
            ctx.terminate, ctx.terminate_reason = "give_up", "no rule matched (stuck)"
            return
        eff = chosen.then
        if eff.kind == "terminate":
            ctx.terminate, ctx.terminate_reason = eff.value, eff.reason or chosen.name
        else:
            ctx.objective = eff.value

    def _transition(self, ctx: Ctx, state: str) -> None:
        self.wm.put("state", {"state": state})
        self._emit("state", {"state": state, "iter": ctx.iterations,
                             "objective": ctx.objective})

    def run(self, topic) -> Ctx:
        ctx = Ctx(topic=topic)
        self.wm.put("topic", {"topic": str(topic)})
        self._emit("run_start", {"topic": str(topic), "max_iters": self.max_iters})
        state = OBSERVE_CONTRACT
        guard = 0
        while state != DONE and guard < 4 * self.max_iters + 8:
            guard += 1
            self._transition(ctx, state)
            if state == OBSERVE_CONTRACT:
                if ctx.construct is None:
                    self._set_signals({"has_construct": False, "iterations": ctx.iterations})
                else:
                    self._set_signals({**self.effector.gate(ctx.construct, topic),
                                       "has_construct": True, "iterations": ctx.iterations})
                state = SELECT_OBJECTIVE
            elif state == SELECT_OBJECTIVE:
                self._decide(ctx)
                state = TERMINATE_OR_ITERATE if ctx.terminate else ACT_AND_SIGNAL
            elif state == ACT_AND_SIGNAL:
                assert ctx.objective is not None  # SELECT/TERMINATE only route here with an objective set
                feedback = self.wm.signals()
                ctx.construct = self.effector.mint(topic, ctx.objective, feedback, ctx.construct)
                ctx.iterations += 1
                self._emit("mint", {"objective": ctx.objective, "iter": ctx.iterations,
                                    "construct_id": (ctx.construct or {}).get("template_id")})
                state = EVALUATE_FEEDBACK
            elif state == EVALUATE_FEEDBACK:
                sig = {**self.effector.gate(ctx.construct, topic),
                       "has_construct": True, "iterations": ctx.iterations}
                self._set_signals(sig)
                ctx.history.append(sig.get("r1_on"))
                state = TERMINATE_OR_ITERATE
            elif state == TERMINATE_OR_ITERATE:
                self._decide(ctx)
                state = DONE if ctx.terminate else ACT_AND_SIGNAL
        if state != DONE:
            ctx.terminate = ctx.terminate or "give_up"
            ctx.terminate_reason = ctx.terminate_reason or "guard tripped"
        self._transition(ctx, DONE)
        self._emit("run_end", {"outcome": ctx.terminate, "reason": ctx.terminate_reason,
                              "iterations": ctx.iterations, "r1_history": ctx.history,
                              "construct_id": (ctx.construct or {}).get("template_id")})
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text("\n".join(json.dumps(e) for e in self.trace) + "\n")
        return ctx


# ── Seed rule set (grounded in the committed gates; see design note §5) ────
def seed_rules() -> list[Rule]:
    g = lambda s, k, d=False: s.get(k, d)
    return [
        Rule("contract_satisfied", 120,
             lambda s, c: (g(s, "has_construct") and g(s, "deeponto_ok") and g(s, "deeponto_complex")
                           and g(s, "consistent") and g(s, "polyglot_ok") and g(s, "novelty_ok")
                           and g(s, "schema_ok") and s.get("r1_ci_low", -1) > 0),
             Effect("terminate", "promote", "full contract satisfied (incl. HermiT coherence) + R1 CI-clean"),
             specificity=8),
        Rule("budget_exhausted", 110,
             lambda s, c: c["iterations"] >= c["max_iters"],
             Effect("terminate", "give_up", "max iterations")),
        Rule("no_construct", 100,
             lambda s, c: not g(s, "has_construct"),
             Effect("objective", "draft_initial")),
        Rule("deeponto_fail", 90,
             lambda s, c: g(s, "has_construct") and not g(s, "deeponto_ok"),
             Effect("objective", "fix_verbalizability")),
        Rule("inconsistent", 89,
             lambda s, c: g(s, "has_construct") and g(s, "deeponto_ok") and not g(s, "consistent"),
             Effect("objective", "fix_consistency"), specificity=2),
        Rule("not_complex", 88,
             lambda s, c: g(s, "deeponto_ok") and not g(s, "deeponto_complex"),
             Effect("objective", "fix_nontriviality")),
        Rule("polyglot_fail", 80,
             lambda s, c: g(s, "has_construct") and not g(s, "polyglot_ok"),
             Effect("objective", "fix_ddl")),
        Rule("r1_not_specific", 70,
             lambda s, c: (g(s, "deeponto_ok") and g(s, "polyglot_ok")
                           and s.get("r1_ci_low", -1) <= 0),
             Effect("objective", "enrich_domain_terms"), specificity=3),
        Rule("novelty_low", 60,
             lambda s, c: g(s, "has_construct") and not g(s, "novelty_ok"),
             Effect("objective", "diversify")),
        Rule("schema_canned", 50,
             lambda s, c: g(s, "has_construct") and not g(s, "schema_ok"),
             Effect("objective", "de_can")),
    ]


# ── Fixture effector (control-plane LOGIC test only — NOT a capability test) ─
class FixtureEffector:
    """Replays a RECORDED-or-DESIGNED fixture (one gate-signal dict per mint) to
    exercise the deterministic control plane.

    It does NOT reason, and it does NOT simulate reasoning — it is *data*, the way
    a parser test feeds recorded inputs. Reasoning an ontology out of FinePDFs is
    irreducibly intelligent, so the **production effector is ALWAYS a real agent**
    (ACP: Qwen / our fine-tunes / Grok); there is no stub that produces ontology.
    Capability is validated only by a real agent moving R1 (inc-1), NEVER by this
    fixture. A fixture may be recorded from a real run (preferred) or hand-authored
    to drive a specific rule path; it is test data, not a verdict.
    """

    def __init__(self, steps: list[dict]):
        self.steps = steps           # gate-signal dict per mint (recorded/designed)
        self._i = -1

    def mint(self, topic, objective, feedback, prev):
        self._i += 1
        return {"template_id": f"fixture_step_{self._i}", "_obj": objective}

    def gate(self, construct, topic):
        return dict(self.steps[min(self._i, len(self.steps) - 1)])


if __name__ == "__main__":
    # Designed control-plane fixture: a scenario that drives the rule paths
    # draft → verbalize-fail → trivial → R1-not-specific → satisfied. This asserts
    # the LOGIC routes correctly given known signals — it makes NO claim that an
    # ontology was reasoned. The real test is a live agent moving R1 (inc-1).
    OK = {"deeponto_ok": True, "deeponto_complex": True, "consistent": True, "polyglot_ok": True,
          "novelty_ok": True, "schema_ok": True, "r1_on": 0.34, "n_cols": 5}
    fixture = [
        {**OK, "deeponto_ok": False, "deeponto_complex": False},   # 0: first draft won't verbalize
        {**OK, "deeponto_complex": False},                          # 1: verbalizes but trivial
        {**OK, "consistent": False},                                # 2: complex but HermiT-incoherent
        {**OK, "r1_ci_low": -0.01},                                 # 3: coherent but R1 not specific
        {**OK, "r1_ci_low": 0.02},                                  # 4: full contract satisfied
    ]
    h = MetaHarness(seed_rules(), FixtureEffector(fixture), max_iters=8,
                    trace_path="/tmp/meta_harness_logictest.jsonl")
    ctx = h.run("fixture_scenario")
    fired = [e["fired"] for e in h.trace if e["event"] == "agenda"]
    print(f"fired: {fired}")
    print(f"outcome={ctx.terminate} ({ctx.terminate_reason!r})  iters={ctx.iterations}")
    expected = ["no_construct", "deeponto_fail", "not_complex", "inconsistent",
                "r1_not_specific", "contract_satisfied"]
    assert fired == expected, f"control-plane logic path mismatch: {fired}"
    assert ctx.terminate == "promote"
    print(f"\nCONTROL-PLANE LOGIC verified against designed fixture (data, not reasoning). "
          f"Capability is NOT tested here — that is inc-1 (a real agent moving R1).")

"""inc-3 — wire the refinement loop into the hx (Iceberg ``raw.exchange``) + OL/Atlas (AGE) lineage plane.

The loop's agent exchanges (PROPOSE prompts / responses / reasoning) land in ``raw.exchange`` via the SAME
``append_exchange`` the generation pipeline uses; the dual-register COMMIT emits an OpenLineage run-event
(idempotent MERGE into ``aegir_hx``/AGE via ``governance.ol.ingest_run_event``) recording the refined surfaces
as OUTPUTS derived from the exchange INPUTS, with the gate verdicts as a facet — the membrane's decisions
become first-class STRUCTURED provenance, not a bespoke cache. This replaces the de-booked inc-3
skills/transcript-cache (RH 2026-06-24). All lineage writes are **best-effort**: a lineage failure must NEVER
break the refinement loop.
"""
from __future__ import annotations

import datetime
import uuid as _uuid


def new_run_id() -> str:
    return str(_uuid.uuid4())


class LineageRecorder:
    """Collects a run's agent exchanges (→ ``raw.exchange``) and emits the COMMIT run-event (→ OL/AGE)."""

    def __init__(self, run_id: str | None = None, *, namespace: str = "aegir") -> None:
        self.run_id = run_id or new_run_id()
        self.namespace = namespace
        self.exchange_ids: list[str] = []
        self.agents: set[tuple[str, str]] = set()   # (provider, model) that GENERATED this run's data

    def record_exchange(self, exchange: dict, *, source_context: dict | None = None) -> "str | None":
        """Append one agent exchange to Iceberg ``raw.exchange``; collect its id + the generating (provider,
        model). The provider/model are FIRST-CLASS provenance — taken from the exchange (the proposer backend:
        ``engine`` | ``grok-build`` | ``xai-api``), NOT hardcoded; so each data element records which agent +
        model produced it."""
        if not exchange or not exchange.get("response"):
            return None
        try:
            from aegir.hx import append_exchange
            from gaius.hx.exchange import ExchangeRecord
            provider = exchange.get("provider") or "engine"
            model = exchange.get("model") or "instruct"
            rec = ExchangeRecord(
                provider=provider,
                request_messages=[{"role": "user", "content": exchange.get("prompt", "")}],
                request_model=model,
                response_content=exchange.get("response", ""),
                response_reasoning=(exchange.get("reasoning") or None),
                source_context={"aegir_module": "agent_refinement", "run_id": self.run_id,
                                "provider": provider, "model": model, **(source_context or {})})
            append_exchange(rec)
            self.exchange_ids.append(rec.id)
            self.agents.add((provider, model))
            return rec.id
        except Exception:  # noqa: BLE001 — lineage is best-effort; never break the loop
            return None

    def emit(self, *, template_id: str, surfaces, gate_verdicts: dict | None = None) -> "dict | None":
        """COMMIT run-event: the refined dual-register surfaces (outputs) derived from the exchanges (inputs),
        gate verdicts (``reGrounding``) + the **generating agents** (provider+model) on the run AND each output
        — so the chapter's provenance directly records WHICH agent/model produced it (queryable on the node,
        not only via the linked exchanges). Idempotent — keyed on ``run_id``."""
        try:
            from aegir.governance import ol
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            verdict = {k: gate_verdicts.get(k) for k in
                       ("placeholder_rate", "disjointness_violations", "ri_ok", "prose_entailment")
                       } if gate_verdicts else {}
            agents = [{"provider": p, "model": m} for p, m in sorted(self.agents)]
            agent_facet = {"generatingAgents": {"_producer": "aegir.refine", "agents": agents}}
            event = {
                "run": {"runId": self.run_id, "facets": dict(agent_facet)},
                "job": {"name": "agent_refinement", "namespace": self.namespace},
                "eventType": "COMPLETE", "eventTime": now,
                "inputs": [{"namespace": "raw", "name": "exchange",
                            "facets": {"refinementExchanges": {"_producer": "aegir.refine",
                                                               "ids": self.exchange_ids[:200]}}}],
                "outputs": [{"namespace": "refined", "name": f"{template_id}.{reg}",
                             "facets": {"reGrounding": {"_producer": "aegir.refine",
                                                        "gateVerdicts": verdict}, **agent_facet}}
                            for reg, _ in (surfaces or [])],
            }
            return ol.ingest_run_event(event)
        except Exception:  # noqa: BLE001
            return None


if __name__ == "__main__":
    from aegir.hx import get_catalog
    cat = get_catalog()
    before = cat.load_table("raw.exchange").scan().to_arrow().num_rows
    rec = LineageRecorder()
    eid = rec.record_exchange(   # a Grok-Build exchange — the provider MUST be recorded as grok-build, not engine
        {"prompt": "Describe the imaging_study tables.", "response": "The imaging_study table tracks studies …",
         "reasoning": "grounded the prose in the cells.", "model": "grok", "provider": "grok-build"},
        source_context={"objective": "fix_prose", "register": "natural"})
    res = rec.emit(template_id="ch0_selftest", surfaces=[("natural", {}), ("semantic", {})],
                   gate_verdicts={"placeholder_rate": 0.0, "disjointness_violations": [], "ri_ok": True,
                                  "prose_entailment": 0.88})
    rows = cat.load_table("raw.exchange").scan().to_arrow().to_pylist()
    after = len(rows)
    row = next((r for r in rows if r.get("id") == eid), {})
    print(f"exchange appended: id={eid} | raw.exchange rows {before} -> {after}")
    print(f"  CAPTURED provider={row.get('provider')!r}  request_model={row.get('request_model')!r}")
    print(f"  generating agents (run): {sorted(rec.agents)}")
    print(f"  OL run-event ingested (run_id={rec.run_id}): {res}")
    assert eid and after == before + 1, "exchange did not append to raw.exchange"
    assert row.get("provider") == "grok-build", f"PROVIDER NOT CAPTURED — got {row.get('provider')!r} (the old hardcoded 'engine' bug)"
    assert row.get("request_model") == "grok", "model not captured"
    assert ("grok-build", "grok") in rec.agents, "generating agent not collected for the run-event facet"
    assert res and res.get("outputs") == 2, "OL run-event did not MERGE 2 output surfaces into AGE"
    print("provenance OK — REAL provider+model in raw.exchange, generatingAgents on the run-event.")

"""Per-step OpenLineage runs — the arc `just metaflow` actually walks, as a graph.

RH 2026-07-26: executing the flow should capture the logical state transformations from FinePDFs
through the discrete metaflow steps — some agent-mediated — into the artifacts we deliver, such that
looking at Atlas or Marquez tells the story.

Why this module exists at all: the flow already emitted plenty, but none of it built a chain.
`emit_event` is OTel (`span.add_event`) — telemetry, no graph. `emit_corpus_lineage` emits ONE run for
the whole flow, so it cannot show an arc through steps by construction. kvasir's own events covered its
DDL work only. So nine of twelve steps contributed nothing to lineage, and a UI had nothing to narrate.

An OL graph is built by dataset IDENTITY: run B's edge to run A exists only when B declares as input
the same (namespace, name) A declared as output. So the whole trick is a stable dataset vocabulary,
declared once, and used by every step.

TWO DECLARATIONS COMPOSE HERE, deliberately:
  * ``DATA_IO`` below — the DATA artifacts (window → passages → entities → ontology → … → release).
  * ``strategy.lineage.STAGE_INPUTS`` — the STRATEGY components a step consumed (prompts, schemas,
    norms) with content SHAs. That declaration already existed for span facets and cache keys; this is
    a further consumer of it, not a parallel copy. One declaration, four consumers.

Every step run is nested under the flow run via the OL ``parent`` facet, so #23's CONSTRUCT-grain
corpus run keeps its identity and gains an expandable arc rather than being replaced.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

# ── the stable dataset vocabulary ────────────────────────────────────────────────────────────────
# Logical identities, NOT per-run paths: the name identifies the dataset across runs (so Atlas
# accumulates history), while the per-run content hash rides a facet. A hash identifies bytes; a
# dataset identity identifies the thing whose versions include those bytes, and lineage needs the
# second — you want "this column derives from that column" to survive regeneration.
DATASETS: "dict[str, tuple[str, str]]" = {
    "window":     ("finepdfs", "window"),            # the upstream input, outside our system
    "passages":   ("aegir", "corpus/passages"),
    "entities":   ("aegir", "corpus/entities"),
    "ontology":   ("aegir", "corpus/ontology"),
    "ddl":        ("aegir", "corpus/ddl"),
    "shapes":     ("aegir", "corpus/shapes"),
    "constructs": ("aegir", "corpus/constructs"),
    "chapters":   ("aegir", "corpus/chapters"),
    "release":    ("aegir", "corpus/release"),
}

# ── the chain: what each step consumes and produces ──────────────────────────────────────────────
# This doubles as the RELEASE MANIFEST (the #72 conservation idea extended): what ships should be
# derived from declared step outputs, so a missing artifact reads as a GAP IN THE GRAPH rather than an
# omission nobody noticed. `sync.py` currently ships from a hand-typed `need = [...]` allowlist, which
# is how "hx excluded" persisted as a decision no one revisited.
DATA_IO: "dict[str, tuple[list[str], list[str]]]" = {
    "harvest":            (["window"], ["passages"]),
    "derive":             (["passages"], ["entities"]),
    "refine_escalations": (["entities"], ["entities"]),
    "realize":            (["entities"], ["ontology", "ddl", "shapes"]),
    "ddl_stage":          (["ontology"], ["ddl"]),
    "build_constructs":   (["ontology"], ["constructs"]),
    "prose_natural":      (["constructs"], ["chapters"]),
    "prose_semantic":     (["constructs"], ["chapters"]),
    "join_verify":        (["entities"], ["entities"]),
    "congruence":         (["chapters"], ["chapters"]),
    "assemble":           (["chapters", "ddl"], ["release"]),
    "project":            (["release"], ["release"]),
}

# Steps whose transformation is performed BY A MODEL. Recorded as a fact about the step, carrying no
# weight-claim: a trace is evidence of what was said, exactly as an OTel span is evidence of what ran.
# Authority lives with the gates (HermiT, kvasir, the membranes), not here.
AGENT_MEDIATED = {"derive", "refine_escalations", "prose_natural", "prose_semantic"}


def _ds(key: str) -> dict:
    ns, name = DATASETS[key]
    return {"namespace": ns, "name": name}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def step_event(step: str, *, flow_run_id: str, state: str,
               facets: "dict | None" = None) -> "dict | None":
    """Build the OL RunEvent for one step, or None when the step declares no data I/O.

    ``state`` is START / COMPLETE / FAIL. The step run id is derived from the flow run so a step's
    START and COMPLETE are the same run, and re-running the same flow run MERGEs rather than
    duplicating (idempotence is the property worth protecting in this whole subsystem).
    """
    if step not in DATA_IO:
        return None
    ins, outs = DATA_IO[step]
    run_facets: dict = {
        # nest under the flow run: #23's corpus-grain identity survives, and gains an arc
        "parent": {"run": {"runId": str(flow_run_id)},
                   "job": {"namespace": "aegir", "name": "SdgCorporaFlow"}},
    }
    if step in AGENT_MEDIATED:
        run_facets["agentMediation"] = {"mediated": True, "capability": "instruct"}
    try:  # the strategy components this stage declared — prompts/schemas/norms, with content SHAs
        from aegir.strategy.lineage import stage_inputs, stage_key
        stage = "prose" if step.startswith("prose") else step
        comps = stage_inputs(stage)
        if comps:
            run_facets["strategyComponents"] = {"stage_key": stage_key(stage),
                                                "components": {c: h[:16] for c, h in comps.items()}}
    except Exception:  # noqa: BLE001 — lineage must never sink a step
        pass
    if facets:
        run_facets.update(facets)
    return {
        "eventType": state,
        "eventTime": _now(),
        "run": {"runId": f"{flow_run_id}/{step}", "facets": run_facets},
        "job": {"namespace": "aegir", "name": f"SdgCorporaFlow.{step}"},
        "inputs": [_ds(k) for k in ins],
        "outputs": [_ds(k) for k in outs],
    }


def emit_step(step: str, *, flow_run_id: str, state: str,
              facets: "dict | None" = None) -> "dict | None":
    """Emit + ingest one step's run event. Never raises: lineage is a record, not a gate."""
    ev = step_event(step, flow_run_id=flow_run_id, state=state, facets=facets)
    if ev is None:
        return None
    try:
        from aegir.governance.ol import ingest_run_event
        return ingest_run_event(ev)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("AEGIR_LINEAGE_DEBUG"):
            print(f"  lineage: {step} {state} not ingested ({type(e).__name__}: {e})", flush=True)
        return None


def chain_check() -> dict:
    """Is the declared chain actually connected? A step whose inputs no prior step produces is a
    BREAK — the graph would show a disconnected island and a UI would have no arc to draw. Cheap
    enough to assert in CI, and it catches a vocabulary typo before a whole run does."""
    produced = {"window"}          # the upstream input is produced outside our system
    breaks, order = [], list(DATA_IO)
    for step in order:
        ins, outs = DATA_IO[step]
        for i in ins:
            if i not in produced:
                breaks.append({"step": step, "unproduced_input": i})
        produced.update(outs)
    unknown = sorted({k for s in DATA_IO.values() for k in (*s[0], *s[1])} - set(DATASETS))
    return {"steps": len(DATA_IO), "breaks": breaks, "unknown_datasets": unknown,
            "connected": not breaks and not unknown}

"""Stage-input declarations — one declaration, three consumers (#148 increment 1).

Each pipeline stage DECLARES which strategy components it consumes. From that single
declaration derive:

1. **OpenLineage-shaped facets** — attached to the stage's OTel span (and, later, emitted as
   OL events proper): input datasets with version facets, Atlas-ingestable.
2. **The stage cache key** — ``hash(sorted input component hashes)``; cache invalidation is
   impact analysis on the lineage graph, never a hand-curated slice table. (Increment 1
   computes and RECORDS these keys; the stages' idempotence checks switch over at increment 2
   — the switch invalidates existing caches once, deliberately, at a window advance.)
3. **Audit** — run-zettels and the lineup render the same refs.

``deriver_version`` was the one-stage special case of this; the declaration generalizes it.
"""
from __future__ import annotations

import hashlib
import json

# stage → the component-path prefixes (or exact paths) it consumes. VERBATIM paths from the
# manifest tree — a change to any matching component is, by declaration, a change to this
# stage's inputs.
STAGE_INPUTS: "dict[str, list[str]]" = {
    "harvest": ["lens/"],
    "derive": ["lens/", "voices/derive.system.md", "voices/derive.schema.json",
               "voices/derive.feedback.py", "voices/derive.dormant_hints.json",
               "voices/mcp_kvasir.tools.json"],
    "realize": ["targets/schemapile_shape_norms.json"],
    "build_constructs": ["targets/schemapile_key_norms.json"],
    "prose": ["voices/register_voice.json", "voices/turn_protocol.md",
              "voices/backend.local.toml", "voices/aegir_writer.md"],
    "join_verify": ["targets/"],
    "congruence": ["lens/"],
}


def stage_inputs(stage: str, manifest: "dict | None" = None) -> "dict[str, str]":
    """{component_path: sha} for a stage's declared inputs, resolved against the manifest."""
    if manifest is None:
        from aegir.strategy.manifest import declared
        manifest = declared()
    if not manifest:
        return {}
    comps = manifest.get("components", {})
    pats = STAGE_INPUTS.get(stage, [])
    return {c: h for c, h in comps.items()
            if any(c == p or (p.endswith("/") and c.startswith(p)) for p in pats)}


def stage_key(stage: str, manifest: "dict | None" = None) -> str:
    """The lineage-entailed cache key for a stage (recorded now; keys switch at increment 2)."""
    inp = stage_inputs(stage, manifest)
    if not inp:
        return ""
    return hashlib.sha256(json.dumps(inp, sort_keys=True).encode()).hexdigest()[:12]


def span_facets(stage: str, manifest: "dict | None" = None) -> "dict[str, str]":
    """OTel span attributes in OpenLineage spirit: inputs + versions + the derived key."""
    if manifest is None:
        from aegir.strategy.manifest import declared
        manifest = declared()
    inp = stage_inputs(stage, manifest)
    out = {"lineage.strategy_id": (manifest or {}).get("strategy_id", ""),
           "lineage.stage_key": stage_key(stage, manifest)}
    for c, h in inp.items():
        out[f"lineage.input.{c}"] = h[:16]
    return out

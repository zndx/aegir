"""inc-2 scaffold operations — RI-safe table manipulations the agent's scaffold agency drives.

The agent PROPOSES structured edits (re-synthesize this column from this concept; retype that column); these
deterministic, RI-safe operations DISPOSE. The membrane-oracle invariant holds and now reaches the table
STRUCTURE (not just prose) — the audit's concept-salad fix — yet the agent never writes cells directly: the
realization stays deterministic + RI-true (keys are never touched, values come from the ontology's domain
pools). Edit schema: ``{"op": "synth_column"|"retype_column", "table": str, "column": str, "concept": str?}``.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

_POOLS_PATH = Path("src/aegir/ontology/entity_value_pools.json")


@functools.lru_cache(maxsize=1)
def _concept_pools() -> dict:
    """{concept → [domain values]} indexed across the shipped entity_value_pools (the ontology's real values)."""
    if not _POOLS_PATH.exists():
        return {}
    from aegir.ontology.subtree_mix import concept_value_index
    return concept_value_index(json.loads(_POOLS_PATH.read_text()))


def _is_key(table: dict, column: dict) -> bool:
    return table.get("pk") == column["name"] or any(fk["col"] == column["name"]
                                                    for fk in table.get("fks", []))


def synth_column(construct: dict, table_name: str, column_name: str, *,
                 concept: str | None = None, pool: list | None = None) -> bool:
    """Re-synthesize a NON-KEY column's cells from a domain pool for ``concept`` (RI-safe — keys untouched;
    values drawn deterministically from the ontology's domain pools, else a concept-derived exemplar)."""
    for t in construct.get("tables", []):
        if t["name"] != table_name:
            continue
        for col in t["columns"]:
            if col["name"] != column_name:
                continue
            if _is_key(t, col):
                return False    # RI: never mutate a PK/FK column
            tgt = concept or col.get("concept") or "value"
            vals = pool if pool is not None else _concept_pools().get(tgt, [])
            col["concept"] = tgt
            for i, cell in enumerate(col["cells"]):
                cell["value"] = str(vals[i % len(vals)]) if vals else f"{tgt}_{i + 1}"
                cell["source"] = tgt
            return True
    return False


def apply_edits(construct: dict, edits: list) -> tuple[list, list]:
    """Apply agent-proposed structured edits in order; returns (applied, skipped). Unknown ops / key columns /
    missing targets are skipped (the membrane refuses them) — the agent proposes, the scaffold disposes."""
    applied: list = []
    skipped: list = []
    for e in edits or []:
        op = (e or {}).get("op")
        ok = False
        if op in ("synth_column", "retype_column"):
            ok = synth_column(construct, e.get("table"), e.get("column"), concept=e.get("concept"))
        (applied if ok else skipped).append(e)
    return applied, skipped

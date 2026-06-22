"""C2.5 L1 — ontology subtree value-mixing (Path A (c) counterfactual augmentation).

A column typed at a PARENT concept draws cells from the UNION of its (SKOS-broader) children's value pools,
so it holds a realistic HETEROGENEOUS subtype population (``specimen`` ← blood ∪ serum ∪ plasma) instead of a
single concept's values. Two effects:
  * **de-leaks further** — the values span subtypes, so no single concept name is recoverable from them; and
  * **sets up the hypernym-CTA** — the generalizable task is "infer the parent type from a heterogeneous
    value population," which requires relational understanding rather than surface pattern-matching.

The SKOS ``broader`` graph IS the HermiT-verified subsumption (it is derived from the consistent ontology at
build time), so the graph itself gates the mixing — children mixed under a parent are genuinely subsumed by
it; no per-table reasoner call is needed. Pure + deterministic (stdlib hashlib seed per column).

Composes with ``eval_rwkv7_relational_probe --hypernym`` (label = ``broader(concept)``) and feeds both the
corpus materialization (``build_ddl_spine``) and the probe's ``build_dataset``.
"""
from __future__ import annotations

import hashlib
import random
import re

_STOP = {"id", "pk", "fk", "code", "num", "no", "key", "ref", "uid", "guid", "name", "label", "value", "val"}


def concept_of(name: str) -> str:
    """Normalize a column name to its concept token (must match eval_rwkv7_relational_probe._concept)."""
    toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t and t not in _STOP and not t.isdigit()]
    return "_".join(toks) or "misc"


def build_children(broader_map: "dict[str, str]") -> "dict[str, list[str]]":
    """{parent_concept → [child_concepts]} — inverse of a {child → parent} SKOS-broader map."""
    children: dict[str, list[str]] = {}
    for child, parent in broader_map.items():
        children.setdefault(parent, []).append(child)
    return children


def concept_value_index(entity_pools: "dict[str, dict[str, list[str]]]") -> "dict[str, list[str]]":
    """{concept → sorted unique values} gathered across every table's column pools."""
    idx: dict[str, set] = {}
    for cols in entity_pools.values():
        for col, vals in cols.items():
            idx.setdefault(concept_of(col), set()).update(vals)
    return {k: sorted(v) for k, v in idx.items()}


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest(), "big")


def subtree_mixed_pools(entity_pools: "dict[str, dict[str, list[str]]]", broader_map: "dict[str, str]",
                        *, cap: int = 40, seed: int = 0xC25) -> "tuple[dict, int]":
    """Return a COPY of ``entity_pools`` where each column whose concept ``C`` has a SKOS-broader parent ``P``
    (with ≥2 children, i.e. ``C`` plus ≥1 sibling) draws from the union of ALL of ``P``'s children's pools —
    "mix the subtrees under their common parent ``P``" (blood ∪ serum ∪ plasma for any specimen-subtype
    column). This is the high-coverage form (keys on the column's PARENT, so every leaf with siblings mixes)
    and it pairs exactly with the hypernym label ``broader(C)=P``: predict ``P`` from its heterogeneous
    subtree. Leaf concepts with no parent / no siblings pass through. Deterministic. Returns
    (mixed_pools, n_columns_mixed)."""
    children = build_children(broader_map)
    cval = concept_value_index(entity_pools)
    mixed: dict[str, dict[str, list[str]]] = {}
    n_mixed = 0
    for table, cols in entity_pools.items():
        mixed[table] = {}
        for col, vals in cols.items():
            parent = broader_map.get(concept_of(col))
            siblings = children.get(parent, []) if parent else []
            if len(siblings) < 2:  # need the column's concept + ≥1 sibling under a common parent
                mixed[table][col] = vals
                continue
            pool = set(vals)
            for sib in siblings:
                pool.update(cval.get(sib, []))
            pool_list = sorted(pool)
            random.Random(_seed(seed, table, col)).shuffle(pool_list)
            mixed[table][col] = pool_list[:cap]
            n_mixed += 1
    return mixed, n_mixed

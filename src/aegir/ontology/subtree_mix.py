"""C2.5 L1 — ontology subtree value-mixing (Path A (c) counterfactual augmentation).

A column typed at a PARENT concept draws cells from the UNION of its (SKOS-broader) children's value pools,
so it holds a realistic HETEROGENEOUS subtype population (``specimen`` ← blood ∪ serum ∪ plasma) instead of a
single concept's values. Two effects:
  * **naturalizes further** — the values span subtypes (a realistic heterogeneous population), so no single
    concept name is recoverable from them; and
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


def subtree_mixed_pools_traced(entity_pools: "dict[str, dict[str, list[str]]]",
                               broader_map: "dict[str, str]", *, cap: int = 40, seed: int = 0xC25
                               ) -> "tuple[dict, int, dict]":
    """As ``subtree_mixed_pools`` but ALSO returns the per-value SOURCE provenance the mix would otherwise
    discard: ``source_map[table][col][value] = source_sibling_concept`` (which child pool the value came from,
    preferring the column's own concept ``C``). The source is **seed-independent** — a value belongs to
    whichever sibling pool contains it — so it can be recovered post-hoc for any materialized cell. This is
    what lets the VALUE_GATE AUDIT the mixing: every mixed cell's source vs its column concept under the
    hypernym-disjointness (within-hypernym mixing → clean; a cross-hypernym source → a bad ``broader`` edge /
    too-high parent, flagged). Returns (mixed_pools, n_columns_mixed, source_map)."""
    children = build_children(broader_map)
    cval = concept_value_index(entity_pools)
    cval_set = {k: set(v) for k, v in cval.items()}
    mixed: dict[str, dict[str, list[str]]] = {}
    source_map: dict[str, dict[str, dict[str, str]]] = {}
    n_mixed = 0
    for table, cols in entity_pools.items():
        mixed[table] = {}
        source_map[table] = {}
        for col, vals in cols.items():
            c = concept_of(col)
            parent = broader_map.get(c)
            siblings = children.get(parent, []) if parent else []
            if len(siblings) < 2:  # need the column's concept + ≥1 sibling under a common parent
                mixed[table][col] = vals
                continue
            own = set(vals)
            pool = set(vals)
            for sib in siblings:
                pool.update(cval.get(sib, []))
            pool_list = sorted(pool)
            random.Random(_seed(seed, table, col)).shuffle(pool_list)
            kept = pool_list[:cap]
            mixed[table][col] = kept
            source_map[table][col] = {
                v: (c if (v in own or v in cval_set.get(c, ()))
                    else next((s for s in siblings if v in cval_set.get(s, ())), c))
                for v in kept}
            n_mixed += 1
    return mixed, n_mixed, source_map


def subtree_mixed_pools(entity_pools: "dict[str, dict[str, list[str]]]", broader_map: "dict[str, str]",
                        *, cap: int = 40, seed: int = 0xC25) -> "tuple[dict, int]":
    """Mix each column's pool with its SKOS-broader siblings' pools (heterogeneous subtype population under the
    common parent ``P``; pairs with the hypernym label ``broader(C)=P``). Leaf concepts with no parent / no
    siblings pass through. Deterministic. Returns (mixed_pools, n_columns_mixed) — see
    ``subtree_mixed_pools_traced`` for the per-value source provenance."""
    mixed, n_mixed, _ = subtree_mixed_pools_traced(entity_pools, broader_map, cap=cap, seed=seed)
    return mixed, n_mixed

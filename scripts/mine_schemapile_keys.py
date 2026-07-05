#!/usr/bin/env python
"""Mine SchemaPile's KEY-SHAPE + SQL-idiom distributions → ``build/schemapile_key_norms.json``.

RH (2026-07-05): chapters feel synthetic when every table's PK is ``id`` — real schemas carry a
DISTRIBUTION of primary-key shapes (bare surrogate ``id``, named surrogate ``course_id``, natural
keys ``code``/``email``, composite keys) and matching FK naming conventions. This instrument
measures them over the full 22,989-database SchemaPile so the realization layer can MATCH the
distributions instead of guessing. Complements ``mine_schemapile_shapes.py`` (width/fk/junction
structure) and ``check_decanning_entropy.py`` (value substance).

Mined axes:
- pk_arity        — 0 (no PK) / 1 / 2 / 3+ column PKs
- pk_naming       — for single-col PKs: bare_id | table_id (<singular>_id/<table>id) | natural
- fk_naming       — FK column style: target_id | role_id (other *_id) | same_as_pk | other
- audit_columns   — rate of created/updated timestamp idioms per table
- common_columns  — the most frequent column names (the "boring reality" vocabulary)
- type_dist       — SQL type families (int/varchar/text/timestamp/decimal/bool/…)
- table_naming    — snake vs prefixed (t_/tbl_) vs camel; plural-ish rate

Run: uv run --no-sync python scripts/mine_schemapile_keys.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path("/raid/datasets/schemapile/schemapile_full.parquet")
OUT = Path("build/schemapile_key_norms.json")

_AUDIT = re.compile(r"^(created?_?(at|on|date|time)?|updated?_?(at|on|date|time)?|modified.*|"
                    r"date_(created|modified|updated)|last_modified.*|deleted_at)$", re.I)
_TYPE_FAMILY = [
    ("int", re.compile(r"int|serial|number\b|long", re.I)),
    ("varchar", re.compile(r"varchar|character varying|nvarchar|char\(", re.I)),
    ("text", re.compile(r"^text|clob|longtext|mediumtext", re.I)),
    ("timestamp", re.compile(r"timestamp|datetime|^date$|^time$", re.I)),
    ("decimal", re.compile(r"decimal|numeric|float|double|real|money", re.I)),
    ("bool", re.compile(r"bool|bit\b|tinyint\(1\)", re.I)),
    ("uuid", re.compile(r"uuid|uniqueidentifier", re.I)),
    ("blob", re.compile(r"blob|binary|bytea|image", re.I)),
]


def _singular(name: str) -> str:
    n = name.lower()
    for suf, rep in (("ies", "y"), ("ses", "s"), ("s", "")):
        if n.endswith(suf) and len(n) > len(suf) + 1:
            return n[: -len(suf)] + rep
    return n


def _pk_name_class(pk: str, table: str) -> str:
    p = pk.lower()
    if p in ("id", "pk", "key"):
        return "bare_id"
    stem = re.sub(r"^(t_|tbl_)", "", table.lower())
    stems = {stem, _singular(stem), stem.replace("_", ""), _singular(stem).replace("_", "")}
    body = re.sub(r"_?(id|no|num|key|code)$", "", p)
    if p.endswith(("_id", "id")) and (body in stems or any(s.endswith(body) or body.endswith(s)
                                                           for s in stems if body)):
        return "table_id"
    if p.endswith("_id") or p.endswith("id"):
        return "other_id"     # role/foreign-flavored surrogate
    return "natural"          # code / email / sku / name / isbn / …


def _fk_name_class(col: str, target_table: str) -> str:
    c = col.lower()
    tgt = _singular(re.sub(r"^(t_|tbl_)", "", (target_table or "").lower()))
    if c == f"{tgt}_id" or c == f"{tgt}id":
        return "target_id"
    if c.endswith("_id") or c.endswith("id"):
        return "role_id"
    if tgt and (c == tgt or c.startswith(tgt)):
        return "target_named"
    return "other"


def main() -> int:
    pk_arity, pk_naming, fk_naming = Counter(), Counter(), Counter()
    type_dist, common_cols, table_naming = Counter(), Counter(), Counter()
    n_tables = n_audit = 0
    f = pq.ParquetFile(SRC)
    for rg in range(f.num_row_groups):
        for _, row in f.read_row_group(rg, columns=["TABLES"]).to_pandas().iterrows():
            for t in row["TABLES"]:
                tname = (t.get("TABLE_NAME") or "").strip()
                cols = list(t.get("COLUMNS") if t.get("COLUMNS") is not None else [])
                if not tname or not cols:
                    continue
                n_tables += 1
                # table naming convention
                if re.match(r"^(t|tbl)_", tname, re.I):
                    table_naming["prefixed"] += 1
                elif "_" in tname or tname.islower():
                    table_naming["snake"] += 1
                else:
                    table_naming["camel_or_pascal"] += 1
                # pks: prefer the table-level list, fall back to IS_PRIMARY
                pk_raw = t.get("PRIMARY_KEYS")
                pks = [str(p) for p in (pk_raw if pk_raw is not None else [])]
                if not pks:
                    pks = [c["NAME"] for c in cols if c.get("IS_PRIMARY")]
                arity = min(len(pks), 3)
                pk_arity[str(arity) if arity < 3 else "3+"] += 1
                if arity == 1:
                    pk_naming[_pk_name_class(pks[0], tname)] += 1
                # fks
                for fk in (t.get("FOREIGN_KEYS") if t.get("FOREIGN_KEYS") is not None else []):
                    try:
                        raw = fk.get("FOREIGN_KEY")
                        if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
                            raw = fk.get("COLUMNS")
                        fcols = [str(c) for c in (raw if raw is not None else [])]
                        tgt = str(fk.get("REFERENCE_TABLE") or "")
                        for c in fcols:
                            fk_naming[_fk_name_class(c, tgt)] += 1
                    except Exception:  # noqa: BLE001 — heterogenous dump formats
                        continue
                # columns: audit idioms, vocabulary, types
                names = [str(c.get("NAME") or "").lower() for c in cols]
                if any(_AUDIT.match(n) for n in names):
                    n_audit += 1
                for c in cols:
                    nm = str(c.get("NAME") or "").lower()
                    if nm:
                        common_cols[nm] += 1
                    ty = str(c.get("TYPE") or "")
                    fam = next((k for k, rx in _TYPE_FAMILY if rx.search(ty)), "other")
                    type_dist[fam] += 1

    def _norm(c: Counter) -> dict:
        s = sum(c.values()) or 1
        return {k: round(v / s, 4) for k, v in c.most_common()}

    out = {
        "source": str(SRC), "n_tables": n_tables,
        "pk_arity": _norm(pk_arity),
        "pk_naming_single": _norm(pk_naming),
        "fk_naming": _norm(fk_naming),
        "audit_column_rate": round(n_audit / max(n_tables, 1), 4),
        "type_dist": _norm(type_dist),
        "table_naming": _norm(table_naming),
        "common_columns_top50": dict(common_cols.most_common(50)),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "common_columns_top50"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

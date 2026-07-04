#!/usr/bin/env python
"""Mine SchemaPile's STRUCTURAL shape distribution → the frequentist parity yardstick (#139).

The de-canning instrument (``check_decanning_entropy.py``) already mines SchemaPile's
SUBSTANCE axes (distinct_ratio, h_colset — value realism). This is the complementary
STRUCTURE-depth+diversity instrument: the empirical distribution of table WIDTH (columns
per table), FK fan-out, PK presence, and tables-per-database, over 22,989 real DB schemas.

It writes ``build/schemapile_shape_norms.json``, whose ``col_count_histogram`` the lineup's
``relational_shape`` (``sources.py``) consumes to compute a discrete 1-Wasserstein (EMD)
distance between OUR realized-spine width distribution and SchemaPile's — turning "parity"
from floor-clearing into a MEASURED distance (RH sanity-check, 2026-07-04). Re-runnable and
idempotent; a standing instrument that also lets ``compose.WIDTH_CYCLE`` be re-fit.

WIDTH CONVENTION (apples-to-apples with the spine): the realized spine counts ``n_cols - 1``
(it excludes the one synthetic surrogate ``id``), so SchemaPile width here excludes one
identity column too — ``n_columns - (1 if the table has any PRIMARY-KEY column else 0)`` —
comparing NON-IDENTITY column counts on both sides. A ``--pk-inclusive`` summary is also
recorded for reference.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_SRC = Path("/raid/datasets/schemapile/schemapile_full.parquet")
OUT = Path("build/schemapile_shape_norms.json")


def _pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def mine(src: Path) -> dict:
    import pyarrow.parquet as pq

    tables_col = pq.read_table(str(src), columns=["TABLES"]).column("TABLES").to_pylist()

    widths: list[int] = []          # non-identity column counts (spine-comparable)
    widths_full: list[int] = []     # all columns (reference)
    fk_out: list[int] = []          # FK edges per table
    tables_per_db: list[int] = []
    n_pk = n_tables = 0

    for db in tables_col:
        if not db:
            continue
        tables_per_db.append(len(db))
        for t in db:
            cols = t.get("COLUMNS") or []
            if not cols:
                continue
            n_tables += 1
            has_pk = any(c.get("IS_PRIMARY") for c in cols)
            n_pk += int(has_pk)
            widths_full.append(len(cols))
            widths.append(max(0, len(cols) - (1 if has_pk else 0)))
            fks = t.get("FOREIGN_KEYS")
            fk_out.append(len(fks) if isinstance(fks, list) else 0)

    widths.sort()
    hist = Counter(widths)
    return {
        "source": str(src),
        "n_databases": len(tables_per_db),
        "n_tables": n_tables,
        "width_convention": "n_columns - 1 identity column (spine-comparable)",
        # the histogram the lineup EMD consumes (col_count → n_tables)
        "col_count_histogram": {str(k): v for k, v in sorted(hist.items())},
        "width": {
            "median": _pct(widths, 0.50),
            "p90": _pct(widths, 0.90),
            "p99": _pct(widths, 0.99),
            "max": widths[-1] if widths else 0,
            "ge5_rate": round(sum(w >= 5 for w in widths) / max(1, len(widths)), 4),
            "wide_rate_ge20": round(sum(w >= 20 for w in widths) / max(1, len(widths)), 4),
        },
        "width_pk_inclusive": {
            "median": _pct(sorted(widths_full), 0.50),
            "p90": _pct(sorted(widths_full), 0.90),
            "p99": _pct(sorted(widths_full), 0.99),
            "max": max(widths_full) if widths_full else 0,
        },
        "fk_out": {
            "mean": round(sum(fk_out) / max(1, len(fk_out)), 3),
            "p90": _pct(sorted(fk_out), 0.90),
            "max": max(fk_out) if fk_out else 0,
            "has_fk_rate": round(sum(f > 0 for f in fk_out) / max(1, len(fk_out)), 4),
        },
        "pk_rate": round(n_pk / max(1, n_tables), 4),
        "tables_per_db": {
            "median": _pct(sorted(tables_per_db), 0.50),
            "p90": _pct(sorted(tables_per_db), 0.90),
            "max": max(tables_per_db) if tables_per_db else 0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine SchemaPile structural shape norms (#139)")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not args.src.exists():
        print(f"SchemaPile parquet not found: {args.src} "
              f"(run scripts/download_schemapile.py)")
        return 1
    norms = mine(args.src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(norms, indent=2) + "\n")
    w = norms["width"]
    print(f"SchemaPile shape norms → {args.out}")
    print(f"  {norms['n_tables']} tables / {norms['n_databases']} DBs")
    print(f"  width (non-identity): median {w['median']} / p90 {w['p90']} / p99 {w['p99']} "
          f"/ max {w['max']} / ge5 {w['ge5_rate']} / wide≥20 {w['wide_rate_ge20']}")
    print(f"  fk_out mean {norms['fk_out']['mean']} / has-fk {norms['fk_out']['has_fk_rate']} "
          f"| pk_rate {norms['pk_rate']} | tables/db median {norms['tables_per_db']['median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

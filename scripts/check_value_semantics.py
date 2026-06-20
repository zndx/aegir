#!/usr/bin/env python
"""Value-semantics scorer (Semantic-Layer-Upkeep Comp 1b) — the second quality dimension of the
embedded-view semantic-quality gate, beside de-canning (column-name diversity, ``check_decanning_entropy``)
and verbalization entropy (``audit_verbalization_entropy``).

Over a DDL-spine run's ``base_rows.parquet`` (one row per materialized cell) it classifies each
*content* cell — NOT pk/fk, which are structurally determined for referential integrity, not semantic
payload — into:

  * ``placeholder`` — synthetic stand-ins of the form ``<TitleCase> NN`` ("Entity 01", "Process 02"),
    emitted by ``rows.py::_entity_value`` when no enum / curated pool applies. Low/zero pretraining value.
  * ``typed``       — xsd-synthetic scalars (numeric / dateTime / date / boolean). Structurally valid but
    domain-thin (a model learns the format, not the domain).
  * ``domain``      — ontology-grounded enum values + curated semantic-pool terms (real words). The
    pretraining-valuable cells; the thing Comp 4's upkeep is trying to raise.

Reports ``placeholder_ratio`` / ``domain_fraction`` / ``typed_fraction`` overall and per (table, column),
plus an integrity probe: time-ordered column pairs where ``start_* > end_*`` (the rows.py bug Comp 4
fixes). Gates on a PROVISIONAL ``--max-placeholder-ratio`` floor (ratchet down as upkeep improves) so the
gate is RED at baseline and motivates the upkeep — mirroring the evidence discipline (#42/#43):
instrument + pre-register, then improve.

PROVISIONAL (per provisional_scaffolding_not_goals): placeholders / typed-thin values are scaffolding to
get an early result over the line, NOT the goal. The north star is domain-real cell values; this scorer
just measures how far we are from it.

    uv run --no-sync python scripts/check_value_semantics.py --spine-run /tmp/ddl_spine_trackA/<run>
    uv run --no-sync python scripts/check_value_semantics.py --spine-run <run> --max-placeholder-ratio 0.40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# rows.py placeholder forms: ``f"{noun} {i+1:02d}"`` / ``f"{name.title()} {n:02d}"`` → TitleCase word(s)
# then a >=2-digit number. (PK form ``PREFIX-0001`` and FK cells are excluded before classification.)
_PLACEHOLDER = re.compile(r"^[A-Za-z][A-Za-z]*( [A-Za-z]+)* +\d{2,}$")
_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}|$)")
_BOOL = {"true", "false"}


def classify(v: str | None) -> str:
    s = (v or "").strip()
    if not s:
        return "empty"
    if _PLACEHOLDER.match(s):
        return "placeholder"
    if _NUMERIC.match(s) or _ISO_DATE.match(s) or s.lower() in _BOOL:
        return "typed"
    return "domain"


def _is_true(x) -> bool:
    return x is True or str(x).lower() == "true"


def score(spine_run: Path) -> dict:
    rows = pq.read_table(spine_run / "base_rows.parquet").to_pylist()
    content = [r for r in rows if not _is_true(r.get("is_pk")) and not _is_true(r.get("is_fk"))]

    overall: dict[str, int] = defaultdict(int)
    per_col: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # collect time-ordered values per (table,row) to probe start_* > end_* ordering
    time_cells: dict[tuple, dict[str, str]] = defaultdict(dict)

    for r in content:
        cls = classify(r.get("value"))
        overall[cls] += 1
        per_col[(r["table_name"], r["col_name"])][cls] += 1
        name = (r.get("col_name") or "").lower()
        if _ISO_DATE.match((r.get("value") or "").strip()) and ("time" in name or "date" in name or "_at" in name):
            time_cells[(r["table_name"], r["row_ix"])][r["col_name"]] = r["value"]

    n = sum(overall.values()) or 1
    placeholder_ratio = overall["placeholder"] / n
    domain_fraction = overall["domain"] / n
    typed_fraction = overall["typed"] / n

    # integrity probe: any (start_*, end_*) pair in the same row where start > end (lexicographic on ISO
    # is chronological). Names matched loosely: start/begin vs end/finish/stop, or *_start vs *_end.
    def _role(c: str) -> str | None:
        c = c.lower()
        if any(k in c for k in ("start", "begin")):
            return "start"
        if any(k in c for k in ("end", "finish", "stop")):
            return "end"
        return None

    violations = 0
    checked_pairs = 0
    for cols in time_cells.values():
        starts = {c: v for c, v in cols.items() if _role(c) == "start"}
        ends = {c: v for c, v in cols.items() if _role(c) == "end"}
        for sv in starts.values():
            for ev in ends.values():
                checked_pairs += 1
                if sv > ev:
                    violations += 1

    col_breakdown = []
    for (tbl, col), counts in sorted(per_col.items()):
        tot = sum(counts.values()) or 1
        col_breakdown.append({
            "table": tbl, "column": col, "n": tot,
            "placeholder": counts["placeholder"] / tot,
            "domain": counts["domain"] / tot,
            "typed": counts["typed"] / tot,
        })

    return {
        "spine_run": str(spine_run),
        "n_content_cells": n,
        "placeholder_ratio": placeholder_ratio,
        "domain_fraction": domain_fraction,
        "typed_fraction": typed_fraction,
        "empty": overall["empty"],
        "time_order_pairs_checked": checked_pairs,
        "time_order_violations": violations,
        "n_placeholder_columns": sum(1 for c in col_breakdown if c["placeholder"] >= 0.99),
        "n_columns": len(col_breakdown),
        "columns": col_breakdown,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine-run", required=True, help="ddl_spine run dir (reads base_rows.parquet)")
    ap.add_argument("--max-placeholder-ratio", type=float, default=0.50,
                    help="PROVISIONAL gate floor: fail if placeholder_ratio exceeds this (ratchet down)")
    ap.add_argument("--min-domain-fraction", type=float, default=0.30,
                    help="PROVISIONAL gate floor: fail if domain_fraction is below this (ratchet up — the north star)")
    ap.add_argument("--out", default=None, help="write value_semantics_report.json here (default: spine-run)")
    ap.add_argument("--top", type=int, default=12, help="how many worst (most-placeholder) columns to print")
    args = ap.parse_args()

    run = Path(args.spine_run)
    if not (run / "base_rows.parquet").exists():
        print(f"no base_rows.parquet in {run} (materialize a spine run with --materialize-rows)", file=sys.stderr)
        return 2

    rep = score(run)
    out = Path(args.out) if args.out else run / "value_semantics_report.json"
    out.write_text(json.dumps(rep, indent=2))

    print(f"value-semantics over {rep['n_content_cells']} content cells (pk/fk excluded) — {run.name}")
    print(f"  placeholder_ratio = {rep['placeholder_ratio']:.3f}   (lower better; floor ≤ {args.max_placeholder_ratio:.2f})")
    print(f"  domain_fraction   = {rep['domain_fraction']:.3f}   (higher better)")
    print(f"  typed_fraction    = {rep['typed_fraction']:.3f}")
    print(f"  fully-placeholder columns: {rep['n_placeholder_columns']}/{rep['n_columns']}")
    print(f"  time-order (start>end) violations: {rep['time_order_violations']}/{rep['time_order_pairs_checked']} pairs")
    worst = sorted(rep["columns"], key=lambda c: -c["placeholder"])[:args.top]
    if worst:
        print(f"  worst columns (placeholder share):")
        for c in worst:
            print(f"    {c['placeholder']:.2f}  {c['table']}.{c['column']}  (n={c['n']})")
    print(f"  → {out}")

    ok = (rep["placeholder_ratio"] <= args.max_placeholder_ratio
          and rep["domain_fraction"] >= args.min_domain_fraction
          and rep["time_order_violations"] == 0)
    print(f"\nVALUE-SEMANTICS GATE: {'PASS ✓' if ok else 'FAIL ✘'} "
          f"(placeholder {rep['placeholder_ratio']:.3f}≤{args.max_placeholder_ratio:.2f}? · "
          f"domain {rep['domain_fraction']:.3f}≥{args.min_domain_fraction:.2f}? · "
          f"{rep['time_order_violations']} time-order violations)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

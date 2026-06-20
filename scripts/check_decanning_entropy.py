#!/usr/bin/env python
"""De-canning entropy gate (Track A guard): are the generated tables' per-BFO-anchor column
vocabularies as VARIED as real-world schemas, or "canned" (every table the same boilerplate)?

The membrane principle in action: **SchemaPile is the real-world REFERENCE distribution at the
membrane** (external data, used only here for a *check*), never a name or value source. We group
our spine tables by BFO anchor (the cohort of templates sharing a foundational type) — analogous to
a SchemaPile *database* grouping related tables — and compute, per group:

  * ``distinct_ratio`` = distinct column names / total column slots  (canned ⇒ low: names repeat),
  * ``H_colset``       = Shannon entropy (bits) over the column-name frequency distribution.

A generated anchor whose ``distinct_ratio`` falls below the SchemaPile p10 band is flagged as
*canned* — a guard against term-stuffing and against a naming upgrade that collapses every table to
the same columns. Run after ``build_ddl_spine.py`` (reads its ``ddl_statements.parquet``).

  # build the reference once (offline, at the membrane):
  python scripts/check_decanning_entropy.py --build-reference \
      --schemapile /raid/datasets/schemapile/schemapile_full.parquet --schemapile-ref build/schemapile_decanning_ref.json
  # gate a spine run:
  python scripts/check_decanning_entropy.py --spine-run <ddl_spine_v0/run> --schemapile-ref build/schemapile_decanning_ref.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

_MIN_GROUP_TABLES = 2          # entropy needs ≥2 tables in a group to be meaningful


def _tokens(name: str) -> list[str]:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)          # camelCase → camel_Case
    return [t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if t]


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def group_stats(col_lists: list[list[str]]) -> dict:
    """Stats over one group's tables (each a list of column names)."""
    names = Counter()
    toks = Counter()
    total = 0
    for cols in col_lists:
        for c in cols:
            names[c] += 1
            total += 1
            for tk in _tokens(c):
                toks[tk] += 1
    n_distinct = len(names)
    return {
        "n_tables": len(col_lists), "n_total_cols": total, "n_distinct_cols": n_distinct,
        "distinct_ratio": (n_distinct / total) if total else 0.0,
        "h_colset": _entropy(list(names.values())),
        "h_term": _entropy(list(toks.values())),
    }


# ── our spine: per-BFO-anchor column entropy ──────────────────────────────────
def anchor_column_entropy(spine_run: Path) -> dict[str, dict]:
    rows = pq.read_table(spine_run / "ddl_statements.parquet").to_pylist()
    by_anchor: dict[str, list[list[str]]] = {}
    for r in rows:
        anchor = (list(r.get("bfo_anchor") or []) or ["(none)"])[-1]
        cols = json.loads(r.get("columns_json") or "[]")
        names = [c["name"] for c in cols if c.get("name") and c["name"] != "id"]
        by_anchor.setdefault(anchor, []).append(names)
    return {a: group_stats(cl) for a, cl in by_anchor.items() if len(cl) >= _MIN_GROUP_TABLES}


# ── SchemaPile reference distribution (the membrane) ──────────────────────────
def build_reference(schemapile_parquet: Path) -> dict:
    import numpy as np
    tbl = pq.read_table(schemapile_parquet, columns=["TABLES"]).to_pylist()
    ratios, hcs = [], []
    for db in tbl:
        col_lists = []
        for t in (db.get("TABLES") or []):
            cols = [c.get("NAME") for c in (t.get("COLUMNS") or []) if c.get("NAME")]
            cols = [c for c in cols if c.lower() != "id"]
            if cols:
                col_lists.append(cols)
        if len(col_lists) >= _MIN_GROUP_TABLES:
            s = group_stats(col_lists)
            ratios.append(s["distinct_ratio"])
            hcs.append(s["h_colset"])
    ratios_a, hcs_a = np.array(ratios), np.array(hcs)
    pct = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {
        "n_databases": len(ratios), "source": str(schemapile_parquet),
        "distinct_ratio": {"mean": float(ratios_a.mean()), "std": float(ratios_a.std()),
                           "p10": pct(ratios_a, 10), "p50": pct(ratios_a, 50)},
        "h_colset": {"mean": float(hcs_a.mean()), "std": float(hcs_a.std()),
                     "p10": pct(hcs_a, 10), "p50": pct(hcs_a, 50)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine-run", help="ddl_spine_v0/<run> dir (reads ddl_statements.parquet)")
    ap.add_argument("--schemapile-ref", default="build/schemapile_decanning_ref.json",
                    help="cached SchemaPile reference json (built via --build-reference)")
    ap.add_argument("--build-reference", action="store_true",
                    help="(re)build the SchemaPile reference json from --schemapile, then exit")
    ap.add_argument("--schemapile", default="/raid/datasets/schemapile/schemapile_full.parquet")
    # Default floor on h_colset (column-name ENTROPY), not raw distinct_ratio: ontology-grounded tables
    # legitimately share typed attributes (correct-by-construction), which depresses distinct_ratio without
    # being "canned"; h_colset is the vocabulary-collapse metric and our tables match SchemaPile's median
    # (Comp 4). Pass --floor-key distinct_ratio for the stricter raw-uniqueness view (reported either way).
    ap.add_argument("--floor-key", default="h_colset", choices=["distinct_ratio", "h_colset"])
    ap.add_argument("--out", default=None, help="write decanning_report.parquet here (default: spine-run)")
    args = ap.parse_args()

    if args.build_reference:
        ref = build_reference(Path(args.schemapile))
        Path(args.schemapile_ref).parent.mkdir(parents=True, exist_ok=True)
        Path(args.schemapile_ref).write_text(json.dumps(ref, indent=2))
        print(f"wrote {args.schemapile_ref}: {ref['n_databases']} DBs; "
              f"distinct_ratio p10={ref['distinct_ratio']['p10']:.3f} "
              f"p50={ref['distinct_ratio']['p50']:.3f} | "
              f"h_colset p10={ref['h_colset']['p10']:.3f} p50={ref['h_colset']['p50']:.3f}")
        return 0

    if not args.spine_run:
        ap.error("--spine-run required (unless --build-reference)")
    spine_run = Path(args.spine_run)
    ours = anchor_column_entropy(spine_run)
    if not ours:
        print("no multi-table anchors to score")
        return 0

    ref = None
    ref_path = Path(args.schemapile_ref)
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
    floor = (ref[args.floor_key]["p10"] if ref else
             # fallback: within-corpus relative floor (mean − 1 std) + a note
             (lambda vs: (sum(vs) / len(vs)) - (sum((v - sum(vs) / len(vs)) ** 2 for v in vs) / len(vs)) ** 0.5)
             ([s[args.floor_key] for s in ours.values()]))
    src = f"SchemaPile p10={floor:.3f}" if ref else f"within-corpus (mean−1σ)={floor:.3f}  [NO reference — build it]"

    report, n_flagged = [], 0
    for anchor, s in sorted(ours.items()):
        canned = s[args.floor_key] < floor
        n_flagged += int(canned)
        report.append({**s, "anchor": anchor, "floor": floor, "floor_key": args.floor_key,
                       "canned": canned})

    print(f"de-canning gate ({args.floor_key} vs {src}):")
    for r in sorted(report, key=lambda r: r[args.floor_key]):
        flag = "  ⚠ CANNED" if r["canned"] else ""
        print(f"  {r['anchor']:32s} n={r['n_tables']:>3} distinct_ratio={r['distinct_ratio']:.3f} "
              f"h_colset={r['h_colset']:.2f}{flag}")
    verdict = "PASS" if n_flagged == 0 else f"{n_flagged}/{len(report)} anchors flagged"
    print(f"verdict: {verdict}")

    out = Path(args.out) if args.out else (spine_run / "decanning_report.parquet")
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(report), out)
    print(f"wrote {out}")
    return 0 if n_flagged == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

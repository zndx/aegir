#!/usr/bin/env python
"""Score the ACTUAL DDL the Rust toolchain generates from the REALIZED ontology (#141).

The authoritative structural lens on the sdg-corpora Data Product (RH ruling, 2026-07-04):
templates are development artifacts; what matters is the *real ontology* a user feeds to
HermiT/DeepOnto (``corpora/ontology/sdg-ontology.omn``) and the *real DDL* a user parses
with datafusion/postgres. This feeds the realized ontology through ``kvasir ddl`` (the
feature-complete Rust toolchain) and measures what actually comes out, quantitatively,
against the empirical SchemaPile distribution (``build/schemapile_shape_norms.json``, #139).

The output is the number that drives ONTOLOGY enhancement: seeing what the DDL *is* — its
width distribution, FK fan-out, junction/lookup counts, and EMD-distance to SchemaPile —
tells us which DataProperties and restrictions the *actual ontology* must grow (not which
templates to tweak) to move the shipped corpus toward structure parity.

  uv run --no-sync python scripts/score_ontology_ddl.py [corpora/ontology/sdg-ontology.omn]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KVASIR = REPO / "components/kvasir/target/release/kvasir"
NORMS = REPO / "build/schemapile_shape_norms.json"


def _pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def _emd(ours: list[int], ref_hist: dict[str, float]) -> float:
    """Discrete 1-Wasserstein between our width list and the SchemaPile histogram — the
    SAME computation the lineup uses, so the two agree."""
    if not ours or not ref_hist:
        return float("nan")
    hi = max(max(ours), max(int(k) for k in ref_hist))
    n = len(ours)
    a = [0.0] * (hi + 1)
    for x in ours:
        a[min(x, hi)] += 1 / n
    tot = sum(ref_hist.values())
    b = [0.0] * (hi + 1)
    for k, v in ref_hist.items():
        b[min(int(k), hi)] += v / tot
    cum = emd = 0.0
    for x, y in zip(a, b):
        cum += x - y
        emd += abs(cum)
    return round(emd, 3)


def naming_concentration(tables: "list[str]") -> dict:
    """Formulaic-naming tells over table names (RH gate before release): leading-token and
    trailing-token concentration + classic-prefix rate, vs SchemaPile norms (prefixed ~.0185).
    A generated corpus whose names all share a stem reads as one author's tic, not a world."""
    from collections import Counter
    lead, trail = Counter(), Counter()
    classic = 0
    for t in tables:
        parts = t.split("_")
        if len(parts) > 1:
            lead[parts[0]] += 1
            trail[parts[-1]] += 1
            if parts[0] in ("tbl", "t", "rel", "tb", "data", "dim", "fact", "stg", "raw"):
                classic += 1
    n = max(1, len(tables))
    top_lead = lead.most_common(5)
    top_trail = trail.most_common(5)
    return {
        "n_tables": len(tables),
        "classic_prefix_rate": round(classic / n, 4),
        "top_leading_tokens": [{"token": k, "share": round(v / n, 4)} for k, v in top_lead],
        "top_trailing_tokens": [{"token": k, "share": round(v / n, 4)} for k, v in top_trail],
        "lead_top1_share": round(top_lead[0][1] / n, 4) if top_lead else 0.0,
    }


def score(omn: Path) -> dict:
    """Run the realized ontology through `kvasir ddl` and profile the generated DDL."""
    p = subprocess.run([str(KVASIR), "ddl", str(omn), "--json"],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"kvasir ddl failed ({p.returncode}): {p.stderr[:300]}")
    plan = json.loads(p.stdout)
    tables = plan.get("tables", [])

    # elected tables = those with >1 column (id + real structure); width = non-identity cols
    # (spine-comparable with the SchemaPile mine convention)
    elected = [t for t in tables if len(t.get("columns", [])) > 1]
    widths = sorted(max(0, len(t["columns"]) - 1) for t in elected)
    n_attr_zero = sum(
        1 for t in elected
        if not any(c.get("prop") and c["name"] not in {f["column"] for f in t.get("fks", [])}
                   for c in t["columns"])
    )
    hist = Counter(widths)

    pk_shapes = Counter()
    for t in tables:
        pks = [c["name"] for c in t["columns"] if c.get("pk")]
        if not pks:
            pk_shapes["none"] += 1
        elif len(pks) > 1:
            pk_shapes["composite"] += 1
        elif pks[0] == "id":
            pk_shapes["bare_id"] += 1
        else:
            pk_shapes["natural_or_named"] += 1
    _npk = sum(pk_shapes.values()) or 1

    norms = json.loads(NORMS.read_text()) if NORMS.exists() else {}
    sp = norms.get("width") or {}
    ref_hist = norms.get("col_count_histogram") or {}

    return {
        "ontology": str(omn),
        "n_tables": len(tables),
        "n_elected": len(elected),
        "n_reference": plan.get("n_reference_tables", 0),
        "n_junctions": len(plan.get("junctions", [])),
        "n_lookups": len(plan.get("lookups", [])),
        "total_fks": sum(len(t.get("fks", [])) for t in tables),
        "sql_valid": plan.get("sql_valid"),
        "width": {
            "median": _pct(widths, 0.50), "p90": _pct(widths, 0.90),
            "p99": _pct(widths, 0.99), "max": widths[-1] if widths else 0,
            "ge5_rate": round(sum(w >= 5 for w in widths) / max(1, len(widths)), 4),
        },
        "attr_zero_ratio": round(n_attr_zero / max(1, len(elected)), 4),
        "col_count_histogram": {str(k): v for k, v in sorted(hist.items())},
        "schemapile": {"median": sp.get("median"), "p90": sp.get("p90"), "p99": sp.get("p99")},
        "pk_shapes": {k: round(v / _npk, 4) for k, v in pk_shapes.most_common()},
        "naming": naming_concentration([t.get("name", "") for t in tables]),
        "shape_emd": _emd(widths, ref_hist),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the realized ontology's actual DDL (#141)")
    ap.add_argument("omn", nargs="?", type=Path,
                    default=REPO / "corpora/ontology/sdg-ontology.omn")
    ap.add_argument("--json", action="store_true", help="emit the full profile as JSON")
    args = ap.parse_args()
    if not KVASIR.exists():
        print(f"kvasir binary missing: {KVASIR} (cargo build --release in components/kvasir)",
              file=sys.stderr)
        return 1
    if not args.omn.exists():
        print(f"ontology not found: {args.omn}", file=sys.stderr)
        return 1
    prof = score(args.omn)
    if args.json:
        print(json.dumps(prof, indent=2))
        return 0
    w, s = prof["width"], prof["schemapile"]
    print(f"REALIZED ONTOLOGY → DDL (the shipped artifact, via kvasir): {prof['ontology']}")
    print(f"  {prof['n_elected']} elected + {prof['n_reference']} reference tables | "
          f"{prof['total_fks']} FKs | {prof['n_junctions']} junctions | "
          f"{prof['n_lookups']} lookups | sql_valid={prof['sql_valid']}")
    print(f"  WIDTH  median {w['median']} / p90 {w['p90']} / p99 {w['p99']} / max {w['max']} "
          f"/ ge5 {w['ge5_rate']}   (attr_zero {prof['attr_zero_ratio']})")
    print(f"  SchemaPile  median {s['median']} / p90 {s['p90']} / p99 {s['p99']}")
    print(f"  SHAPE EMD (DDL ↔ SchemaPile): {prof['shape_emd']}   "
          f"← drive this toward 0 by enhancing the ACTUAL ontology's DataProperties/relations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

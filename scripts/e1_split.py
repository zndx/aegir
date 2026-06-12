#!/usr/bin/env python
"""E1 (EVIDENCE.md): quartile-split the corpus by verifier composite, matched byte budgets.

Joins chapters.parquet + verification.parquet across the given runs, sorts by r_composite,
and packs chapters into 4 slices of ~equal total response bytes (Q1 = lowest scores … Q4 =
highest). Writes per-quartile chapters.parquet (feed to chapters_to_byte_parquet.py) plus
the pre-registered confound report (per-quartile model/family mix, length, score).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs each containing chapters.parquet + verification.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-filter", default=None,
                    help="restrict to chapters from this model (the stratified secondary split)")
    args = ap.parse_args()
    out = Path(args.out)

    chapters, scores = [], {}
    for run in args.runs:
        chapters.extend(pq.read_table(Path(run) / "chapters.parquet").to_pylist())
        for v in pq.read_table(Path(run) / "verification.parquet").to_pylist():
            scores[v["chapter_id"]] = float(v["r_composite"])

    if args.model_filter:
        chapters = [c for c in chapters if c.get("model") == args.model_filter]
    rows = [(scores[c["chapter_id"]], c) for c in chapters if c["chapter_id"] in scores]
    rows.sort(key=lambda x: x[0])
    total_bytes = sum(len(c["response_text"] or "") for _, c in rows)
    target = total_bytes / 4

    quartiles: list[list] = [[], [], [], []]
    qi, acc = 0, 0
    for s, c in rows:
        if acc >= target * (qi + 1) and qi < 3:
            qi += 1
        quartiles[qi].append((s, c))
        acc += len(c["response_text"] or "")

    schema = pq.read_table(Path(args.runs[0]) / "chapters.parquet").schema
    report = {}
    for i, q in enumerate(quartiles, 1):
        qd = out / f"Q{i}"
        qd.mkdir(parents=True, exist_ok=True)
        cs = [c for _, c in q]
        pq.write_table(pa.Table.from_pylist(cs, schema=schema), qd / "chapters.parquet",
                       compression="zstd")
        ss = [s for s, _ in q]
        report[f"Q{i}"] = {
            "n": len(cs),
            "bytes": sum(len(c["response_text"] or "") for c in cs),
            "r_composite_mean": round(sum(ss) / len(ss), 4),
            "r_composite_range": [round(min(ss), 3), round(max(ss), 3)],
            "model_mix": dict(Counter(c["model"] for c in cs)),
            "family_mix": dict(Counter(c["family"] for c in cs).most_common(4)),
            "mean_chars": int(sum(len(c["response_text"] or "") for c in cs) / len(cs)),
        }
    (out / "confound_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

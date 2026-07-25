#!/usr/bin/env python
"""emit_schemapile_postings — permissive-schema entity postings (#45 (iii)).

Term = table/column NAME (verbatim, typically snake_case); doc lineage =
sp:<db-id>:<table>. Columns post to their TABLE's doc, so co-occurrence
queries answer "which permissive schema carries {work_order, status,
quantity}?" — novel-entity discovery for module authoring with the license
boundary STRUCTURAL: rows with PERMISSIVE != true are dropped AT CONSTRUCTION
and counted (the index cannot surface what it never contained).

    uv run python scripts/emit_schemapile_postings.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build/foreign/schemapile/postings"
SRC = "/raid/datasets/schemapile/schemapile_full.parquet"


def main() -> int:
    import pyarrow.parquet as pq
    import zstandard as zstd
    OUT.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(SRC, columns=["ID", "PERMISSIVE", "TABLES"])
    stats = Counter()
    cctx = zstd.ZstdCompressor(level=6)
    with open(OUT / "shard_00.tsv.zst", "wb") as raw, cctx.stream_writer(raw) as w:
        for row in t.to_pylist():
            if row.get("PERMISSIVE") is not True:
                stats["non_permissive_DROPPED"] += 1
                continue
            stats["schemas"] += 1
            for tb in row.get("TABLES") or []:
                if isinstance(tb, str):
                    try:
                        tb = json.loads(tb)
                    except Exception:
                        stats["unparseable_tables"] += 1
                        continue
                name = str(tb.get("TABLE_NAME") or "").strip()
                if not name or len(name) > 64:
                    continue
                lin = f"sp:{row['ID']}:{name}"
                stats["tables"] += 1
                w.write(f"V\t{name}\t{lin}\t1\n".encode("utf-8", "ignore"))
                stats["postings"] += 1
                for col in tb.get("COLUMNS") or []:
                    cn = str(col.get("NAME") or "").strip()
                    if cn and len(cn) <= 64 and "\t" not in cn:
                        w.write(f"V\t{cn}\t{lin}\t1\n".encode("utf-8", "ignore"))
                        stats["postings"] += 1
    (OUT / "TOTALS.json").write_text(json.dumps(dict(stats), indent=1))
    print(f"DONE {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

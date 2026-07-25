#!/usr/bin/env python
"""emit_gittables_postings — the aegir half of the RISE-backed value index (#45).

Scans ALL GitTables parquets and emits the raw postings stream the kvasir/RISE
index builds from: one record per (value, table:column) observation, plus a
column-name entity stream. Shard-resumable (one output shard per input prefix
bucket; completed shards skip), CPU-only, parallel.

TAILORING CAPS (visible, never silent — the no-silent-caps doctrine):
  * values: strings only, 2..64 chars, per-column distinct ≤ 2048 (vocabulary/
    categorical focus — free-text and high-cardinality ID columns drop OUT and
    are COUNTED per shard in the manifest).
  * lineage grain: table-stem:column (the gtvs shape used by sdg:valueProvenance).

Output: build/foreign/gittables/postings/shard_<XX>.tsv.zst
  V\t<value>\t<table>:<column>\t<freq>
  C\t<column-name>\t<table>
plus manifest.json per shard (files, columns, kept/dropped counts).

    uv run python scripts/emit_gittables_postings.py [--workers 10]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/raid/datasets/gittables")
OUT = Path(__file__).resolve().parent.parent / "build/foreign/gittables/postings"
MAX_LEN, MAX_DISTINCT = 64, 2048
SHARDS = 64


def _shard_of(name: str) -> int:
    import hashlib
    return int(hashlib.blake2b(name.encode(), digest_size=2).hexdigest(), 16) % SHARDS


def emit_shard(shard: int) -> dict:
    import zstandard as zstd
    import pyarrow.parquet as pq
    out_p = OUT / f"shard_{shard:02d}.tsv.zst"
    man_p = OUT / f"shard_{shard:02d}.manifest.json"
    if man_p.exists():
        return json.loads(man_p.read_text())
    files = [f for f in ROOT.glob("*.parquet") if _shard_of(f.stem) == shard]
    stats = Counter()
    cctx = zstd.ZstdCompressor(level=6)
    with open(out_p, "wb") as raw, cctx.stream_writer(raw) as w:
        for f in files:
            stats["files"] += 1
            try:
                t = pq.read_table(f)
            except Exception:
                stats["unreadable_files"] += 1
                continue
            for col in t.schema.names:
                stats["columns"] += 1
                w.write(f"C\t{col}\t{f.stem}\n".encode("utf-8", "ignore"))
                try:
                    vals = Counter(str(v).strip() for v in t.column(col).to_pylist()
                                   if isinstance(v, str) and v is not None)
                except Exception:
                    stats["unreadable_columns"] += 1
                    continue
                vals = Counter({v: n for v, n in vals.items()
                                if 2 <= len(v) <= MAX_LEN and "\t" not in v and "\n" not in v})
                if not vals:
                    stats["columns_no_strings"] += 1
                    continue
                if len(vals) > MAX_DISTINCT:
                    stats["columns_over_cardinality_DROPPED"] += 1
                    continue
                lin = f"{f.stem}:{col}"
                for v, n in vals.items():
                    w.write(f"V\t{v}\t{lin}\t{n}\n".encode("utf-8", "ignore"))
                    stats["postings"] += 1
    man = {"shard": shard, **stats}
    man_p.write_text(json.dumps(man, indent=1))
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for man in ex.map(emit_shard, range(SHARDS)):
            for k, v in man.items():
                if k != "shard":
                    done[k] += v
            print(f"shard {man['shard']:02d}: files={man.get('files', 0)} "
                  f"postings={man.get('postings', 0)} "
                  f"dropped-hi-card={man.get('columns_over_cardinality_DROPPED', 0)}", flush=True)
    (OUT / "TOTALS.json").write_text(json.dumps(dict(done), indent=1))
    print(f"TOTALS: {dict(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

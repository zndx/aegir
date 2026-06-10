#!/usr/bin/env python
"""Pull SchemaPile (trl-lab/schemapile) as a first-class STRUCTURED benchmark dataset, alongside
gittables/sotab. The DDL-text form already feeds the pretraining mix (download_corpus_item8); this
fetches the STRUCTURED parquet (each row = a real DB schema: TABLES → COLUMNS with TYPE/NULLABLE/
IS_PRIMARY + PRIMARY_KEYS/FOREIGN_KEYS) so the schema-realism instrument (scripts/schema_complexity.py
--schemapile) can measure the empirical target distribution reproducibly.

Idempotent: skips if the local copy already exists.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = "trl-lab/schemapile"
DEST = Path("/raid/datasets/schemapile")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "schemapile_full.parquet"
    if out.exists() and out.stat().st_size > 0:
        print(f"exists: {out} ({out.stat().st_size / 1e6:.1f} MB) — skipping")
        return 0
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=["data/*.parquet"]))
    src = next(snap.glob("data/*.parquet"))
    shutil.copy2(src, out)
    print(f"SchemaPile structured → {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

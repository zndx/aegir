#!/usr/bin/env python
"""emit_finepdfs_postings — token postings over the NEXT FinePDFs window (#45 (ii)).

The re-stream SCOUT: streams TEXT ONLY (no GPU, no embedding — the cheap half
of a stream pass) for the window AHEAD of the harvest cursor and emits token
postings in the kvasir .kci shard format. `kvasir find … vocab` then answers
per-territory prose-density questions (MBSE terms, logistics terms, DAS terms)
BEFORE any wall-time commitment to a full embedded advance.

Doc lineage grain: fp:<ordinal>:<sha8> (ordinal = absolute stream position, so
scout hits translate directly to cursor ranges). Caps VISIBLE: lowercase ascii
tokens 3..24 chars, per-doc distinct ≤ 4096, docs ≤ --max-chars head.

    uv run python scripts/emit_finepdfs_postings.py --from-cursor auto --n-docs 30000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build/foreign/finepdfs/postings"
TOK = re.compile(r"[a-z][a-z0-9]{2,23}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-cursor", default="auto",
                    help="'auto' = the harvest cursor (scout AHEAD of it), or an int")
    ap.add_argument("--n-docs", type=int, default=30000)
    ap.add_argument("--dataset", default="HuggingFaceFW/finepdfs")
    ap.add_argument("--config", default="eng_Latn")
    ap.add_argument("--max-chars", type=int, default=20000)
    a = ap.parse_args()

    if a.from_cursor == "auto":
        cur = json.loads((REPO / "build/domain_harvest/cursor.json").read_text())["cursor"]
    else:
        cur = int(a.from_cursor)
    OUT.mkdir(parents=True, exist_ok=True)

    import zstandard as zstd
    from datasets import load_dataset
    ds = load_dataset(a.dataset, a.config, split="train", streaming=True)
    it = iter(ds.skip(cur))
    cctx = zstd.ZstdCompressor(level=6)
    stats = Counter()
    shard_docs, shard_id, w, raw = 5000, 0, None, None

    def roll():
        nonlocal shard_id, w, raw
        if w is not None:
            w.close(); raw.close()
        raw = open(OUT / f"shard_{shard_id:02d}.tsv.zst", "wb")
        w = cctx.stream_writer(raw)
        shard_id += 1

    roll()
    for i in range(a.n_docs):
        try:
            doc = next(it)
        except StopIteration:
            stats["stream_ended"] = 1
            break
        if i and i % shard_docs == 0:
            roll()
        text = (doc.get("text") or "")[: a.max_chars]
        h = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:8]
        lin = f"fp:{cur + i}:{h}"
        toks = Counter(TOK.findall(text.lower()))
        stats["docs"] += 1
        if len(toks) > 4096:
            toks = Counter(dict(toks.most_common(4096)))
            stats["docs_token_capped"] += 1
        for t, n in toks.items():
            w.write(f"V\t{t}\t{lin}\t{n}\n".encode("utf-8", "ignore"))
            stats["postings"] += 1
        if (i + 1) % 2000 == 0:
            print(f"  … {i + 1}/{a.n_docs} docs · {stats['postings']:,} postings", flush=True)
    w.close(); raw.close()
    (OUT / "TOTALS.json").write_text(json.dumps(
        {"from_cursor": cur, **stats}, indent=1))
    print(f"DONE from_cursor={cur} {dict(stats)}")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Tokenize one-or-more jsonl corpora with the RWKV World tokenizer → ONE binidx, with per-source
duplication — the augmentation-proportion (α) / upsample lever, in a single tool. Reuses rwkv-lm's
``TRIE_TOKENIZER`` + binidx ``Index`` writer, but controls the vocab + OUTPUT paths *absolutely* so
artifacts land on ``/raid`` (``make_data.py`` hardcodes cwd-basename output → the ``/``-bound rwkv-lm dir).

    uv run --no-sync python scripts/tokenize_corpus_binidx.py \
        --out /raid/build/aegir/path-a/mix/train_a02 \
        --source /raid/datasets/rwkv-world-v3/100k/subsample_100k.jsonl:1 \
        --source /raid/build/aegir/path-a/aug/pathA_aug.jsonl:4 --ctx-len 4096 --seed 42
    # base ×1 + aug ×4 → reports realized α = aug_tokens / total_tokens + the RWKV magic_prime.

Memory note: holds the (duplicated) TEXT lines in RAM and streams tokenization one doc at a time (light on
tokens). Fine to ~1m rows; for the full v3 stream tokenize shards. Round-trips every doc (encode→decode==raw)
and drops any that fail, like make_data. End-of-doc token [0] appended per doc (RWKV convention).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RWKV_V5 = Path("/home/rch/local/src/oss/rwkv-lm/RWKV-v5")
sys.path.insert(0, str(RWKV_V5))
sys.path.insert(0, str(RWKV_V5 / "tokenizer"))
sys.path.insert(0, str(RWKV_V5 / "src"))
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402
from binidx import MMapIndexedDataset  # noqa: E402  (for the .idx writer)

VOCAB = str(RWKV_V5 / "tokenizer" / "rwkv_vocab_v20230424.txt")


def _is_prime(n: int) -> bool:
    if n <= 3:
        return n > 1
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _magic_prime(total_tokens: int, ctx_len: int) -> int:
    """Largest prime p ≡ 2 (mod 3) with p < total_tokens/ctx_len (RWKV data-shuffle convention)."""
    n = (total_tokens // ctx_len) - 1
    while n >= 2 and not (n % 3 == 2 and _is_prime(n)):
        n -= 1
    return n if n >= 2 else -1


class _Builder:
    def __init__(self, prefix: str):
        self._bin = open(prefix + ".bin", "wb")
        self._sizes: list[int] = []
        self._doc_idx: list[int] = [0]

    def add(self, arr: np.ndarray) -> None:
        self._bin.write(arr.tobytes(order="C"))
        self._sizes.append(arr.size)
        self._doc_idx.append(len(self._sizes))

    def finalize(self, prefix: str) -> None:
        self._bin.close()
        with MMapIndexedDataset.Index.writer(prefix + ".idx", np.uint16) as index:
            index.write(self._sizes, self._doc_idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.bin/.idx; put on /raid)")
    ap.add_argument("--source", action="append", required=True, metavar="PATH.jsonl:DUP",
                    help="a jsonl source and its duplication count, e.g. base.jsonl:1 aug.jsonl:4")
    ap.add_argument("--ctx-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-key", default="text")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    tok = TRIE_TOKENIZER(VOCAB)

    # Build the (label, text) work-list with duplication, then shuffle at the document level.
    work: list[tuple[str, str]] = []
    for spec in args.source:
        path, _, dup_s = spec.rpartition(":")
        dup = int(dup_s)
        label = Path(path).stem
        lines = [ln for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        for _ in range(dup):
            work.extend((label, ln) for ln in lines)
    rng.shuffle(work)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    b = _Builder(args.out)
    total = 0
    per_src: Counter = Counter()
    docs = dropped = 0
    for i, (label, ln) in enumerate(work):
        try:
            raw = json.loads(ln).get(args.text_key, "")
        except Exception:  # noqa: BLE001
            raw = ""
        if not raw:
            dropped += 1
            continue
        t = tok.encode(raw)
        if tok.decode(t) != raw:  # non-roundtrippable → drop (matches make_data integrity gate)
            dropped += 1
            continue
        t.append(0)  # end-of-doc
        b.add(np.array(t, dtype=np.uint16))
        total += len(t)
        per_src[label] += len(t)
        docs += 1
        if i % 2000 == 0:
            print(f"  …{i}/{len(work)} docs", flush=True)
    b.finalize(args.out)

    mp = _magic_prime(total, args.ctx_len)
    print(f"\n### {args.out}.bin/idx: {total} tokens, {docs} docs ({dropped} dropped)")
    for label, tk in per_src.most_common():
        print(f"###   {label}: {tk} tokens ({100 * tk / max(1, total):.3f}%)")
    print(f"### magic_prime = {mp} (ctx_len {args.ctx_len})")
    print(f"--my_exit_tokens {total} --magic_prime {mp} --ctx_len {args.ctx_len}")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

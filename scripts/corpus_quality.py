#!/usr/bin/env python
"""Hard, deterministic quality metrics for the synthetic corpus — and the TREND over
generation order, to catch "collapse into noise" (mode collapse, degenerate repetition,
structural decay) early.

All metrics are string / structural / lexical — no GPU, no LLM, fully reproducible:

  Anti-repetition (degeneration):
    - corpus distinct-1/2/3 (unique n-grams / total) — low ⇒ repetitive
    - per-chapter gzip compression ratio — high ⇒ repetitive/low-information
    - per-chapter distinct-3 — flags within-chapter degenerate loops
  Mode-collapse (chapters becoming same-y):
    - TF-IDF inter-chapter cosine: mean pairwise AND mean nearest-neighbour
      (a rising NN-sim = chapters duplicating each other)
    - exact + near-duplicate (NN cosine ≥ 0.95) counts
  Structure (our verifiable edge):
    - JSON-emit rate, tables / columns / cells per chapter
    - template + family coverage (content breadth)
  Stability:
    - response + reasoning length

The TREND bins chapters by generation order and reports each metric per bin: flat ⇒ healthy,
monotone-degrading ⇒ collapse. Run repeatedly as the corpus grows.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

JSON_RE = re.compile(r"```json\s*(.*?)```", re.S)
WORD_RE = re.compile(r"\w+")


def ngrams(toks: list[str], n: int) -> list[tuple]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(toks: list[str], n: int) -> float:
    g = ngrams(toks, n)
    return len(set(g)) / len(g) if g else 0.0


def gzip_ratio(text: str) -> float:
    b = text.encode("utf-8", "ignore")
    if not b:
        return 0.0
    return len(b) / max(len(gzip.compress(b, 6)), 1)


def chapter_cells(text: str) -> tuple[bool, int, int, int]:
    m = JSON_RE.search(text or "")
    if not m:
        return False, 0, 0, 0
    try:
        tables = json.loads(m.group(1)).get("tables", [])
    except Exception:
        return False, 0, 0, 0
    nt = len(tables)
    ncol = ncell = 0
    for tb in tables:
        rows = tb.get("rows") or []
        w = max((len(r) for r in rows if isinstance(r, list)), default=0)
        ncol += w
        ncell += sum(len(r) for r in rows if isinstance(r, list))
    return True, nt, ncol, ncell


def fmt_trend(name: str, vals: list[float], pct: bool = False) -> str:
    cells = " ".join(f"{(v*100 if pct else v):>7.1f}" + ("%" if pct else "") for v in vals)
    # trend arrow: compare last bin to first
    arrow = ""
    if len(vals) >= 2 and vals[0]:
        d = (vals[-1] - vals[0]) / abs(vals[0])
        arrow = "↑" if d > 0.10 else "↓" if d < -0.10 else "→"
    return f"  {name:<26} {cells}   {arrow}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-run", nargs="+", required=True,
                    help="one or more dirs containing chapters.parquet")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    rows = []
    for run in args.corpus_run:
        t = pq.read_table(Path(run) / "chapters.parquet").to_pylist()
        rows.extend(t)
    # generation order
    rows.sort(key=lambda r: str(r.get("created_at")))
    texts = [r.get("response_text") or "" for r in rows]
    n = len(rows)
    print(f"corpus: {n} chapters from {len(args.corpus_run)} run(s)\n")

    # ── per-chapter ───────────────────────────────────────────────────────────
    char_len, word_len, reas_len, gz, d3c, jsonok, cells, cols, tabs = ([] for _ in range(9))
    all_tokens: list[str] = []
    templ, fams = Counter(), Counter()
    for r, txt in zip(rows, texts):
        toks = WORD_RE.findall(txt.lower())
        all_tokens.extend(toks)
        char_len.append(len(txt)); word_len.append(len(toks))
        reas_len.append(len(r.get("response_reasoning") or ""))
        gz.append(gzip_ratio(txt)); d3c.append(distinct_n(toks, 3))
        ok, nt, ncol, ncell = chapter_cells(txt)
        jsonok.append(1.0 if ok else 0.0); tabs.append(nt); cols.append(ncol); cells.append(ncell)
        for tid in (r.get("template_ids") or []):
            templ[tid] += 1
        for f in (r.get("template_families") or []):
            fams[f] += 1

    # ── corpus diversity ──────────────────────────────────────────────────────
    d1 = distinct_n(all_tokens, 1); d2 = distinct_n(all_tokens, 2); d3 = distinct_n(all_tokens, 3)
    exact_dups = n - len(set(texts))

    # ── mode-collapse via TF-IDF cosine ───────────────────────────────────────
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(texts)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)
    mean_pair = float(S[np.triu_indices(n, 1)].mean()) if n > 1 else 0.0
    nn = S.max(axis=1)
    mean_nn = float(nn.mean()) if n > 1 else 0.0
    near_dups = int((nn >= 0.95).sum())

    print("DIVERSITY / ANTI-REPETITION")
    print(f"  distinct-1/2/3 (corpus):   {d1:.3f} / {d2:.3f} / {d3:.3f}   (healthy: d3 > 0.5)")
    print(f"  gzip ratio (mean ± range): {np.mean(gz):.2f}  [{np.min(gz):.2f}, {np.max(gz):.2f}]   (healthy ~3-4; >6 repetitive)")
    print(f"  worst within-chapter distinct-3: {np.min(d3c):.3f}   ({int((np.array(d3c)<0.30).sum())} chapters < 0.30)")
    print("MODE COLLAPSE")
    print(f"  inter-chapter cosine: mean-pair {mean_pair:.3f} | mean-NN {mean_nn:.3f}   (healthy: NN < 0.6)")
    print(f"  exact duplicates: {exact_dups}/{n} | near-dups (NN≥0.95): {near_dups}/{n}")
    print("STRUCTURE (verifiable edge)")
    print(f"  JSON-emit rate: {np.mean(jsonok)*100:.1f}%  | tables/ch {np.mean(tabs):.1f} | cols/ch {np.mean(cols):.1f} | cells/ch {np.mean(cells):.0f}")
    print(f"  template coverage: {len(templ)}/540 | family coverage: {len(fams)}/7")
    print("STABILITY")
    print(f"  response chars (mean): {np.mean(char_len):.0f} | reasoning chars (mean): {np.mean(reas_len):.0f}")

    # ── TREND over generation order ───────────────────────────────────────────
    idx = np.array_split(np.arange(n), args.bins)
    def binned(v):
        a = np.array(v); return [float(a[ix].mean()) if len(ix) else 0.0 for ix in idx]
    # per-bin mean nearest-neighbour similarity WITHIN the bin (collapse localises in time)
    nn_bin = []
    for ix in idx:
        if len(ix) > 1:
            sub = S[np.ix_(ix, ix)].copy(); np.fill_diagonal(sub, 0.0)
            nn_bin.append(float(sub.max(axis=1).mean()))
        else:
            nn_bin.append(0.0)
    print(f"\nTREND over generation order ({args.bins} bins, early→late):")
    print(fmt_trend("distinct-3 (per-ch)", binned(d3c)))
    print(fmt_trend("gzip ratio", binned(gz)))
    print(fmt_trend("response chars", binned(char_len)))
    print(fmt_trend("reasoning chars", binned(reas_len)))
    print(fmt_trend("cells / chapter", binned(cells)))
    print(fmt_trend("JSON-emit rate", binned(jsonok), pct=True))
    print(fmt_trend("intra-bin NN-cosine", nn_bin))
    print("  (distinct-3/len/cells/JSON flat-or-↑ = healthy; NN-cosine rising = collapse)")

    if args.out:
        rep = {"n": n, "distinct_1": d1, "distinct_2": d2, "distinct_3": d3,
               "gzip_mean": float(np.mean(gz)), "mean_pair_cos": mean_pair, "mean_nn_cos": mean_nn,
               "exact_dups": exact_dups, "near_dups": near_dups, "json_rate": float(np.mean(jsonok)),
               "template_coverage": len(templ), "family_coverage": len(fams),
               "trend_nn_bin": nn_bin, "trend_distinct3": binned(d3c), "trend_cells": binned(cells)}
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

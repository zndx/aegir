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


# ── #42: corpus non-repetitive token-yield ceiling (byte-level, the model's token space) ──
# M2 mixes the corpus at fraction α. If the corpus carries only Y_eff genuinely-novel byte-tokens,
# then beyond a budget N you are re-feeding duplicate n-grams, so α_max(N) = Y_eff / N caps α at
# scale. We report two bracketing estimates: an entropy-rate estimate (Y_eff) and the gzip
# information content (Y_gzip); they should agree within ~2× — a big gap flags long-range repetition
# the k-gram window misses. Pre-registered as a MEASUREMENT (the number is the deliverable, no pass/fail).

def _distinct_kgrams(arr_u64, k: int) -> int:
    """Count distinct k-grams over a uint8→uint64 byte array by packing each k-window into one
    uint64 (exact for k ≤ 8: a full 8-byte window is exactly a uint64) and de-duplicating."""
    import numpy as np
    L = arr_u64.size - k + 1
    if L <= 0:
        return 0
    packed = np.zeros(L, dtype=np.uint64)
    base = np.uint64(256)
    for j in range(k):                       # Horner; no overflow for k ≤ 8
        packed = packed * base + arr_u64[j:j + L]
    return int(np.unique(packed).size)


def entropy_rate(data: bytes, max_k: int = 8) -> dict[int, float]:
    """H_k = log2(distinct k-grams)/k for k=1..max_k. As k grows H_k → the per-byte entropy rate
    h (bits/byte ∈ [0,8]); H_{max_k} is our estimate of h."""
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    return {k: (float(np.log2(d)) / k if (d := _distinct_kgrams(arr, k)) > 0 else 0.0)
            for k in range(1, max_k + 1)}


def token_yield_ceiling(data: bytes, max_k: int = 8) -> dict:
    """Non-repetitive byte-token ceiling. Y_eff = total · (H_max_k / 8) (entropy-rate estimate of the
    novel fraction); Y_gzip = compressed length in bytes (information content). Report both."""
    total = len(data)
    H = entropy_rate(data, max_k)
    h8 = H.get(max_k, 0.0)
    return {"total_bytes": total, "H_per_k": H, "h_bits_per_byte": h8,
            "Y_eff": total * (h8 / 8.0), "Y_gzip": len(gzip.compress(data, 6)),
            "distinct_byte_values": int(round(2 ** H.get(1, 0.0)))}


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
    ap.add_argument("--no-byte-yield", action="store_true",
                    help="skip the #42 byte-level non-repetitive token-yield ceiling")
    ap.add_argument("--alpha-budgets", nargs="+", type=float, default=[5e9, 1e10, 5e10],
                    help="candidate M2 mix budgets N (tokens); prints α_max(N)=Y_eff/N for each")
    ap.add_argument("--max-bytes", type=int, default=128_000_000,
                    help="cap on bytes fed to the k-gram entropy estimate (memory guard)")
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

    # ── #42: byte-level non-repetitive token-yield ceiling (caps α at scale for M2) ──────
    ty = None
    if not args.no_byte_yield:
        data = "\n".join(texts).encode("utf-8", "ignore")
        truncated = len(data) > args.max_bytes
        ty = token_yield_ceiling(data[: args.max_bytes])
        ty["truncated_to_max_bytes"] = truncated
        ty["alpha_max"] = {f"{N:.0e}": ty["Y_eff"] / N for N in args.alpha_budgets}
        print("TOKEN-YIELD CEILING (#42 — byte-level, the model's token space)")
        print(f"  total bytes: {ty['total_bytes']:,}{' (TRUNCATED for entropy)' if truncated else ''}"
              f"  | h ≈ {ty['h_bits_per_byte']:.3f} bits/byte")
        print(f"  Y_eff (entropy-rate): {ty['Y_eff']:,.0f} tokens   |   Y_gzip (info content): {ty['Y_gzip']:,.0f} bytes"
              f"   (agree within ~2× ⇒ trustworthy)")
        print("  α_max(N) = Y_eff / N  (M2's ontology fraction must not exceed this, or study repetition):")
        for N in args.alpha_budgets:
            print(f"      N={N:.0e} tokens → α_max = {ty['Y_eff']/N:.5f}")

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
               "trend_nn_bin": nn_bin, "trend_distinct3": binned(d3c), "trend_cells": binned(cells),
               "token_yield": ty}
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

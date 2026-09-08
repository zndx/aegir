#!/usr/bin/env python
"""Corpus naturalness / "mechanical character" norms — a 4-arm comparison harness.

The ontology-grounded generated corpus is *expected* to read mechanically at this stage (its prose is
driven by the axiom structure). This harness establishes **reusable norms** for measuring that mechanical
character so we can track progress as the pipeline is refined. It compares, on the SAME source docs:

  - ``qwen``      — Qwen3.8-27B via the capability engine generated chapters (the production generator)
  - ``grok``      — Grok-4.3 generated chapters (remote)
  - ``glm``       — GLM-4.7 generated chapters (remote)
  - ``finepdfs``  — the raw input FinePDFs documents the chapters were grounded in (the NATURAL ANCHOR)

The generated chapters embed deterministic relational tables; a naive comparison would just rediscover
"generated has tables, FinePDFs doesn't". So we split each text into PROSE (tables/code stripped) and
STRUCTURE, score prose-naturalness on the prose, and report structural density separately. ``finepdfs``
is the anchor: for each prose metric the headline number is the generated arm's DISTANCE from it.

    uv run --no-sync python scripts/compare_corpus_naturalness.py \
        --qwen <run>/chapters --grok <run> --glm <run> --finepdfs build/domain_harvest --n 50 \
        --out docs/scratch/$(date -I)/naturalness_norms

Pure-stdlib + numpy; no model calls (so it's cheap and re-runnable in CI as a norms tracker).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Scaffolding / boilerplate phrases that signal ontology-structure-driven (mechanical) prose. Counted
# per 1k tokens. Curated from observed generated chapters — extend as norms evolve.
_SCAFFOLD = [
    "the following table", "primary key", "foreign key", "referential integrity", "this chapter",
    "is defined as", "subclass of", "ensures uniqueness", "cardinality", "the relational",
    "is a subclass", "represents the", "audit trail", "each instance", "the schema", "constraint",
    "as follows", "in this section", "the axiom", "materializ", "anchored to", "is anchored",
]
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
_SENT = re.compile(r"[.!?]+(?:\s|$)")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$|^\s*\|?[\s:|-]{5,}\|?\s*$")
_FENCE = re.compile(r"```.*?```", re.S)


def _strip_structure(text: str) -> "tuple[str, dict]":
    """Return (prose_text, structure_stats). Removes fenced blocks + markdown table lines."""
    n_fences = len(_FENCE.findall(text))
    no_fence = _FENCE.sub(" ", text)
    lines = no_fence.splitlines()
    table_lines = [ln for ln in lines if _TABLE_LINE.match(ln)]
    prose_lines = [ln for ln in lines if not _TABLE_LINE.match(ln)]
    prose = "\n".join(prose_lines)
    tot = max(1, len(no_fence))
    struct = {
        "table_lines": len(table_lines),
        "code_fences": n_fences,
        "struct_char_frac": round(sum(len(ln) for ln in table_lines) / tot, 4),
    }
    return prose, struct


def _syllables(word: str) -> int:
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _mattr(tokens: "list[str]", window: int = 100) -> float:
    """Moving-average type-token ratio — length-robust lexical diversity."""
    if len(tokens) < window:
        return round(len(set(tokens)) / max(1, len(tokens)), 4)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(0, len(tokens) - window, window // 2)]
    return round(statistics.mean(ratios), 4) if ratios else 0.0


def _distinct_n(tokens: "list[str]", n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return round(len(set(grams)) / max(1, len(grams)), 4)


def _cross_doc_redundancy(docs_tokens: "list[list[str]]", n: int = 5) -> float:
    """Fraction of an arm's n-grams that occur in >1 document — the templating/self-repetition signal.

    High ⇒ the same multi-word scaffolding is reused across chapters (mechanical). Computed over the arm,
    so it is the headline 'is the generator stamping out a template?' metric."""
    df: Counter = Counter()
    for toks in docs_tokens:
        for g in {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}:
            df[g] += 1
    if not df:
        return 0.0
    shared = sum(1 for c in df.values() if c > 1)
    return round(shared / len(df), 4)


def _scaffold_rate(prose: str, tokens: "list[str]") -> float:
    low = prose.lower()
    hits = sum(low.count(p) for p in _SCAFFOLD)
    return round(1000 * hits / max(1, len(tokens)), 2)


def _sentence_cv(prose: str) -> float:
    sents = [s for s in _SENT.split(prose) if s.strip()]
    lens = [len(_WORD.findall(s)) for s in sents if _WORD.findall(s)]
    if len(lens) < 2:
        return 0.0
    m = statistics.mean(lens)
    return round(statistics.pstdev(lens) / m, 4) if m else 0.0


def _flesch(prose: str, tokens: "list[str]") -> float:
    sents = [s for s in _SENT.split(prose) if _WORD.findall(s)]
    if not sents or not tokens:
        return 0.0
    syl = sum(_syllables(w) for w in tokens)
    return round(206.835 - 1.015 * (len(tokens) / len(sents)) - 84.6 * (syl / len(tokens)), 1)


def _unigram_dist(tokens: "list[str]") -> Counter:
    return Counter(t.lower() for t in tokens)


def _jsd(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence (base 2, in [0,1]) between two word distributions — distance-from-anchor."""
    vocab = set(p) | set(q)
    pt, qt = sum(p.values()) or 1, sum(q.values()) or 1
    def kl(a, at, b, bt):
        s = 0.0
        for w in vocab:
            pa = a.get(w, 0) / at
            pm = 0.5 * (pa + b.get(w, 0) / bt)
            if pa > 0 and pm > 0:
                s += pa * math.log2(pa / pm)
        return s
    return round(0.5 * kl(p, pt, q, qt) + 0.5 * kl(q, qt, p, pt), 4)


# ── table utility (tables are an INTENTIONAL feature — score their information content, not their count) ──
_ID_HEADER = re.compile(r"^(id|.*_id|.*_key|key|pk|uid|code|no|num|#)$", re.I)
_CODE_VAL = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{2,}(?:[-_]\d+)*$")   # surrogate code keys: ATT-2023-001, ROLE-0001
_NUM_VAL = re.compile(r"^-?\d+(?:\.\d+)?%?$")
_TRAIL_NUM = re.compile(r"\s*\d+\s*$")


def _parse_tables(text: str) -> "list[tuple[list[str], list[list[str]]]]":
    """Extract markdown pipe-tables → [(headers, data_rows)], dropping the ``---|---`` separator row."""
    blocks, cur = [], []
    for ln in text.splitlines():
        if "|" in ln and _TABLE_LINE.match(ln):
            cur.append(ln)
        else:
            if len(cur) >= 2:
                blocks.append(cur)
            cur = []
    if len(cur) >= 2:
        blocks.append(cur)
    out = []
    for block in blocks:
        rows = []
        for ln in block:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if re.fullmatch(r"[\s:|-]+", "|".join(cells)):     # separator row
                continue
            rows.append(cells)
        if len(rows) >= 2:                                     # header + ≥1 data row
            out.append((rows[0], rows[1:]))
    return out


def _classify_column(header: str, values: "list[str]") -> str:
    """id (surrogate/trivial) · constant · placeholder (low-info list like 'Z Label 01'..NN) · informative."""
    vals = [v for v in values if v != ""]
    if not vals:
        return "empty"
    if _ID_HEADER.match(header.strip()):
        return "id"
    if sum(bool(_CODE_VAL.match(v)) for v in vals) >= 0.8 * len(vals):
        return "id"
    if len(set(vals)) == 1:
        return "constant"
    stems = {_TRAIL_NUM.sub("", v) for v in vals}
    if sum(bool(_TRAIL_NUM.search(v)) for v in vals) >= 0.8 * len(vals) and len(stems) <= max(2, 0.3 * len(vals)):
        return "placeholder"                                   # a shared stem + a running index = trivial list
    return "informative"


def _table_metrics(texts: "list[str]") -> dict:
    cats: Counter = Counter()
    n_tables = total_cols = total_cells = placeholder_cells = numeric_cols = 0
    rich_tokens, rows_per = [], []
    for t in texts:
        for headers, rows in _parse_tables(t):
            n_tables += 1
            rows_per.append(len(rows))
            for ci in range(len(headers)):
                col_vals = [r[ci] for r in rows if ci < len(r)]
                if not col_vals:
                    continue
                total_cols += 1
                total_cells += len(col_vals)
                cat = _classify_column(headers[ci], col_vals)
                cats[cat] += 1
                if cat == "placeholder":
                    placeholder_cells += len(col_vals)
                elif cat == "informative":
                    rich_tokens += [len(_WORD.findall(v)) for v in col_vals]
                if cat != "id" and sum(bool(_NUM_VAL.match(v)) for v in col_vals) >= 0.6 * len(col_vals):
                    numeric_cols += 1
    tc = max(1, total_cols)
    return {
        "n_tables": n_tables,
        "mean_rows_per_table": round(statistics.mean(rows_per), 1) if rows_per else 0,
        "informative_col_rate": round(cats["informative"] / tc, 3),     # ↑ = rich/useful tables
        "trivial_col_rate": round((cats["id"] + cats["constant"] + cats["placeholder"]) / tc, 3),  # ↑ = low-utility
        "placeholder_cell_rate": round(placeholder_cells / max(1, total_cells), 3),  # ↑ = 'Z Label NN' filler
        "numeric_col_rate": round(numeric_cols / tc, 3),                # ↑ = real measurements (preprint-like)
        "mean_informative_cell_tokens": round(statistics.mean(rich_tokens), 2) if rich_tokens else 0,
    }


def _load_chapters(run_dir: str, n: int) -> "list[str]":
    """Load up to n generated chapters — prefer the .md files, fall back to chapters.parquet 'response'."""
    mds = sorted(glob.glob(f"{run_dir}/**/*.md", recursive=True) + glob.glob(f"{run_dir}/*.md"))
    if mds:
        return [Path(m).read_text(encoding="utf-8", errors="ignore") for m in mds[:n]]
    pqs = glob.glob(f"{run_dir}/**/chapters.parquet", recursive=True) + glob.glob(f"{run_dir}/chapters.parquet")
    if pqs:
        import pandas as pd
        df = pd.read_parquet(pqs[0])
        col = "response" if "response" in df.columns else ("chapter_md" if "chapter_md" in df.columns else None)
        if col:
            return [str(x) for x in df[col].dropna().tolist()[:n]]
    return []


def _load_finepdfs(harvest_dir: str, n: int) -> "list[str]":
    docs = sorted(glob.glob(f"{harvest_dir}/docs/*.txt"))
    return [Path(d).read_text(encoding="utf-8", errors="ignore") for d in docs[:n]]


def score_arm(name: str, texts: "list[str]", anchor: "Counter | None") -> dict:
    proses, structs, doc_tokens, dist = [], [], [], Counter()
    for t in texts:
        prose, st = _strip_structure(t)
        proses.append(prose)
        structs.append(st)
        toks = _WORD.findall(prose)
        doc_tokens.append(toks)
        dist.update(_unigram_dist(toks))
    all_tokens = [w for d in doc_tokens for w in d]
    full_prose = "\n".join(proses)
    out = {
        "arm": name, "n_docs": len(texts), "prose_tokens": len(all_tokens),
        "mattr": _mattr(all_tokens),                       # lexical diversity ↑ = more natural
        "distinct_2": _distinct_n(all_tokens, 2),
        "distinct_3": _distinct_n(all_tokens, 3),
        "cross_doc_redundancy_5g": _cross_doc_redundancy(doc_tokens),  # ↑ = more templated (mechanical)
        "scaffold_rate_per_1k": _scaffold_rate(full_prose, all_tokens),  # ↑ = more boilerplate
        "sentence_len_cv": _sentence_cv(full_prose),       # ↑ = more burst/varied (natural)
        "flesch": _flesch(full_prose, all_tokens),
    }
    out.update(_table_metrics(texts))                  # table UTILITY (tables are an intentional feature)
    out["_dist"] = dist
    if anchor is not None:
        out["jsd_vs_finepdfs"] = _jsd(dist, anchor)        # ↓ = vocab closer to natural
        a_vocab, g_vocab = set(anchor), set(dist)
        out["vocab_jaccard_vs_finepdfs"] = round(len(a_vocab & g_vocab) / max(1, len(a_vocab | g_vocab)), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen"); ap.add_argument("--grok"); ap.add_argument("--glm")
    ap.add_argument("--finepdfs", default=str(REPO / "build" / "domain_harvest"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None, help="output stem (writes .json + .md)")
    args = ap.parse_args()

    arms_raw = {}
    if args.finepdfs:
        arms_raw["finepdfs"] = _load_finepdfs(args.finepdfs, args.n)
    for k, d in (("qwen", args.qwen), ("grok", args.grok), ("glm", args.glm)):
        if d:
            arms_raw[k] = _load_chapters(d, args.n)
    arms_raw = {k: v for k, v in arms_raw.items() if v}
    print("loaded arms:", {k: len(v) for k, v in arms_raw.items()})

    anchor = None
    if "finepdfs" in arms_raw:
        adist = Counter()
        for t in arms_raw["finepdfs"]:
            p, _ = _strip_structure(t)
            adist.update(_unigram_dist(_WORD.findall(p)))
        anchor = adist

    rows = [score_arm(name, texts, anchor if name != "finepdfs" else None) for name, texts in arms_raw.items()]
    for r in rows:
        r.pop("_dist", None)

    # PROSE naturalness (finepdfs = natural anchor) and TABLE utility (tables are intentional — score content)
    prose_cols = ["arm", "n_docs", "prose_tokens", "mattr", "distinct_3", "cross_doc_redundancy_5g",
                  "scaffold_rate_per_1k", "sentence_len_cv", "flesch", "jsd_vs_finepdfs", "vocab_jaccard_vs_finepdfs"]
    table_cols = ["arm", "n_tables", "mean_rows_per_table", "informative_col_rate", "trivial_col_rate",
                  "placeholder_cell_rate", "numeric_col_rate", "mean_informative_cell_tokens"]

    def _emit(title, cols):
        print(f"\n=== {title} ===")
        print(" | ".join(f"{c:>24}" for c in cols))
        for r in rows:
            print(" | ".join(f"{str(r.get(c, '-')):>24}" for c in cols))

    _emit("PROSE naturalness (finepdfs = natural anchor)", prose_cols)
    _emit("TABLE utility (tables intentional; ↑informative/numeric/cell_tokens, ↓trivial/placeholder)", table_cols)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
        md = ["# Corpus naturalness + table-utility norms (mechanical-character baseline)\n",
              "`finepdfs` = natural anchor. Tables are an INTENTIONAL feature (text generated in concert with "
              "views for structured-data understanding) — so we score table *utility*, not table count.\n",
              "**Prose** ↑natural: mattr, distinct_3, sentence_len_cv, flesch, vocab_jaccard · "
              "↑mechanical: cross_doc_redundancy_5g, scaffold_rate_per_1k, jsd_vs_finepdfs.",
              "**Tables** ↑useful: informative_col_rate, numeric_col_rate, mean_informative_cell_tokens · "
              "↑trivial: trivial_col_rate (id/constant/placeholder), placeholder_cell_rate.\n",
              "## Prose naturalness", "| " + " | ".join(prose_cols) + " |", "|" + "---|" * len(prose_cols)]
        for r in rows:
            md.append("| " + " | ".join(str(r.get(c, "-")) for c in prose_cols) + " |")
        md += ["\n## Table utility", "| " + " | ".join(table_cols) + " |", "|" + "---|" * len(table_cols)]
        for r in rows:
            md.append("| " + " | ".join(str(r.get(c, "-")) for c in table_cols) + " |")
        out.with_suffix(".md").write_text("\n".join(md) + "\n")
        print(f"\nwrote {out.with_suffix('.json')} + .md")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

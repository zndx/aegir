#!/usr/bin/env python
"""Quantify verbalization diversity (Semantic-Layer-Upkeep Comp 1a — the instrument).

The concern: the catalog's `verbal_template`s are low-entropy, "X is a Y"-dominated, which caps the
pretraining value of the corpus. This makes that diagnosis repeatable + gives the baseline to improve
against. The headline is **syntactic frame diversity**: we collapse content words to a skeleton (so
slot-name surface variation doesn't inflate "uniqueness") which exposes the handful of real syntactic
patterns. We also report the bare-subsumption ("X is a Y") share, the relational share, lexical n-gram
entropy, and per-family flatness.

  uv run --no-sync python scripts/audit_verbalization_entropy.py [--catalog combined.json] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import load_catalog  # noqa: E402

# Structural / function words kept in the skeleton; everything else (content) → "·".
_STRUCT = {"is", "a", "an", "that", "and", "or", "equivalent", "to", "some", "only",
           "not", "of", "with", "the", "which", "by", "in", "for"}
_SLOT_RE = re.compile(r"\{[^}]*\}")
_TOK_RE = re.compile(r"§|[a-z0-9]+")


def skeleton(v: str) -> str:
    """Syntactic frame: slots→§, content words→·, structural words kept; runs of · collapsed.

    'X is a process that has input sample Y' and 'X is an artifact attributed to Y' both → roughly
    '§ is a · that · §' — revealing the shared syntactic frame beneath different predicates."""
    s = _SLOT_RE.sub("§", v.lower())
    out: list[str] = []
    for t in _TOK_RE.findall(s):
        tok = t if (t == "§" or t in _STRUCT) else "·"
        if tok == "·" and out and out[-1] == "·":
            continue
        out.append(tok)
    return " ".join(out)


def _entropy(counts) -> float:
    total = sum(counts)
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0) if total else 0.0


def _ngram_entropy(texts: list[str], n: int) -> float:
    grams: Counter = Counter()
    for v in texts:
        toks = [t for t in _TOK_RE.findall(_SLOT_RE.sub("§", v.lower())) if t != "§"]
        grams.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return _entropy(list(grams.values()))


def compute_report(catalog_path: Path) -> dict:
    """Verbalization-diversity report for a catalog. Importable so the composite semantic-layer gate
    can score this dimension without subprocessing (see scripts/semantic_layer_gate.py)."""
    cat = load_catalog(catalog_path)
    templates = list(cat.templates)
    verbals = [(t, (t.verbal_template or "").strip()) for t in templates]
    have = [(t, v) for t, v in verbals if v]
    n, nv = len(templates), len(have)

    skels = [skeleton(v) for _, v in have]
    skel_freq = Counter(skels)
    no_that = sum(1 for _, v in have if " that " not in f" {v.lower()} ")
    relational = nv - no_that

    fam_stats: dict[str, dict] = {}
    by_fam: dict[str, list[str]] = {}
    for t, v in have:
        fam = (t.bfo_anchor_path[-1] if t.bfo_anchor_path else "?")
        by_fam.setdefault(fam, []).append(skeleton(v))
    for fam, sk in by_fam.items():
        fam_stats[fam] = {"n": len(sk), "distinct_skeletons": len(set(sk)),
                          "skeleton_entropy": round(_entropy(list(Counter(sk).values())), 3)}

    return {
        "n_templates": n, "n_with_verbalization": nv,
        "distinct_skeletons": len(skel_freq),
        "skeleton_diversity": round(len(skel_freq) / nv, 4) if nv else 0.0,
        "skeleton_entropy_bits": round(_entropy(list(skel_freq.values())), 3),
        "top5_skeleton_share": round(sum(c for _, c in skel_freq.most_common(5)) / nv, 4) if nv else 0.0,
        "bare_subsumption_share": round(no_that / nv, 4) if nv else 0.0,    # no restriction clause
        "relational_share": round(relational / nv, 4) if nv else 0.0,
        "unigram_entropy_bits": round(_ngram_entropy([v for _, v in have], 1), 3),
        "bigram_entropy_bits": round(_ngram_entropy([v for _, v in have], 2), 3),
        "len_chars_mean": round(sum(len(v) for _, v in have) / nv, 1) if nv else 0.0,
        "top_skeletons": [{"skeleton": s, "count": c} for s, c in skel_freq.most_common(8)],
        "per_anchor": fam_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = compute_report(Path(args.catalog))
    fam_stats = rep["per_anchor"]
    print(json.dumps({k: v for k, v in rep.items() if k != "top_skeletons" and k != "per_anchor"}, indent=2))
    print("\nTOP SYNTACTIC FRAMES (skeleton · = content word, § = slot):")
    for row in rep["top_skeletons"]:
        print(f"  {row['count']:>4}  {row['skeleton']}")
    print("\nPER-ANCHOR skeleton diversity (entropy bits | distinct/n):")
    for fam, s in sorted(fam_stats.items(), key=lambda kv: kv[1]["skeleton_entropy"]):
        print(f"  {fam:32s} n={s['n']:>3}  H={s['skeleton_entropy']:.2f}  distinct={s['distinct_skeletons']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

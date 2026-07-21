#!/usr/bin/env python
"""calibrate_domain_deriver — instrument the instrument (RH; worklist item 1).

The W1 usage-derivation method predicted 8,197 auto-band rdfs:domains for sdg properties —
against NO ground truth. FIBO ships 775 AUTHORED rdfs:domain declarations alongside the same
kind of restriction usage our method reads, so it is the calibration set: run the method over
FIBO's usage, compare predictions against FIBO's declarations, and price the confidence of
every auto-band verdict before arming.

Method (mirrors derive_property_domains at FIBO granularity): per property, subjects of its
restriction usages lift to their DISCRIMINATING ancestor (the pathways top() rule, 35% cap —
FIBO is not BFO-grounded, so the lift targets its own spine); prediction = dominant lifted
root, banded by the same registered shares (auto ≥ 0.70 / review ≥ 0.40).

Scores per declared-domain property with usage:
  family-match   — declared domain lifts to the SAME discriminating root as the prediction
  subsumption    — declared domain ∈ the ancestor closure of ≥ auto-share of usage subjects
                   (usage is consistent with the authored domain)
REGISTERED FLOOR (pre-measurement): auto-band family-match ≥ 0.80 to trust the sdg auto band.

    uv run python scripts/calibrate_domain_deriver.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

AUTO_SHARE = 0.70
REVIEW_SHARE = 0.40
FAMILY_MATCH_FLOOR = 0.80          # registered pre-measurement


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "focov", REPO / "scripts/foreign_ontology_coverage.py")
    assert spec and spec.loader
    focov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(focov)

    print("sweeping FIBO (usage + declared domains + spine)…", flush=True)
    _ax, _ex, _meta, _mods, structure = focov.sweep(
        Path("~/local/src/oss/fibo").expanduser(), r"^(About|All|Metadata)")
    parents = structure["parents"]
    rest = structure["rest_edges"]                  # (subject, prop-local, filler)
    dr = structure["dr"]                            # prop-iri → [("d", domain), ("r", range)]

    # discriminating-ancestor lift (the pathways top() rule)
    children: "dict[str, set]" = defaultdict(set)
    for c, ps in parents.items():
        for p_ in ps:
            children[p_].add(c)
    desc: "dict[str, int]" = {}

    def ndesc(a: str, seen=frozenset()) -> int:
        if a in desc:
            return desc[a]
        if a in seen:
            return 0
        n_ = len(children.get(a, ())) + sum(ndesc(k, seen | {a}) for k in children.get(a, ()))
        desc[a] = n_
        return n_

    cap = max(5, int(0.35 * max(1, len(parents))))
    memo: "dict[str, str]" = {}

    def top(c: str, seen=frozenset()) -> str:
        if c in memo:
            return memo[c]
        if c in seen:
            return c
        ps = sorted(parents.get(c, ()))
        r = c
        if ps and ndesc(ps[0]) <= cap:
            r = top(ps[0], seen | {c})
        memo[c] = r
        return r

    def closure(c: str, seen=frozenset()) -> "set[str]":
        if c in seen:
            return set()
        out = {c}
        for p_ in parents.get(c, ()):
            out |= closure(p_, seen | {c})
        return out

    # usage per property-LOCAL (rest_edges carry locals; dr carries full IRIs)
    usage: "dict[str, list]" = defaultdict(list)
    for subj, prop_local, _filler in rest:
        usage[prop_local].append(subj)
    declared = {}
    for prop_iri, pairs in dr.items():
        ds = [x for k, x in pairs if k == "d"]
        if ds:
            declared[prop_iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]] = (prop_iri, ds[0])

    rows = []
    band_scores: "dict[str, Counter]" = defaultdict(Counter)
    for plocal, (piri, dom) in sorted(declared.items()):
        subs = usage.get(plocal) or []
        if not subs:
            band_scores["no_usage"]["n"] += 1
            continue
        hist = Counter(top(s_) for s_ in subs)
        pred, n_ = hist.most_common(1)[0]
        share = n_ / len(subs)
        band = ("auto" if share >= AUTO_SHARE else
                "review" if share >= REVIEW_SHARE else "polysemous")
        fam_match = top(dom) == pred
        n_subsumed = sum(1 for s_ in subs if dom in closure(s_))
        subsumed = n_subsumed / len(subs) >= AUTO_SHARE
        band_scores[band]["n"] += 1
        band_scores[band]["family_match"] += int(fam_match)
        band_scores[band]["subsumed"] += int(subsumed)
        rows.append({"property": plocal, "declared_domain": dom.rsplit("/", 1)[-1],
                     "n_usage": len(subs), "predicted_root": pred.rsplit("/", 1)[-1][:40],
                     "share": round(share, 3), "band": band,
                     "family_match": fam_match, "subsumption_ok": subsumed})

    summary = {}
    for band in ("auto", "review", "polysemous"):
        bs = band_scores[band]
        if bs["n"]:
            summary[band] = {"n": bs["n"],
                             "family_match": round(bs["family_match"] / bs["n"], 4),
                             "subsumption": round(bs["subsumed"] / bs["n"], 4)}
    out = {"registered": {"auto_share": AUTO_SHARE, "family_match_floor": FAMILY_MATCH_FLOOR},
           "declared_domains": len(declared),
           "with_usage": sum(1 for r in rows),
           "no_usage": band_scores["no_usage"]["n"],
           "by_band": summary,
           "auto_floor_met": (summary.get("auto", {}).get("family_match", 0)
                              >= FAMILY_MATCH_FLOOR),
           "misses_sample": [r for r in rows
                             if r["band"] == "auto" and not r["family_match"]][:12],
           "rows": rows}
    (REPO / "build/fibo_domain_calibration.json").write_text(json.dumps(out, indent=1))

    print(f"declared domains {len(declared)} · with usage {out['with_usage']} · "
          f"no usage {out['no_usage']}")
    for band, s_ in summary.items():
        print(f"  {band:<10} n={s_['n']:<4} family-match {s_['family_match']:.1%} · "
              f"subsumption {s_['subsumption']:.1%}")
    verdict = "MET" if out["auto_floor_met"] else "NOT MET"
    print(f"auto-band family-match floor ({FAMILY_MATCH_FLOOR:.0%}): {verdict}")
    print("→ build/fibo_domain_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

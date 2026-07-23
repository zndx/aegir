#!/usr/bin/env python
"""measure_admission_selectivity — the aperture's admission profile, baselined (worklist 2).

RH: aperture extensions must not dilute semantic MaxSim selectivity — provable via FinePDFs
with direct inspection. The harvest retains only ADMITTED passages, so v1 baselines the
admission profile over the existing corpus; the DELTA harness (simulating candidate anchors
like PRODML/SysML against the same passages) activates when their vectors exist.

Per admitted passage (head window, the admission grain): MaxSim against every aperture
anchor → best anchor, margin (best − second-best, relative), and entropy of the score
distribution. Outputs:

  * per-anchor admission share (which anchors do the admitting)
  * margin histogram + the LOW-MARGIN set (ambiguous admissions — the passages most
    sensitive to any new anchor; the direct-inspection worklist)
  * anchor-level mean margin (an anchor admitting only low-margin passages is weakly
    differentiated — cross-checked against the df promiscuity watchlist)

REGISTERED (pre-measurement): LOW_MARGIN_REL = 0.05 (best within 5% of runner-up).

    uv run python scripts/measure_admission_selectivity.py [--sample 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DOCS = REPO / "build/domain_harvest/docs"
MANIFEST = REPO / "build/domain_harvest/manifest.jsonl"
LOW_MARGIN_REL = 0.05          # registered pre-measurement
WINDOW_CHARS = 2400            # the admission grain (head window)

# ── PHASE-1 recomposition simulation (RH 2026-07-23) — REGISTERED before measurement ──
RECOMPOSE_TAU = 0.10           # the harvest's admission rule of record (--domain-tau)
RECOMPOSE_WINDOW = 4000        # the harvest's nominal feed; the ColBERT encoder
                               # truncates at 512 tokens — the EFFECTIVE grain (and the
                               # max ITEM size for sliding-window procedures)
LEAF_READMIT_FLOOR = 0.50      # driver expectation, never a gate
ANCHOR_COVERAGE_FLOOR = 3      # "several passages per point" (RH); below it the anchor
                               # emits ADVANCE-THE-INPUT-WINDOW — never a quality demerit
                               # (anchors are real-world aim points; local FinePDFs is tiny)


def maxsim(q: "np.ndarray", d: "np.ndarray") -> float:
    return float((q @ d.T).max(axis=1).mean())


def recompose(sample: int = 0, collection: "str | None" = None,
              out_path: "Path | None" = None) -> int:
    """The leaves-only re-admission simulation over the ALREADY-ADMITTED corpus:
    per doc, ONE encode + one top-10 MaxSim query; the baseline view keeps roots, the
    recomposed view drops them client-side (29 points, 6 roots → top-10 always holds
    ≥4 non-root hits, so both views are exact). Re-admission under the rule of record:
    recomposed rel_margin ≥ RECOMPOSE_TAU. Writes build/admission_recomposition.json."""
    from aegir.ontology import domain_index as DI
    af = DI.armed_admission_filter()
    excl = set(af.get("exclude_codes") or [])
    if not excl:
        print("no admission_filter component — nothing to simulate")
        return 1
    hist = {}
    for line in MANIFEST.read_text().splitlines():
        r = json.loads(line)
        hist[str(r.get("hash", ""))] = str(r.get("code", ""))
    docs = sorted(DOCS.glob("*.txt"))
    if sample:
        docs = docs[:sample]

    def view(hits: "list[dict]") -> dict:
        scores = [h["score"] for h in hits[:5]]
        rel = ((scores[0] - scores[1]) / scores[0]) if len(scores) > 1 and scores[0] else 1.0
        return {"top": hits[0] if hits else None, "rel_margin": rel}

    rows = []
    per_anchor = Counter()
    for i, p in enumerate(docs):
        text = p.read_text(errors="ignore")[:RECOMPOSE_WINDOW]
        hits = DI.classify(text, top_k=10, collection=collection or DI.DEFAULT_APERTURE)
        base = view(hits)
        filt = view([h for h in hits if str(h.get("code", "")) not in excl])
        readmit = bool(filt["top"]) and filt["rel_margin"] >= RECOMPOSE_TAU
        if readmit:
            per_anchor[str(filt["top"].get("code", ""))] += 1
        rows.append({"hash": p.stem, "historical_code": hist.get(p.stem, ""),
                     "base_top": str((base["top"] or {}).get("code", "")),
                     "base_rel_margin": round(base["rel_margin"], 4),
                     "leaf_top": str((filt["top"] or {}).get("code", "")),
                     "leaf_rel_margin": round(filt["rel_margin"], 4),
                     "readmit": readmit})
        if (i + 1) % 100 == 0:
            print(f"  … {i + 1}/{len(docs)}", flush=True)
    n = len(rows)
    n_re = sum(r["readmit"] for r in rows)
    orphans = [r for r in rows if not r["readmit"]]
    base_low = sum(r["base_rel_margin"] < LOW_MARGIN_REL for r in rows)
    leaf_low = sum(r["leaf_rel_margin"] < LOW_MARGIN_REL for r in rows if r["readmit"])
    # coverage vs the registered floor — the ADVANCE-INPUT-WINDOW signal set
    pts, _ = DI._client(DI.DEFAULT_QDRANT_URL).scroll(collection or DI.DEFAULT_APERTURE,
                                                      limit=256, with_payload=True)
    leaf_codes = {str((p_.payload or {}).get("code", "")) for p_ in pts} - excl - {""}
    below = sorted((c, per_anchor.get(c, 0)) for c in leaf_codes
                   if per_anchor.get(c, 0) < ANCHOR_COVERAGE_FLOOR)
    out = {"registered": {"tau": RECOMPOSE_TAU, "window_chars": RECOMPOSE_WINDOW,
                          "colbert_token_limit": 512,
                          "leaf_readmit_floor": LEAF_READMIT_FLOOR,
                          "anchor_coverage_floor": ANCHOR_COVERAGE_FLOOR},
           "n_docs": n, "n_readmit": n_re,
           "readmit_rate": round(n_re / max(1, n), 4),
           "vs_floor": ("MEETS" if n_re / max(1, n) >= LEAF_READMIT_FLOOR else "BELOW")
                       + f" the registered {LEAF_READMIT_FLOOR} floor (driver, not gate)",
           "low_margin": {"baseline": base_low, "recomposed_readmits": leaf_low},
           "per_leaf_projected_share": dict(per_anchor.most_common()),
           "advance_input_window": {
               "rule": "coverage below the floor is a STREAM signal (aim points are "
                       "real-world targets; local FinePDFs is a tiny slice) — never an "
                       "anchor-quality demerit",
               "n_below_floor": len(below), "anchors": below},
           "n_orphans": len(orphans),
           "orphans_by_historical_anchor": dict(Counter(r["historical_code"]
                                                        for r in orphans).most_common()),
           "rows": rows}
    (out_path or REPO / "build/admission_recomposition.json").write_text(
        json.dumps(out, indent=1))
    print(f"\nRECOMPOSITION: {n_re}/{n} re-admit leaves-only ({out['readmit_rate']:.1%}) — "
          f"{out['vs_floor']}")
    print(f"low-margin: baseline {base_low}/{n} → recomposed {leaf_low}/{n_re}")
    print(f"orphans {len(orphans)} (by historical anchor: "
          f"{dict(list(out['orphans_by_historical_anchor'].items())[:5])})")
    print(f"advance-input-window signals: {len(below)} leaf anchors below the "
          f"{ANCHOR_COVERAGE_FLOOR}-passage floor")
    print(f"→ {out_path or 'build/admission_recomposition.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--recompose", action="store_true",
                    help="phase-1 pre-flight: leaves-only re-admission simulation over "
                         "the admitted corpus (registered floors; drivers, not gates)")
    a = ap.parse_args()
    if a.recompose:
        return recompose(sample=0 if a.sample == 200 else a.sample)

    from aegir.ontology import domain_index as DI
    from aegir.ontology.colbert_encoder import get_encoder
    enc = get_encoder()

    client = DI._client(DI.DEFAULT_QDRANT_URL)
    pts, _ = client.scroll(DI.DEFAULT_APERTURE, limit=256, with_payload=True,
                           with_vectors=True)
    anchors = []
    for p in pts:
        v = np.asarray(p.vector, dtype=np.float32)
        frag = str((p.payload or {}).get("iri", p.id)).rsplit("#", 1)[-1]
        anchors.append((frag, v))
    print(f"{len(anchors)} aperture anchors (vectors from qdrant)")

    def fam(frag: str) -> str:
        return frag.split("_", 1)[0]

    docs = sorted(DOCS.glob("*.txt"))[:a.sample]
    shares = Counter()
    margins = []
    anchor_margins: "dict[str, list]" = defaultdict(list)
    confusion = Counter()          # (best, second) pairs on low-margin admissions
    low = []
    n_low_intra = n_low_cross = 0
    for i, f in enumerate(docs):
        text = f.read_text(errors="ignore")[:WINDOW_CHARS]
        q = enc.encode([text])[0]
        scores = sorted(((maxsim(q, v), frag) for frag, v in anchors), reverse=True)
        (s1, a1), (s2, a2) = scores[0], scores[1]
        rel_margin = (s1 - s2) / max(1e-9, s1)
        shares[a1] += 1
        margins.append(rel_margin)
        anchor_margins[a1].append(rel_margin)
        if rel_margin < LOW_MARGIN_REL:
            intra = fam(a1) == fam(a2)
            n_low_intra += intra
            n_low_cross += not intra
            confusion[(a1, a2)] += 1
            low.append({"doc": f.stem[:16], "best": a1, "second": a2,
                        "margin": round(rel_margin, 4),
                        "tie": "intra-family" if intra else "CROSS-FAMILY"})
        if (i + 1) % 50 == 0:
            print(f"  …{i + 1}/{len(docs)}", flush=True)

    m = np.array(margins)
    out = {"registered": {"low_margin_rel": LOW_MARGIN_REL, "window_chars": WINDOW_CHARS},
           "n_passages": len(docs), "n_anchors": len(anchors),
           "margin": {"mean": round(float(m.mean()), 4),
                      "median": round(float(np.median(m)), 4),
                      "p10": round(float(np.percentile(m, 10)), 4),
                      "low_share": round(float((m < LOW_MARGIN_REL).mean()), 4)},
           "low_margin_split": {"intra_family": n_low_intra, "cross_family": n_low_cross,
                                "cross_share_of_low": round(
                                    n_low_cross / max(1, n_low_intra + n_low_cross), 4)},
           "confusion_pairs": [{"best": b, "second": s_, "n": n_}
                               for (b, s_), n_ in confusion.most_common(15)],
           "admission_share": dict(shares.most_common()),
           "anchor_mean_margin": {k: round(float(np.mean(v)), 4)
                                  for k, v in sorted(anchor_margins.items())},
           "low_margin_sample": low[:20]}
    (REPO / "build/admission_selectivity.json").write_text(json.dumps(out, indent=1))
    print(f"margins: median {out['margin']['median']:.3f} · p10 {out['margin']['p10']:.3f} · "
          f"low(<{LOW_MARGIN_REL}) {out['margin']['low_share']:.1%}")
    print(f"low-margin split: intra-family {n_low_intra} · CROSS-FAMILY {n_low_cross} "
          f"({out['low_margin_split']['cross_share_of_low']:.1%} of low)")
    print("top confusion pairs:", [(f"{b}~{s_}", n_) for (b, s_), n_ in confusion.most_common(4)])
    print("top admitting anchors:", dict(shares.most_common(5)))
    weak = sorted(out["anchor_mean_margin"].items(), key=lambda kv: kv[1])[:4]
    print("weakest-margin anchors:", weak)
    print("→ build/admission_selectivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

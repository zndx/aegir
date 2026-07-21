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
LOW_MARGIN_REL = 0.05          # registered pre-measurement
WINDOW_CHARS = 2400            # the admission grain (head window)


def maxsim(q: "np.ndarray", d: "np.ndarray") -> float:
    return float((q @ d.T).max(axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=200)
    a = ap.parse_args()

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

    docs = sorted(DOCS.glob("*.txt"))[:a.sample]
    shares = Counter()
    margins = []
    anchor_margins: "dict[str, list]" = defaultdict(list)
    low = []
    for i, f in enumerate(docs):
        text = f.read_text(errors="ignore")[:WINDOW_CHARS]
        q = enc.encode([text])[0]
        scores = sorted(((maxsim(q, v), frag) for frag, v in anchors), reverse=True)
        (s1, a1), (s2, _a2) = scores[0], scores[1]
        rel_margin = (s1 - s2) / max(1e-9, s1)
        shares[a1] += 1
        margins.append(rel_margin)
        anchor_margins[a1].append(rel_margin)
        if rel_margin < LOW_MARGIN_REL:
            low.append({"doc": f.stem[:16], "best": a1, "margin": round(rel_margin, 4)})
        if (i + 1) % 50 == 0:
            print(f"  …{i + 1}/{len(docs)}", flush=True)

    m = np.array(margins)
    out = {"registered": {"low_margin_rel": LOW_MARGIN_REL, "window_chars": WINDOW_CHARS},
           "n_passages": len(docs), "n_anchors": len(anchors),
           "margin": {"mean": round(float(m.mean()), 4),
                      "median": round(float(np.median(m)), 4),
                      "p10": round(float(np.percentile(m, 10)), 4),
                      "low_share": round(float((m < LOW_MARGIN_REL).mean()), 4)},
           "admission_share": dict(shares.most_common()),
           "anchor_mean_margin": {k: round(float(np.mean(v)), 4)
                                  for k, v in sorted(anchor_margins.items())},
           "low_margin_sample": low[:20]}
    (REPO / "build/admission_selectivity.json").write_text(json.dumps(out, indent=1))
    print(f"margins: median {out['margin']['median']:.3f} · p10 {out['margin']['p10']:.3f} · "
          f"low(<{LOW_MARGIN_REL}) {out['margin']['low_share']:.1%}")
    print("top admitting anchors:", dict(shares.most_common(5)))
    weak = sorted(out["anchor_mean_margin"].items(), key=lambda kv: kv[1])[:4]
    print("weakest-margin anchors:", weak)
    print("→ build/admission_selectivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

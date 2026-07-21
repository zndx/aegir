#!/usr/bin/env python
"""measure_scheme_alignment — the three-layer coincidence, measured (RH 2026-07-21; W3).

The meta-structure's alignment target: OWL rdfs:domain structure ⇄ ConceptScheme organization
⇄ embedding neighborhoods. Two candidate partitions are scored:

  (a) overlay-families — the 6 candidate top-concept families over the aperture's concepts
  (b) upper-anchors    — the notation-root families over the full generated vocabulary

Per partition: EMBEDDING COHERENCE (silhouette over MiniLM vectors of c.text(); the ColBERT
MaxSim space is the upgrade check via --colbert on the aperture subset) and DOMAIN PURITY
(each W1-created property's subject-anchor distribution mapped to families; purity = modal
family share). Registered boundary conditions (drivers, gates on ratification):

  SILHOUETTE_FLOOR = 0.15 · PURITY_FLOOR = 0.60

    uv run python scripts/measure_scheme_alignment.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SILHOUETTE_FLOOR = 0.15    # registered pre-measurement
PURITY_FLOOR = 0.60

# notation-root → family label (build_skos_vocab ANCHORS roots)
FAMILY_OF_ROOT = {"1": "PROCESS", "2": "INDEPENDENT_CONTINUANT", "3": "ICE",
                  "0": "GENERIC", "9": "LIMS"}
# anchor curie → family (for domain purity; specific cco ICE anchors → ICE)
FAMILY_OF_ANCHOR = {"bfo:0000015": "PROCESS", "bfo:0000017": "PROCESS", "bfo:0000023": "PROCESS",
                    "bfo:0000040": "INDEPENDENT_CONTINUANT", "bfo:0000004": "INDEPENDENT_CONTINUANT",
                    "bfo:0000002": "INDEPENDENT_CONTINUANT", "bfo:0000027": "INDEPENDENT_CONTINUANT",
                    "bfo:0000031": "ICE", "cco:ont00000995": "ICE", "cco:ont00000958": "ICE"}


def silhouette(X: "np.ndarray", labels: "list[str]") -> "dict[str, float]":
    fams = sorted(set(labels))
    idx = {f: [i for i, l_ in enumerate(labels) if l_ == f] for f in fams}
    S = X @ X.T                                     # cosine (embed() L2-normalizes)
    D = 1.0 - S
    out: "dict[str, list]" = defaultdict(list)
    for i, l_ in enumerate(labels):
        own = [j for j in idx[l_] if j != i]
        if not own:
            continue
        a = float(D[i, own].mean())
        b = min(float(D[i, idx[f]].mean()) for f in fams if f != l_ and idx[f])
        out[l_].append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return {f: round(float(np.mean(v)), 4) for f, v in out.items() if v}


def main() -> int:
    from aegir.ontology import domain_index as DI
    from aegir.ontology.genus_induction import embed

    report: dict = {"registered": {"silhouette_floor": SILHOUETTE_FLOOR,
                                   "purity_floor": PURITY_FLOOR}, "partitions": {}}

    # ── (a) overlay-families over the aperture concepts ──────────────────────
    ov = {k: c for k, c in DI.load_skos(str(DI.DEFAULT_OVERLAY)).items()
          if not getattr(c, "deprecated", False)}

    def root_of(k, seen=frozenset()):
        c = ov.get(k)
        b = (getattr(c, "broader", "") or "") if c else ""
        return k if not b or b in seen or b not in ov else root_of(b, seen | {k})

    labels_a, texts_a = [], []
    for k, c in sorted(ov.items()):
        labels_a.append(str(root_of(str(k))).rsplit("#", 1)[-1])
        texts_a.append(c.text())
    if len(set(labels_a)) > 1:
        sil_a = silhouette(embed(texts_a), labels_a)
        report["partitions"]["overlay_families"] = {
            "n_concepts": len(texts_a), "n_families": len(set(labels_a)),
            "silhouette": sil_a,
            "global": round(float(np.mean(list(sil_a.values()))), 4),
            "below_floor": sorted(f for f, v in sil_a.items() if v < SILHOUETTE_FLOOR)}

    # ── (b) upper-anchor families over the generated vocab ───────────────────
    vocab = {k: c for k, c in DI.load_skos().items()
             if not getattr(c, "deprecated", False) and getattr(c, "code", "")}
    labels_b, texts_b = [], []
    for k, c in sorted(vocab.items()):
        fam = FAMILY_OF_ROOT.get(str(c.code).split(".")[0])
        if fam and c.text():
            labels_b.append(fam)
            texts_b.append(c.text())
    sil_b = silhouette(embed(texts_b), labels_b)
    report["partitions"]["upper_anchors"] = {
        "n_concepts": len(texts_b), "n_families": len(set(labels_b)),
        "silhouette": sil_b,
        "global": round(float(np.mean(list(sil_b.values()))), 4),
        "below_floor": sorted(f for f, v in sil_b.items() if v < SILHOUETTE_FLOOR)}

    # ── domain purity of the W1-created candidates vs the family map ─────────
    dc = json.loads((REPO / "build/domain_candidates.json").read_text())
    pur = []
    low = []
    for r in dc["properties"]:
        if r["domain"]["verdict"] not in ("auto", "review"):
            continue
        hist = Counter()
        for anchor, n in r["domain"]["hist"].items():
            hist[FAMILY_OF_ANCHOR.get(anchor, "OTHER")] += n
        total = sum(hist.values())
        if not total:
            continue
        p = hist.most_common(1)[0][1] / total
        pur.append(p)
        if p < PURITY_FLOOR:
            low.append({"property": r["property"], "purity": round(p, 3),
                        "families": dict(hist.most_common(3))})
    report["domain_purity"] = {
        "n_properties": len(pur), "mean": round(float(np.mean(pur)), 4) if pur else None,
        "share_above_floor": round(sum(1 for p in pur if p >= PURITY_FLOOR) / max(1, len(pur)), 4),
        "below_floor_sample": low[:12]}

    (REPO / "build/scheme_alignment.json").write_text(json.dumps(report, indent=1))
    for name, part in report["partitions"].items():
        print(f"{name}: {part['n_concepts']} concepts · {part['n_families']} families · "
              f"global silhouette {part['global']} · below-floor {part['below_floor'] or 'none'}")
    dp = report["domain_purity"]
    print(f"domain purity: {dp['n_properties']} properties · mean {dp['mean']} · "
          f"≥{PURITY_FLOOR}: {dp['share_above_floor']:.1%}")
    print("→ build/scheme_alignment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""OQuaRE ontology-quality GATE — the formal publish gate on the ontology Data Product.

OQuaRE (Duque-Ramos et al. 2011) adapts ISO/IEC 25000 (SQuaRE) to ontologies: it organizes metrics into a
hierarchical quality model with SIX characteristics — Structural, Functional Adequacy, Reliability,
Operability, Maintainability, Transferability — each metric normalized to a [1,5] score and aggregated into
a single holistic scorecard (unlike pitfall scanners / raw structural suites). We anchor the per-metric
[1,5] bands to the IOF/BFO signature + the OQuaRE-published scales, FIXED a priori — a stable distance-to-IOF,
not a self-referential corpus quantile (cite Duque-Ramos OQuaRE + Smith IOF). Consistency is consumed from the
upstream HermiT certificate (the reasoner gate is upstream; this gate is JVM-free, pure rdflib via compute()).

GREEN ≥ 3.5 (a competent BFO ontology); AIM 3.9 (published Brick 3.93 / RealEstateCore 3.91 OQuaRE class).
HARD gate: wired into `aegir.lineup.sync._gate()` — refuses `sync --push` of the ontology below 3.5.

    uv run --no-sync python scripts/ontology_oquare.py corpora/ontology/sdg-ontology.owl \
        --certificate corpora/ontology/HERMIT_CERTIFICATE.md [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from ontology_metrology import compute  # noqa: E402

# ── per-metric [1,5] normalization bands (IOF-anchored + OQuaRE-published; FIXED a priori) ──
# Each: sorted (metric_value, [1,5] score) breakpoints → piecewise-linear interpolation, clamped to [1,5].
# Inverted metrics (tangledness: low is good) carry DECREASING scores.
NORM_BANDS = {
    "definitional_completeness": [(0.0, 1), (0.10, 2), (0.25, 3), (0.40, 4), (0.55, 5)],  # IOF 0.55 → 5
    "bfo_grounded":              [(0.50, 1), (0.70, 2), (0.85, 3), (0.95, 4), (1.00, 5)],  # IOF 1.0 → 5
    "realizable_machinery":      [(0, 1), (2, 2), (5, 3), (9, 4), (14, 5)],                # IOF 14+ → 5 (count)
    "def_annotation_coverage":   [(0.0, 1), (0.50, 2), (0.70, 3), (0.90, 4), (1.00, 5)],   # IAO 1.0 req
    "rr":                        [(0.0, 1), (0.10, 2), (0.25, 3), (0.40, 4), (0.50, 5)],   # OntoQA RR
    "ir":                        [(0.0, 1), (0.50, 2), (1.00, 3), (2.00, 4), (3.00, 5)],   # inheritance richness
    "ar":                        [(0.0, 1), (0.10, 2), (0.30, 3), (0.60, 4), (1.00, 5)],   # attribute richness
    "aronto":                    [(0.0, 1), (0.30, 2), (0.60, 3), (1.00, 4), (1.50, 5)],   # axiomatic strength
    "dit":                       [(1, 1), (2, 2), (3, 3), (5, 4), (8, 5)],                 # depth (more developed)
    "tm":                        [(0.0, 5), (0.05, 4), (0.15, 3), (0.30, 2), (0.50, 1)],   # tangledness (INVERTED)
}
# ── the 6 SQuaRE characteristics → their constituent metric scores (auditable + tunable) ──
CHARACTERISTIC_MAP = {
    "Structural":         ["aronto", "dit", "tm", "bfo_grounded", "rr"],
    "FunctionalAdequacy": ["definitional_completeness", "realizable_machinery", "def_annotation_coverage"],
    "Reliability":        ["bfo_grounded", "consistent", "tm"],
    "Operability":        ["def_annotation_coverage", "rr"],
    "Maintainability":    ["tm", "dit", "ir"],
    "Transferability":    ["bfo_grounded", "def_annotation_coverage"],
}
# GREEN requires BOTH the holistic aggregate ≥3.5 AND the Functional-Adequacy characteristic ≥3.0 — the
# latter is where the IOF discriminators live (definitional completeness, realizable machinery, annotations),
# so the gate forces *definitional rigor AND BFO discipline*, not structural/grounding gains alone.
FLOORS = {"aggregate": 3.5, "FunctionalAdequacy": 3.0}
AIM = 3.9


def _norm(value, bands) -> float:
    bands = sorted(bands)
    if value <= bands[0][0]:
        return float(bands[0][1])
    if value >= bands[-1][0]:
        return float(bands[-1][1])
    for (x0, s0), (x1, s1) in zip(bands, bands[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0) if x1 > x0 else 0.0
            return round(s0 + t * (s1 - s0), 2)
    return float(bands[-1][1])


def _check(label, value, op, floor) -> dict:
    ok = {"≥": value >= floor, "≤": value <= floor, "==": value == floor}[op]
    return {"metric": label, "value": value, "op": op, "floor": floor, "pass": bool(ok)}


def _read_consistent(cert_path) -> "bool | None":
    """Consume the upstream HermiT verdict from the realizer's certificate (JVM-free)."""
    p = Path(cert_path) if cert_path else None
    if not p or not p.exists():
        return None
    for ln in p.read_text().splitlines():
        if "isConsistent" in ln:
            return "true" in ln.lower()
    return None


def oquare(onto_path: str, cert_path: "str | None") -> dict:
    m = compute(onto_path)
    consistent = _read_consistent(cert_path)
    scores = {k: _norm(m[k], NORM_BANDS[k]) for k in NORM_BANDS}
    scores["consistent"] = 5.0 if consistent else (1.0 if consistent is False else 3.0)  # unknown → neutral 3
    chars = {c: round(sum(scores[k] for k in ks) / len(ks), 2) for c, ks in CHARACTERISTIC_MAP.items()}
    aggregate = round(sum(chars.values()) / len(chars), 2)
    checks = [
        _check("oquare_aggregate", aggregate, "≥", FLOORS["aggregate"]),
        _check("functional_adequacy", chars["FunctionalAdequacy"], "≥", FLOORS["FunctionalAdequacy"]),
        _check("hermit_consistent", bool(consistent), "==", True),
    ]
    green = all(c["pass"] for c in checks)
    return {
        "ontology": onto_path, "certificate": str(cert_path) if cert_path else None,
        "consistent": consistent, "gate_green": green,
        "aggregate": aggregate, "aim": AIM, "threshold": FLOORS["aggregate"],
        "characteristics": chars, "metric_scores_1to5": scores, "metrics_raw": m,
        "checks": checks, "n_pass": sum(c["pass"] for c in checks), "n_checks": len(checks),
        "benchmark": {"target": AIM, "class": "Brick 3.93 / RealEstateCore 3.91 / Haystack 4.25 (published OQuaRE)"},
        "competency_questions": {"scaffold": True, "items": [
            {"cq_id": "CQ1", "question": "Which roles does entity X bear?", "status": "unimplemented"},
            {"cq_id": "CQ2", "question": "What is the BFO category of class C?", "status": "unimplemented"},
            {"cq_id": "CQ3", "question": "Which classes are defined (≡) vs primitive (→)?", "status": "unimplemented"},
        ]},
        "provenance": "IOF-anchored fixed [1,5] bands (Duque-Ramos OQuaRE + Smith IOF), pre-registered a priori.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="OQuaRE ontology-quality gate")
    ap.add_argument("ontology", nargs="?", default="corpora/ontology/sdg-ontology.owl")
    ap.add_argument("--certificate", default="corpora/ontology/HERMIT_CERTIFICATE.md")
    ap.add_argument("--out", default="build/oquare_score.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rep = oquare(args.ontology, args.certificate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0 if rep["gate_green"] else 1
    print("═══ OQuaRE ONTOLOGY-QUALITY GATE ═══")
    print(f"  ontology: {rep['ontology'].split('/')[-1]}   consistent: {rep['consistent']}")
    print(f"  aggregate: {rep['aggregate']}/5.0   (GREEN ≥ {rep['threshold']}; aim {rep['aim']} = Brick/RealEstateCore)")
    print("  characteristics (1-5):")
    for c, s in rep["characteristics"].items():
        print(f"      {c:20s} {s}")
    print("  metric scores (1-5):")
    for k, s in rep["metric_scores_1to5"].items():
        print(f"      {k:28s} {s}")
    print(f"\n  checks: {rep['n_pass']}/{rep['n_checks']}")
    for c in rep["checks"]:
        print(f"      {'✓' if c['pass'] else '✘'} {c['metric']:20s} {c['value']!s:>6}  {c['op']} {c['floor']}")
    print(f"  → {args.out}")
    print(f"GATE: {'🟢 GREEN — clear to publish the ontology' if rep['gate_green'] else '🔴 RED — refuse publish; raise rigor (Phase A/B), re-gate'}")
    return 0 if rep["gate_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

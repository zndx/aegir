"""cold_start_grounding.py — the cold-start S0→S1 grounding measure (P3's payoff).

Does the content-first derived ontology GROUND in the harvested input? For each cold-start concept:
  S0 = its FinePDFs source passage (source_span, from the un-stripped candidate file)
  S1 = its verbalization (verbal_template)
Both are MaxSim-scored over the cold-start ontology basis (the 223 promoted concepts, a Qdrant MAX_SIM
collection); the transit S0→S1 is the signature correlation net of a shuffled-pairing null. BEATS-NULL ⇒
a concept's source and the concept resolve to the same ontology region = the ontology is grounded in the
input (content-first derivation worked) — the contrast to v0.3's S0→S1 AT-NULL (topic-first sampling let
the ontology float free of the input). Late-interaction MaxSim throughout (the directive).

  LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs CUDA_VISIBLE_DEVICES=4 \
    uv run --no-sync python scripts/cold_start_grounding.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from comprehension_metric import build_basis, maxsim_matrix, normalize  # noqa: E402
from e6a_trace import boot_ci, null_transfer, row_cos  # noqa: E402
from aegir.ontology import domain_index as DI  # noqa: E402

CAT = REPO / "src/aegir/ontology/catalog"


def _source_span(raw: dict) -> "str | None":
    """The source passage lives in an un-stripped staging field; try the known locations."""
    prov = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    for v in (raw.get("_source_span"), raw.get("source_span"), prov.get("source_span"),
              prov.get("_source_span"), prov.get("quote"), prov.get("source")):
        if isinstance(v, str) and v.strip():
            return v
    return None


def main() -> None:
    rng = np.random.default_rng(20260701)
    promoted = {t["template_id"] for t in json.load(open(CAT / "08_derived.json"))["templates"]}
    cands = json.load(open(CAT / "08_derived.candidate.json"))["templates"]

    rows = []
    for c in cands:
        if c["template_id"] not in promoted:
            continue
        vt = (c.get("verbal_template") or "").strip()
        src = _source_span(c)
        if vt and src and src.strip():
            rows.append((c["template_id"], src.strip(), vt))
    print(f"cold-start concepts with source+verbal: {len(rows)} / {len(promoted)} promoted")
    if len(rows) < 20:
        print("  !! too few source spans found — inspect candidate provenance format"); import os; os._exit(0)

    labels = [r[0] for r in rows]
    n = len(rows)
    build_basis([r[2] for r in rows], DI.DEFAULT_QDRANT_URL)   # basis = the concept verbalizations
    print("scoring S0 (source passages) and S1 (verbalizations) by MaxSim…")
    S0 = normalize(maxsim_matrix([r[1] for r in rows], n, DI.DEFAULT_QDRANT_URL))
    S1 = normalize(maxsim_matrix([r[2] for r in rows], n, DI.DEFAULT_QDRANT_URL))

    obs = row_cos(S0, S1)
    nul = null_transfer(S0, S1, rng)
    o, _, _ = boot_ci(obs, rng)
    d_m, d_lo, d_hi = boot_ci(obs - nul, rng)
    green = d_lo > 0
    print(f"\n=== COLD-START S0→S1 GROUNDING (MaxSim, n={n}) ===")
    print(f"  obs={o:.3f}   Δ-vs-null={d_m:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]   "
          f"{'🟢 BEATS-NULL — input-grounded' if green else '🔴 at-null'}")
    print(f"  contrast: v0.3 S0→S1 was AT-NULL (topic-first) → content-first "
          f"{'GROUNDS the ontology in the input by construction' if green else 'did NOT lift grounding'}")

    per = obs - nul
    order = np.argsort(-per)
    print("\n  best-grounded concepts:", [labels[i] for i in order[:6]])
    print("  weakest-grounded      :", [labels[i] for i in order[-6:]])
    import os
    os._exit(0)


if __name__ == "__main__":
    main()

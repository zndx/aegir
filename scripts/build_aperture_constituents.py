#!/usr/bin/env python
"""build_aperture_constituents — make the Canonical Aperture's constituent mapping EXPLICIT (#31 legs 1+2).

The aperture's 29 anchors are textually composite (rich authored domain concepts) but the
domain→constituent mapping was implicit. This derives it by the aperture's OWN semantics — each
anchor's ColBERT multivector queries the VOCAB collection (MaxSim, the same procedure that admits
items) — yielding the M:N LATTICE by construction: a concept may clear the threshold under several
anchors (RH 2026-07-19: domain⇄concept is many-to-many). Results are written three ways:

  * qdrant: each aperture point's payload gains ``constituents`` (code · label · score)
  * artifact: build/aperture_constituents.json (the full lattice, both directions)
  * the strategy lens snapshot picks the payloads up on the next ``manifest seed``

``--verify`` runs the SUFFICIENCY-OF-DIFFERENTIATION check (leg 2): within each domain, the primary
constituents must be mutually differentiated — measured with the SAME margin machinery as the taxonomy
(genus_induction.embed → pairwise cosine); a pair above the pre-registered ceiling is flagged as an
insufficient-differentia candidate for that domain specification. Re-run under any domain refinement
(in situ — novel physical compute environments included). → build/aperture_sufficiency.json

    uv run python scripts/build_aperture_constituents.py --top-k 40 --tau 0.55 --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import domain_index as DI  # noqa: E402

# pre-registered (before first measurement): intra-domain constituent pairs with cosine above this
# are insufficient-differentia candidates under that domain specification.
PAIR_COSINE_CEILING = 0.90


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DI.DEFAULT_QDRANT_URL)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="per-anchor RELATIVE threshold: primary iff score >= alpha * anchor's best "
                         "(MaxSim sums are unnormalized — an absolute tau is dimensionless here)")
    ap.add_argument("--verify", action="store_true", help="run the per-domain sufficiency check")
    ap.add_argument("--write-payloads", action="store_true", default=True)
    a = ap.parse_args()

    anchors = {k: c for k, c in DI.load_skos(str(DI.DEFAULT_OVERLAY)).items()
               if not getattr(c, "deprecated", False)}   # bridges are resolution aids, never anchors
    # the ENUMERATED include points (FINTECH/BIOTECH/MBSE …) live in integration
    # files — outside load_skos's regex; without them the scratch chord drops
    # live aperture points (measured 2026-07-25: 30/33 rendered)
    _inc = DI.integration_overlay_concepts()
    for m in DI._sot_admission_filter().get("aperture_include", {}).get("members", []):
        iri = str(m.get("iri", ""))
        if iri in _inc and iri not in anchors:
            anchors[iri] = _inc[iri]
    vocab = DI.load_skos()                                   # full vocab (defaults)
    client = DI._client(a.url)
    from aegir.ontology.colbert_encoder import get_encoder
    enc = get_encoder()

    lattice: "dict[str, list[dict]]" = {}                    # anchor label → constituents
    concept_domains: "dict[str, list[str]]" = defaultdict(list)  # concept code → anchor labels
    anchor_points = {p.payload.get("pref_label"): p.id for p in
                     client.scroll(DI.DEFAULT_APERTURE, limit=256, with_payload=True)[0]}
    anchor_labels = {c.pref_label for c in anchors.values()}
    for lbl, c in sorted(anchors.items(), key=lambda kv: kv[1].pref_label):
        q = enc.encode([c.text()])[0]
        res = client.query_points(collection_name=DI.DEFAULT_COLLECTION, query=q.tolist(),
                                  limit=a.top_k, with_payload=True).points
        # the anchor's own vocab twin is trivially rank-1 — a domain is not its own constituent
        res = [p for p in res if p.payload.get("pref_label") != c.pref_label]
        # α binds against the best CONCEPT-kind score ("primary constituent CONCEPTS" — RH);
        # sibling domains retrieved alongside are ADJACENCY, kept separately (top-8), never constituency
        conc = [p for p in res if p.payload.get("pref_label") not in anchor_labels]
        adj = [p for p in res if p.payload.get("pref_label") in anchor_labels]
        best = max((float(p.score) for p in conc), default=0.0)
        # IRI is the reference; code/label are display (RH 2026-07-21: the ontology's own
        # discipline — labels mutate freely, references never break, stale IRIs degrade to
        # the archive root's frozen kasten rather than to string forensics)
        cons = [{"iri": p.payload.get("iri"), "code": p.payload.get("code"),
                 "label": p.payload.get("pref_label"), "score": round(float(p.score), 4),
                 "rel": round(float(p.score) / best, 3) if best else 0.0, "kind": "concept"}
                for p in conc if best and float(p.score) >= a.alpha * best]
        cons += [{"iri": p.payload.get("iri"), "code": p.payload.get("code"),
                  "label": p.payload.get("pref_label"), "score": round(float(p.score), 4),
                  "rel": round(float(p.score) / best, 3) if best else 0.0, "kind": "adjacent-domain"}
                 for p in adj[:8]]
        lattice[c.pref_label] = {"iri": c.iri, "point_id": anchor_points.get(c.pref_label),
                                 "constituents": cons}
        for x in cons:
            if x["kind"] == "concept":
                concept_domains[x["code"]].append(c.pref_label)
        if a.write_payloads and c.pref_label in anchor_points:
            client.set_payload(collection_name=DI.DEFAULT_APERTURE,
                               payload={"constituents": cons},
                               points=[anchor_points[c.pref_label]])

    shared = {k: v for k, v in concept_domains.items() if len(v) > 1}
    sizes = [len(v["constituents"]) for v in lattice.values()]
    print(f"{len(lattice)} anchors · constituents per anchor "
          f"min/med/max = {min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}")
    print(f"LATTICE (M:N) confirmed empirically: {len(shared)}/{len(concept_domains)} concepts "
          f"occur in >1 domain")

    out = {"alpha": a.alpha, "top_k": a.top_k, "anchors": lattice,
           "concept_domains": dict(concept_domains),
           "n_shared_concepts": len(shared)}
    (REPO / "build/aperture_constituents.json").write_text(json.dumps(out, indent=1))
    print(f"→ build/aperture_constituents.json (+ payloads written to {DI.DEFAULT_APERTURE})")

    if a.verify:
        # leg 2 — the same margin machinery as the taxonomy (genus_induction.embed)
        from aegir.ontology.genus_induction import embed
        report = {}
        n_flag = 0
        for dom, entry in lattice.items():
            cons = [x for x in entry["constituents"] if x["kind"] == "concept"]
            texts = [vocab[next(k for k, v in vocab.items() if v.code == x["code"])].text()
                     if any(v.code == x["code"] for v in vocab.values()) else x["label"]
                     for x in cons]
            if len(texts) < 2:
                report[dom] = {"n": len(texts), "flagged_pairs": []}
                continue
            E = embed(texts)
            import numpy as np
            S = E @ E.T
            flags = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if float(S[i, j]) > PAIR_COSINE_CEILING:
                        flags.append({"a": cons[i]["label"], "b": cons[j]["label"],
                                      "cosine": round(float(S[i, j]), 4)})
            n_flag += len(flags)
            iu = np.triu_indices(len(texts), 1)
            report[dom] = {"n": len(texts), "max_pair_cosine": round(float(S[iu].max()), 4),
                           "flagged_pairs": flags[:8]}
        (REPO / "build/aperture_sufficiency.json").write_text(json.dumps(
            {"ceiling": PAIR_COSINE_CEILING, "domains": report,
             "total_flagged_pairs": n_flag}, indent=1))
        worst = sorted(report.items(), key=lambda kv: -kv[1].get("max_pair_cosine", 0))[:4]
        print(f"SUFFICIENCY: {n_flag} intra-domain pairs above ceiling {PAIR_COSINE_CEILING}")
        for d, r in worst:
            print(f"  · {d}: n={r['n']} max_pair_cosine={r.get('max_pair_cosine')}")
        print("→ build/aperture_sufficiency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

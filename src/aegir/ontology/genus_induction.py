"""genus_induction — induce the authentic taxonomy mid-tier, CALIBRATED to the generated lexicon's
discriminating potential.

RH constraint (2026-07-14): the logical breadth × depth of the taxonomy is naturally bounded by the
distribution of probability mass relative to the discriminating potential among element specifications within
the generated lexicon (concretely: the vector-space separability the NHSVM / Atelier discriminators can
exploit). Build finer than that frontier and the differentia are below the discriminator's resolution —
inauthentic AND unusable.

RATE-DISTORTION framing: the lexicon (N element specs) is a source; the taxonomy quantizes it into genus cells.
Adding genera lowers within-cell distortion only while the source HAS the information to support it; past the
frontier (the inertia elbow / silhouette peak) we quantize NOISE. So the authentic genus count is the frontier
k, not an arbitrary one — and differentia-sufficiency is measured RELATIVE to that achievable resolution, not
an absolute. Depth recurses the same frontier within each genus (bounded by the same budget → finer scale needs
a richer lexicon / stronger model, i.e. more probability mass — the go-forward lever).

Provisional genera are a SCAFFOLD (statistically-supported clusters); the engine-mediated differentia-authoring
loop AUTHENTICATES them (propose genus label + per-species differentia → HermiT dispose). This module does the
induction + the frontier measurement + the per-element differentia-sufficiency observable. Deterministic given
the encoder; no engine here. [[convert_priority_cas]] [[ontoclean_rigor_roadmap]] [[bfo_cco_grounding_mandate]]
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def load_specs(entities_dir: Path) -> "dict[str, str]":
    """{class: spec_text} — name + definition + attribute names, the element specification the lexicon generated."""
    specs: "dict[str, str]" = {}
    for f in glob.glob(str(entities_dir / "*.json")):
        for e in json.loads(Path(f).read_text()).get("entities", []):
            n = e.get("name")
            if n and n not in specs:
                attrs = ", ".join(a.get("name", "") for a in (e.get("attributes") or []))
                specs[n] = f"{n}. {e.get('definition', '')} Attributes: {attrs}".strip()
    return specs


def embed(texts: "list[str]", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    m = SentenceTransformer(model_name)
    return np.asarray(m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False))


def _intrinsic_dim(X: np.ndarray) -> float:
    """Participation ratio of the covariance eigenspectrum — the effective # of structural dimensions the
    lexicon spans (an information ceiling on how many distinctions it can support)."""
    ev = np.linalg.eigvalsh(np.cov(X.T))
    ev = ev[ev > 1e-9]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def discriminating_potential(X: np.ndarray, ks: "list[int]") -> dict:
    """Sweep k; the FRONTIER k = where the marginal distortion drop per added genus collapses (the elbow) —
    beyond it we split noise. Silhouette (sampled) corroborates. This IS the achievable resolution."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(X)
    rng = np.random.RandomState(13)
    samp = rng.choice(n, min(1200, n), replace=False)
    curve = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=4, random_state=13).fit(X)
        sil = float(silhouette_score(X[samp], km.labels_[samp])) if k < n else 0.0
        curve.append({"k": k, "inertia": float(km.inertia_), "silhouette": round(sil, 4)})
    # elbow: max second-difference of the inertia curve (the knee), guarded to interior points
    inr = [c["inertia"] for c in curve]
    d2 = [inr[i - 1] - 2 * inr[i] + inr[i + 1] for i in range(1, len(inr) - 1)]
    elbow_k = curve[1 + int(np.argmax(d2))]["k"] if d2 else curve[len(curve) // 2]["k"]
    peak_sil_k = max(curve, key=lambda c: c["silhouette"])["k"]
    return {"curve": curve, "elbow_k": elbow_k, "peak_silhouette_k": peak_sil_k,
            "intrinsic_dim": round(_intrinsic_dim(X), 1),
            "frontier_k": elbow_k,  # the authentic genus count = the rate-distortion knee
            "note": "genera beyond frontier_k quantize noise → fabricated differentia below discriminator resolution"}


def induce(X: np.ndarray, k: int) -> "tuple[np.ndarray, np.ndarray]":
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=8, random_state=13).fit(X)
    return np.asarray(km.labels_), np.asarray(km.cluster_centers_)


def differentia_sufficiency(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, names: "list[str]") -> dict:
    """Per element: is it separable from its genus-siblings AT the achievable resolution? margin = (distance to
    the 2nd-nearest genus centroid − distance to its own) ; a small margin OR a near-duplicate sibling ⇒
    INSUFFICIENT differentia (the projection can't individuate it within its genus). Calibrated, not absolute."""
    # cosine sims to all centroids (X, centers are unit-normalised)
    sims = X @ centers.T
    own = sims[np.arange(len(X)), labels]
    sims_masked = sims.copy()
    sims_masked[np.arange(len(X)), labels] = -1
    second = sims_masked.max(axis=1)
    margin = own - second                                   # genus-membership confidence
    # nearest same-genus sibling similarity (near-duplicate ⇒ no differentia vs that sibling)
    insufficient = []
    per_genus = {}
    for g in range(len(centers)):
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            per_genus[g] = {"n": len(idx), "cohesion": None}
            continue
        Xg = X[idx]
        sib = Xg @ Xg.T
        np.fill_diagonal(sib, -1)
        nearest_sib = sib.max(axis=1)                       # each member's closest sibling
        per_genus[g] = {"n": int(len(idx)), "median_margin": round(float(np.median(margin[idx])), 3),
                        "max_sib_sim": round(float(nearest_sib.max()), 3)}
        for j, i in enumerate(idx):
            if margin[i] < 0.03 or nearest_sib[j] > 0.92:   # boundary element OR near-duplicate sibling
                insufficient.append({"class": names[i], "genus": int(g),
                                     "margin": round(float(margin[i]), 3),
                                     "nearest_sibling_sim": round(float(nearest_sib[j]), 3)})
    return {"n": len(X), "insufficient": insufficient,
            "differentia_sufficiency": round(1 - len(insufficient) / len(X), 4),
            "per_genus_sample": {str(g): per_genus[g] for g in list(per_genus)[:8]}}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entities", required=True, help="run entities/ dir (the generated lexicon)")
    ap.add_argument("--ks", default="8,16,32,64,128,256,512")
    ap.add_argument("--k", type=int, default=0, help="override the induced genus count (0 = use frontier_k)")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    specs = load_specs(Path(a.entities))
    names = sorted(specs)
    X = embed([specs[n] for n in names])
    ks = [k for k in (int(x) for x in a.ks.split(",")) if k < len(names)]
    pot = discriminating_potential(X, ks)
    k = a.k or pot["frontier_k"]
    labels, centers = induce(X, k)
    suf = differentia_sufficiency(X, labels, centers, names)
    # a readable genus digest: the 3 most-central members of each genus
    genera = []
    for g in range(k):
        idx = np.where(labels == g)[0]
        if not len(idx):
            continue
        central = idx[np.argsort(-(X[idx] @ centers[g]))][:4]
        genera.append({"genus": g, "n": int(len(idx)), "exemplars": [names[i] for i in central]})
    out = {"n_classes": len(names), "discriminating_potential": pot, "induced_genus_count": k,
           "differentia": {kk: suf[kk] for kk in ("n", "differentia_sufficiency")},
           "n_insufficient": len(suf["insufficient"]),
           "genera": sorted(genera, key=lambda g: -g["n"]), "insufficient_sample": suf["insufficient"][:40]}
    print(f"lexicon {len(names)} element specs · intrinsic_dim {pot['intrinsic_dim']}")
    print(f"  DISCRIMINATING POTENTIAL → frontier_k {pot['frontier_k']} (elbow) · silhouette-peak k "
          f"{pot['peak_silhouette_k']}")
    print(f"  inertia/silhouette curve: " + " ".join(f"{c['k']}:{c['silhouette']}" for c in pot["curve"]))
    biggest = ", ".join("{}×{}".format(g["exemplars"][0], g["n"]) for g in out["genera"][:4])
    print(f"  induced {k} provisional genera (biggest: {biggest})")
    print(f"  differentia_sufficiency {suf['differentia_sufficiency']} @ resolution k={k} "
          f"({len(suf['insufficient'])} insufficient of {len(names)})")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=1))
        print(f"→ {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

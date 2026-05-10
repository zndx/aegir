#!/usr/bin/env python3
"""SOTAB-CTA diagnostic pass — why did the model plateau?

Runs three analyses against the trained SOTAB checkpoint and its val split:

    1. Prediction distribution
       Is the classifier collapsing to a mode class? Report the top-k
       predicted classes, their fractions, and the exact-match confusion
       against the gt.

    2. Cluster geometry (intra vs inter)
       For each val column, extract the pooled-pre-classifier embedding.
       Report the ratio of intra-class cosine distance to inter-class.
       < 1 means features do cluster by label; >= 1 means they don't.

    3. MCL inflation sweep
       Van Dongen's Markov-cluster algorithm — stochastic flow simulation
       on the cosine-similarity graph. Sweep inflation ∈ {1.4, 2.0, 3.0,
       4.0}; report cluster counts + purity vs leaf labels and vs a
       Schema.org parent-bucket mapping. If purity rises monotonically
       at coarser inflations, the embeddings have recoverable hierarchical
       structure. If not, the representation itself (not the classifier
       head) is the bottleneck.

Outputs at build/diagnostics/sotab/{run_id}/:

    report.md           — human-readable summary
    embeddings.npy      — (N, D) pre-classifier pooled embeddings
    labels.npy          — (N,) int leaf labels
    predictions.npy     — (N,) argmax predictions
    confusion.npy       — (K, K) confusion matrix
    mcl_clusters.npz    — per-inflation cluster assignments

Why MCL: we report hierarchical-F1 alongside exact-F1 at eval time, but
we never VERIFY that the embedding geometry actually has hierarchy
recoverable from raw similarity. MCL-inflation-sweep is the audit —
see docs/scratch/2026-04-19/ for the reasoning chain.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegir.data.table_dataset import SotabCTADataset, TASK_NUM_CLASSES
from aegir.data.tokenizer import ByteTokenizer
from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForColumnAnnotation


log = logging.getLogger("sotab_diag")


# ── Schema.org upper-type bucketing ────────────────────────────
#
# SOTAB v2 CTA Schema.org has 91 leaf classes. Group them into Schema.org
# top-level types so MCL purity can be measured at both leaf and parent
# granularity. Derived from Schema.org's documented hierarchy; unrecognized
# labels fall through to "Thing".
_PARENT_BUCKETS: dict[str, tuple[str, ...]] = {
    "Organization": (
        "Organization", "LocalBusiness", "Restaurant", "Hotel", "Airline",
        "Bank", "Bookstore", "ClothingStore", "GasStation", "MovieTheater",
        "Museum", "Library", "Hospital", "Pharmacy", "Dentist",
        "FoodEstablishment", "LodgingBusiness", "NGO", "Corporation",
        "EducationalOrganization", "CollegeOrUniversity", "School",
        "SportsTeam", "SportsClub", "SportsOrganization",
        "AutomotiveBusiness", "AutoDealer", "AutoRepair", "AutoRental",
        "FinancialService", "Store", "ShoppingCenter",
    ),
    "Place": (
        "Place", "City", "Country", "AdministrativeArea", "PostalAddress",
        "TouristAttraction", "Park", "CivicStructure", "Residence",
        "Accommodation", "Apartment", "House", "Beach", "Mountain",
        "River", "Lake", "Cemetery",
    ),
    "CreativeWork": (
        "CreativeWork", "Book", "Article", "BlogPosting", "NewsArticle",
        "Movie", "TVSeries", "TVEpisode", "VideoObject", "ImageObject",
        "MusicRecording", "MusicAlbum", "MusicGroup", "Recipe",
        "Episode", "Review", "Comment", "Dataset", "SoftwareApplication",
        "MobileApplication", "VideoGame", "Photograph", "PhotoAlbum",
        "AudioObject", "WebPage", "WebSite",
    ),
    "Product": (
        "Product", "IndividualProduct", "ProductModel", "SomeProducts",
        "Drug", "MedicalEntity", "FoodProduct", "Offer", "AggregateOffer",
        "Car", "Vehicle", "AutomotiveVehicle", "Boat",
    ),
    "Person": (
        "Person",
    ),
    "Event": (
        "Event", "BusinessEvent", "ChildrensEvent", "ComedyEvent",
        "DanceEvent", "EducationEvent", "FoodEvent", "LiteraryEvent",
        "MusicEvent", "SocialEvent", "SportsEvent", "TheaterEvent",
        "VisualArtsEvent", "PublicationEvent",
    ),
    "Intangible": (
        "Intangible", "Rating", "AggregateRating", "Quantity", "Distance",
        "Duration", "Mass", "Energy", "ContactPoint", "Brand", "Enumeration",
        "StructuredValue", "PropertyValue", "MonetaryAmount", "PriceSpecification",
        "QuantitativeValue", "Role", "Service", "Language", "Audience",
    ),
    "Action": (
        "Action", "MoveAction", "TradeAction", "PlayAction", "AssessAction",
    ),
    "MedicalEntity": (
        "MedicalEntity", "AnatomicalStructure", "DrugClass", "MedicalCause",
        "MedicalCondition", "MedicalDevice", "MedicalGuideline",
        "MedicalIndication", "MedicalProcedure", "MedicalRiskFactor",
        "MedicalStudy", "MedicalTest", "MedicalTherapy",
    ),
}


def _parent_bucket(label: str) -> str:
    """Map a Schema.org leaf label to its documented top-level parent."""
    for parent, leaves in _PARENT_BUCKETS.items():
        if label in leaves:
            return parent
    return "Thing"


# ── Model loading ──────────────────────────────────────────────


def _small_config(num_classes: int, vocab_size: int = 65536) -> AegirConfig:
    """Reproduce the small-model config used by train.py."""
    return AegirConfig(
        arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
        d_model=[256, 384, 384],
        d_intermediate=[0, 0, 0],
        vocab_size=vocab_size,
        ssm_cfg=SSMConfig(),
        attn_cfg=AttnConfig(
            num_heads=[4, 6, 6], rotary_emb_dim=[16, 24, 24], window_size=[]
        ),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=num_classes,
        task_type="cta",
    )


def _load_model(checkpoint_path: Path, device: str) -> AegirForColumnAnnotation:
    num_classes = TASK_NUM_CLASSES["sotab"]
    config = _small_config(num_classes)
    model = AegirForColumnAnnotation(config=config, device=device, dtype=torch.bfloat16)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Diagnostic phases ──────────────────────────────────────────


def _extract_pooled(
    model: AegirForColumnAnnotation,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run forward pass; collect pooled embeddings + predictions.

    We intercept at the pooler output (post-tanh, pre-classifier) — that's
    the most information-dense representation before the head squashes it
    into 91-way logits.
    """
    all_emb: list[np.ndarray] = []
    all_lbl: list[int] = []
    all_pred: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        cls_indexes = batch["cls_indexes"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"]

        # Replicate AegirForColumnAnnotation.forward up to the pooler output.
        hidden = model.embeddings(input_ids) + model.role_embeddings(role_ids)
        B, L, D = hidden.shape
        hidden, _ = model.backbone(hidden, mask=mask)
        hidden = hidden.view(B, L, D)
        from aegir.models.heads import _extract_at_indexes

        pooled = _extract_at_indexes(hidden, cls_indexes)
        pooled = torch.tanh(model.pooler(pooled))
        logits = model.classifier(pooled)
        preds = logits.argmax(dim=-1)

        all_emb.append(pooled.float().cpu().numpy())
        all_lbl.extend(labels.tolist())
        all_pred.extend(preds.cpu().tolist())

    return (
        np.concatenate(all_emb, axis=0),
        np.asarray(all_lbl, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
    )


def _prediction_distribution(
    preds: np.ndarray, labels: np.ndarray, label_names: list[str]
) -> dict:
    counts = Counter(preds.tolist())
    total = int(preds.size)
    top = counts.most_common(10)
    top_rows = [
        {
            "label": label_names[c] if c < len(label_names) else f"<{c}>",
            "fraction": n / total,
            "count": n,
        }
        for c, n in top
    ]
    correct = int((preds == labels).sum())
    return {
        "total": total,
        "unique_preds": len(counts),
        "top10": top_rows,
        "exact_match_acc": correct / total,
        # "Collapse" = single class accounts for >= 50% of predictions.
        "top1_fraction": top[0][1] / total if top else 0.0,
        "mode_collapse": (top[0][1] / total) >= 0.5 if top else False,
    }


def _cluster_geometry(
    embeddings: np.ndarray, labels: np.ndarray, max_samples: int = 2000
) -> dict:
    """Sample-based intra- vs inter-class cosine distance."""
    if embeddings.shape[0] > max_samples:
        idx = np.random.default_rng(0).choice(
            embeddings.shape[0], max_samples, replace=False
        )
        embeddings = embeddings[idx]
        labels = labels[idx]

    # Pairwise L2 BEFORE normalization — primary collapse indicator.
    # If embeddings are identical (representation collapse), L2 distances
    # are ~0 across the board and normalized cosine distance is meaningless.
    # This variance check runs first and short-circuits the rest.
    per_dim_var = embeddings.var(axis=0)
    embed_energy = float(np.linalg.norm(embeddings, axis=1).mean())
    max_pair_l2 = 0.0
    probe_idxs = np.random.default_rng(0).choice(
        embeddings.shape[0], min(50, embeddings.shape[0]), replace=False
    )
    for i in probe_idxs:
        for j in probe_idxs:
            if i != j:
                d = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                max_pair_l2 = max(max_pair_l2, d)
    # "Collapsed" = the spread across samples is <1% of the typical vector
    # energy. Picks up bf16-rounding-noise-only variation as collapse, not
    # just bit-for-bit identity. Tune: this threshold corresponds to a
    # relative perturbation of ~1e-2.
    collapse_ratio = max_pair_l2 / max(embed_energy, 1e-9)
    collapsed = collapse_ratio < 0.01

    emb = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    )
    cos = emb @ emb.T
    dist = 1.0 - cos

    same = labels[:, None] == labels[None, :]
    n = dist.shape[0]
    iu = np.triu_indices(n, k=1)
    same_mask = same[iu]
    dist_upper = dist[iu]

    intra = dist_upper[same_mask]
    inter = dist_upper[~same_mask]

    return {
        "n_samples_used": int(n),
        "collapsed": bool(collapsed),
        "max_pairwise_l2_probe": max_pair_l2,
        "collapse_ratio": collapse_ratio,
        "embed_energy_mean": embed_energy,
        "per_dim_var_max": float(per_dim_var.max()),
        "per_dim_var_median": float(np.median(per_dim_var)),
        "intra_mean": float(np.mean(intra)) if intra.size else None,
        "intra_median": float(np.median(intra)) if intra.size else None,
        "inter_mean": float(np.mean(inter)) if inter.size else None,
        "inter_median": float(np.median(inter)) if inter.size else None,
        "ratio_intra_over_inter": (
            float(np.mean(intra) / np.mean(inter))
            if intra.size and inter.size and np.mean(inter) > 1e-6
            else None
        ),
        "intra_pairs": int(intra.size),
        "inter_pairs": int(inter.size),
    }


# ── MCL ─────────────────────────────────────────────────────────


def _mcl(
    similarity: np.ndarray,
    inflation: float,
    max_iter: int = 100,
    prune_threshold: float = 1e-4,
    converge_tol: float = 1e-5,
) -> np.ndarray:
    """Run Markov cluster (van Dongen, 2000) on a dense similarity matrix.

    Steps:
        - add self-loops (keeps isolated nodes from vanishing)
        - column-normalize → stochastic matrix M
        - repeat: M = M @ M; M = M^inflation (elementwise); renormalize
        - prune tiny entries below threshold each iteration
        - converge when max |Δ| < tol

    Clusters are read off as the connected components of the final
    matrix's support: row i with any non-zero in M[i, j] → i attracted to
    j. Attractors are the diagonal entries of M > 0.
    """
    n = similarity.shape[0]
    # Symmetric similarity. Self-loop so isolated nodes survive inflation.
    M = similarity.astype(np.float64).copy()
    np.fill_diagonal(M, np.maximum(np.diag(M), 1.0))

    # Column-stochastic
    M = M / (M.sum(axis=0, keepdims=True) + 1e-12)

    prev = M.copy()
    for _ in range(max_iter):
        # Expansion
        M = M @ M
        # Inflation
        M = np.power(M, inflation)
        M /= M.sum(axis=0, keepdims=True) + 1e-12
        # Prune
        M[M < prune_threshold] = 0.0
        M /= M.sum(axis=0, keepdims=True) + 1e-12
        # Convergence
        delta = float(np.max(np.abs(M - prev)))
        prev = M.copy()
        if delta < converge_tol:
            break

    # Attractors: rows where diagonal is positive.
    attractors = np.where(np.diag(M) > prune_threshold)[0]
    cluster_ids = -np.ones(n, dtype=np.int64)
    for cid, a in enumerate(attractors):
        members = np.where(M[a] > prune_threshold)[0]
        cluster_ids[members] = cid
    # Unassigned → put each in its own cluster (rare after convergence).
    unassigned = np.where(cluster_ids < 0)[0]
    next_id = int(cluster_ids.max()) + 1 if attractors.size else 0
    for u in unassigned:
        cluster_ids[u] = next_id
        next_id += 1
    return cluster_ids


def _purity(cluster_ids: np.ndarray, labels: np.ndarray) -> float:
    """Cluster purity: fraction of members that share the modal label."""
    total = 0
    correct = 0
    for c in np.unique(cluster_ids):
        members = labels[cluster_ids == c]
        if members.size == 0:
            continue
        mode_count = Counter(members.tolist()).most_common(1)[0][1]
        correct += mode_count
        total += members.size
    return correct / total if total else 0.0


def _mcl_sweep(
    embeddings: np.ndarray,
    leaf_labels: np.ndarray,
    parent_labels: np.ndarray,
    inflations: list[float],
    max_samples: int = 1500,
) -> dict:
    if embeddings.shape[0] > max_samples:
        idx = np.random.default_rng(0).choice(
            embeddings.shape[0], max_samples, replace=False
        )
        embeddings = embeddings[idx]
        leaf_labels = leaf_labels[idx]
        parent_labels = parent_labels[idx]

    # Cosine similarity → clip negatives (MCL is a stochastic-flow algorithm;
    # negative weights break the non-negativity assumption).
    emb = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    )
    cos = emb @ emb.T
    sim = np.maximum(cos, 0.0)

    results: dict = {"n_samples_used": int(sim.shape[0]), "sweep": []}
    cluster_arrays: dict = {}
    for infl in inflations:
        cids = _mcl(sim, inflation=infl)
        n_clusters = int(np.unique(cids).size)
        purity_leaf = _purity(cids, leaf_labels)
        purity_parent = _purity(cids, parent_labels)
        results["sweep"].append({
            "inflation": infl,
            "n_clusters": n_clusters,
            "purity_leaf": purity_leaf,
            "purity_parent": purity_parent,
        })
        cluster_arrays[f"inflation_{infl}"] = cids
    results["_cluster_arrays"] = cluster_arrays
    return results


# ── Report rendering ───────────────────────────────────────────


def _render_report(
    run_id: str, pred: dict, geom: dict, mcl_res: dict, out: Path
) -> str:
    lines = [
        f"# SOTAB diagnostic — {run_id}\n",
        "## 1. Prediction distribution\n",
        f"- N val samples: {pred['total']}",
        f"- Distinct predicted classes: {pred['unique_preds']} / {TASK_NUM_CLASSES['sotab']}",
        f"- Exact-match accuracy: {pred['exact_match_acc']:.4f}",
        f"- Top-1 fraction: {pred['top1_fraction']:.3f}",
        f"- **Mode collapse?** {'yes' if pred['mode_collapse'] else 'no'} "
        "(threshold: top-1 class ≥ 50% of preds)",
        "",
        "| Rank | Label | Count | Fraction |",
        "|------|-------|-------|----------|",
    ]
    for i, row in enumerate(pred["top10"], 1):
        lines.append(
            f"| {i} | `{row['label']}` | {row['count']} | {row['fraction']:.3f} |"
        )
    lines.append("")

    lines += [
        "## 2. Cluster geometry\n",
        f"- Samples used: {geom['n_samples_used']}",
        f"- Embedding energy (mean ||x||): {geom['embed_energy_mean']:.4f}",
        f"- Per-dim variance: max {geom['per_dim_var_max']:.2e}, "
        f"median {geom['per_dim_var_median']:.2e}",
        f"- Max pairwise L2 (50-sample probe): {geom['max_pairwise_l2_probe']:.6f} "
        f"(ratio to energy: {geom['collapse_ratio']:.2e})",
    ]
    if geom["collapsed"]:
        lines += [
            "",
            f"> **COLLAPSE DETECTED**: max-pairwise-L2 / mean-embedding-energy",
            f"> = {geom['collapse_ratio']:.2e} < 1e-2. The pre-classifier",
            f"> embedding is effectively **constant across inputs**; the",
            f"> spread we observe (≤{geom['max_pairwise_l2_probe']:.4f}) is",
            f"> bf16 rounding noise on a mean vector of norm",
            f"> {geom['embed_energy_mean']:.2f}. This is representation",
            f"> collapse, not weak clustering. Subsequent intra/inter and MCL",
            f"> numbers are mathematically degenerate and reported only for",
            f"> reference.",
            "",
        ]
    if geom["intra_mean"] is not None:
        lines.append(
            f"- Intra-class mean cosine distance: "
            f"{geom['intra_mean']:.4f} (median {geom['intra_median']:.4f}, "
            f"n_pairs={geom['intra_pairs']})"
        )
    if geom["inter_mean"] is not None:
        lines.append(
            f"- Inter-class mean cosine distance: "
            f"{geom['inter_mean']:.4f} (median {geom['inter_median']:.4f}, "
            f"n_pairs={geom['inter_pairs']})"
        )
    ratio = geom.get("ratio_intra_over_inter")
    if ratio is not None and not geom["collapsed"]:
        interpretation = (
            "features DO cluster by label (ratio < 1)"
            if ratio < 1.0
            else "features do NOT cluster by label (ratio ≥ 1); "
            "representation — not classifier head — is the bottleneck"
        )
        lines.append(f"- **intra/inter ratio**: {ratio:.4f} → {interpretation}")
    elif geom["collapsed"]:
        lines.append(
            "- **intra/inter ratio not meaningful** under representation collapse."
        )
    lines.append("")

    lines += [
        "## 3. MCL inflation sweep\n",
        f"- Samples used: {mcl_res['n_samples_used']}",
        "",
        "| Inflation | # Clusters | Purity (leaf) | Purity (parent) |",
        "|-----------|------------|---------------|-----------------|",
    ]
    for row in mcl_res["sweep"]:
        lines.append(
            f"| {row['inflation']:.1f} | {row['n_clusters']} | "
            f"{row['purity_leaf']:.4f} | {row['purity_parent']:.4f} |"
        )
    lines.append("")

    # Interpretation
    sweep = mcl_res["sweep"]
    all_single_cluster = all(r["n_clusters"] == 1 for r in sweep)
    if all_single_cluster and geom.get("collapsed"):
        lines += [
            "### Interpretation\n",
            "- MCL produces 1 cluster at every inflation because the embedding",
            "  space has collapsed to a single point. The reported",
            f"  `purity_parent` ≈ {sweep[0]['purity_parent']:.3f} is just the",
            "  fraction of val samples whose label rolls up to the *modal*",
            "  Schema.org parent — a property of the label distribution, NOT",
            "  of the embedding geometry.",
            "- **Verdict: geometry-audit cannot proceed on a collapsed model.**",
            "  Fix the collapse first (gradient-flow hygiene), then re-run.",
            "",
        ]
    else:
        purity_trend = [r["purity_parent"] for r in sweep]
        monotonic = all(
            purity_trend[i] >= purity_trend[i + 1] - 0.02
            for i in range(len(purity_trend) - 1)
        )
        lines += [
            "### Interpretation\n",
            "- Coarser inflation → larger clusters. If **parent-level purity**",
            "  rises (or stays flat) at coarser inflation, the embedding",
            "  geometry admits recoverable hierarchical structure.",
            f"- Parent purity across inflations {purity_trend} — "
            f"{'monotonically coherent' if monotonic else 'NON-monotonic (structure is weak/noisy)'}.",
            "",
        ]

    report = "\n".join(lines)
    (out / "report.md").write_text(report)
    return report


# ── Entrypoint ──────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    ap.add_argument(
        "--run-id",
        default="20260419T182156Z_d8feb9905f_small_sotab",
        help="Run ID for labeling the output directory",
    )
    ap.add_argument("--sotab-dir", type=Path, default=Path("/raid/datasets/sotab"))
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-context-cols", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--inflations",
        nargs="+",
        type=float,
        default=[1.4, 2.0, 3.0, 4.0],
    )
    ap.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Subsample val for MCL (full set is memory-heavy)",
    )
    args = ap.parse_args()

    out_dir = Path("build/diagnostics/sotab") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", out_dir)

    # 1. Load val dataset
    log.info("Loading SOTAB val split...")
    tokenizer = ByteTokenizer()
    val_ds = SotabCTADataset(
        data_dir=args.sotab_dir,
        split="val",
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_context_cols=args.max_context_cols,
    )
    if args.max_val_samples and len(val_ds) > args.max_val_samples:
        from torch.utils.data import Subset

        val_ds = Subset(val_ds, list(range(args.max_val_samples)))
    log.info("  val samples: %d", len(val_ds))

    # Resolve the discovered label vocab so we can rename integer labels.
    vocab = val_ds.dataset._label_vocab_cache.get(
        ("CTA", "schemaorg", str(args.sotab_dir))
    ) if hasattr(val_ds, "dataset") else None
    if vocab is None:
        # Direct dataset case
        vocab = SotabCTADataset._label_vocab_cache.get(
            ("CTA", "schemaorg", str(args.sotab_dir))
        )
    label_names = [""] * TASK_NUM_CLASSES["sotab"]
    if vocab:
        for name, idx in vocab.items():
            if 0 <= idx < len(label_names):
                label_names[idx] = name

    # Import train.py's collate_fn from the repo root (not on sys.path by default).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_aegir_train", str(Path(__file__).parent.parent / "train.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import train.py")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    collate_fn = train_mod.collate_fn

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 2. Load model
    log.info("Loading checkpoint %s...", args.checkpoint)
    model = _load_model(args.checkpoint, args.device)

    # 3. Extract embeddings + predictions
    log.info("Running val forward pass...")
    with torch.no_grad():
        embeddings, labels, preds = _extract_pooled(model, loader, args.device)
    log.info("  embeddings: %s  labels: %s", embeddings.shape, labels.shape)

    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "predictions.npy", preds)

    # Parent mapping
    parent_ids_by_leaf = {
        i: _parent_bucket(label_names[i]) for i in range(len(label_names))
    }
    parent_name_to_int = {
        p: i for i, p in enumerate(sorted(set(parent_ids_by_leaf.values())))
    }
    parent_labels = np.array(
        [parent_name_to_int[parent_ids_by_leaf[int(l)]] for l in labels]
    )

    # Confusion
    K = TASK_NUM_CLASSES["sotab"]
    conf = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(labels, preds):
        conf[int(t), int(p)] += 1
    np.save(out_dir / "confusion.npy", conf)

    # 4. Run three phases
    log.info("Phase 1: prediction distribution...")
    pred_stats = _prediction_distribution(preds, labels, label_names)

    log.info("Phase 2: cluster geometry...")
    geom_stats = _cluster_geometry(embeddings, labels)

    log.info("Phase 3: MCL inflation sweep...")
    mcl_res = _mcl_sweep(
        embeddings, labels, parent_labels, args.inflations
    )
    cluster_arrays = mcl_res.pop("_cluster_arrays")
    np.savez(out_dir / "mcl_clusters.npz", **cluster_arrays)

    # 5. Report
    log.info("Rendering report...")
    report = _render_report(args.run_id, pred_stats, geom_stats, mcl_res, out_dir)

    # Also dump machine-readable summary
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "prediction_distribution": pred_stats,
                "cluster_geometry": geom_stats,
                "mcl_sweep": mcl_res,
                "label_names": label_names,
            },
            indent=2,
        )
    )

    print("\n" + report)
    print(f"\nFull artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

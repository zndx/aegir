"""build_topic_model — fit T_I on the pinned input corpus + compute null_stats.

Per the v0.5 concept brief P1b-β: fits the input-corpus topic
model *T_I* once on a held-out subset of v2's SchemaPile +
FinePDFs-lab slices, caches it to disk, then generates a
structural-shuffle null distribution by sampling random
compositions over the combined catalog and computing alignment
against *T_I*. The resulting null statistics
(``null_mean`` / ``null_std`` / ``null_p5`` / ``null_p95``) are
written into the catalog's ``null_stats`` field so the runtime
verifier can normalize R_D without re-running the null
construction.

Usage (via Justfile recipe `just build-topic-model`):

    bash scripts/setup_jvm_env.sh    # not strictly needed for
                                     # this script — only for libz
    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python \\
        scripts/build_topic_model.py \\
        --catalog src/aegir/ontology/catalog/01_foundation.json \\
                  src/aegir/ontology/catalog/02_observation_measurement.json \\
        --output  src/aegir/ontology/catalog/null_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from aegir.ontology.schema import Catalog, CatalogTemplate, load_catalog  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

DEFAULT_INPUT_CORPUS_PATHS = [
    "/raid/datasets/aegir-corpus-v1/schemapile/schemapile_eval.txt",
    "/raid/datasets/aegir-corpus-v1/finepdfs-lab/finepdfs_lab_eval.txt",
]
DEFAULT_T_I_CACHE = Path("src/aegir/ontology/T_I.pkl")
DEFAULT_PER_CORPUS_LINES = 5000
DEFAULT_TI_K = 100
DEFAULT_NULL_SAMPLES = 200
DEFAULT_NULL_TEMPLATES_PER_COMPOSITION = 24
DEFAULT_NULL_TV_K = 12


# ---- Corpus loading ----


def load_input_corpus(paths: list[str], per_corpus_lines: int) -> list[str]:
    """Load and minimally clean input-corpus sentences from each
    path. Trims to ``per_corpus_lines`` per source for a balanced
    *I* across SchemaPile and FinePDFs-lab.
    """
    sentences: list[str] = []
    for path in paths:
        n_kept = 0
        with open(path) as f:
            for line in f:
                if n_kept >= per_corpus_lines:
                    break
                line = line.strip()
                if len(line) < 12:
                    continue
                if len(line) > 1500:
                    line = line[:1500]
                sentences.append(line)
                n_kept += 1
        logger.info("loaded %d sentences from %s", n_kept, path)
    return sentences


# ---- Composition rendering for the null ----


def render_composition_verbalizations(
    composition: list[tuple[str, dict[str, str]]],
    catalog: Catalog,
) -> list[str]:
    """Render a composition into a list of verbalization sentences
    by substituting slot fillers into pre-cached ``verbal_template``.
    """
    out: list[str] = []
    for tid, fillers in composition:
        try:
            tmpl = catalog.by_id(tid)
        except KeyError:
            continue
        if not tmpl.verbal_template:
            continue
        verbal = tmpl.verbal_template
        for slot_name, filler in fillers.items():
            verbal = verbal.replace("{" + slot_name + "}", filler)
        out.append(verbal)
    return out


def random_composition(
    catalog: Catalog,
    n_templates: int,
    seed: int,
    filler_inventory: list[str] | None = None,
) -> list[tuple[str, dict[str, str]]]:
    """Sample a random composition of ``n_templates`` distinct
    templates from the catalog with random slot fillers.

    The structural-shuffle null preserves count statistics of the
    catalog (axiom-shape distribution, slot-type distribution)
    while randomizing the specific template selection and slot
    fillers — destroying any topic-aligned selection signal.
    """
    rng = random.Random(seed)
    eligible = [t for t in catalog.templates if t.verbal_template]
    if not eligible:
        return []
    chosen = rng.sample(eligible, min(n_templates, len(eligible)))
    inventory = filler_inventory or _DEFAULT_FILLER_INVENTORY

    composition: list[tuple[str, dict[str, str]]] = []
    for tmpl in chosen:
        slot_fillers = {
            slot_name: rng.choice(inventory) for slot_name in tmpl.slot_types
        }
        composition.append((tmpl.template_id, slot_fillers))
    return composition


_DEFAULT_FILLER_INVENTORY = [
    f"sdg:NullFiller_{i:04d}" for i in range(200)
]


# ---- T_I fit ----


def fit_T_I(
    paths: list[str],
    per_corpus_lines: int,
    k: int,
    cache_path: Path,
    encoder_name: str | None = None,
):
    """Fit the input-corpus topic model and cache to disk."""
    from aegir.ontology.topic_alignment import (
        DEFAULT_ENCODER_MODEL,
        fit_topic_model,
        save_topic_model,
        get_encoder,
    )
    encoder_name = encoder_name or DEFAULT_ENCODER_MODEL

    logger.info("loading input corpus")
    sentences = load_input_corpus(paths, per_corpus_lines)
    logger.info("input corpus: %d sentences", len(sentences))

    logger.info("loading encoder %s", encoder_name)
    encoder = get_encoder(encoder_name)

    logger.info("fitting T_I (k=%d)", k)
    t0 = time.time()
    t_i = fit_topic_model(sentences, k=k, encoder=encoder, encoder_model=encoder_name)
    logger.info("T_I fit done in %.1fs", time.time() - t0)

    save_topic_model(t_i, cache_path)
    return t_i, encoder


# ---- Null distribution ----


def compute_null_distribution(
    catalog: Catalog,
    t_i,
    encoder,
    n_samples: int,
    n_templates_per_composition: int,
    tv_k: int,
) -> dict:
    """Compute the structural-shuffle null distribution: ``n_samples``
    random compositions over the catalog, alignment of each against
    *T_I*. Returns mean/std/p5/p95 of the resulting null scores.
    """
    from aegir.ontology.topic_alignment import (
        alignment_score,
        fit_topic_model,
    )

    scores: list[float] = []
    t0 = time.time()
    for i in range(n_samples):
        comp = random_composition(catalog, n_templates_per_composition, seed=i)
        verbalizations = render_composition_verbalizations(comp, catalog)
        if len(verbalizations) < 2:
            continue
        t_v = fit_topic_model(verbalizations, k=tv_k, encoder=encoder)
        score = alignment_score(t_v, t_i)
        scores.append(score)
        if (i + 1) % 25 == 0:
            logger.info(
                "null %d/%d  mean=%.4f  std=%.4f  (%.1fs)",
                i + 1, n_samples,
                float(np.mean(scores)), float(np.std(scores)),
                time.time() - t0,
            )

    arr = np.array(scores)
    return {
        "null_mean": float(arr.mean()),
        "null_std": float(arr.std()),
        "null_p5": float(np.percentile(arr, 5)),
        "null_p95": float(np.percentile(arr, 95)),
        "n_samples": int(arr.shape[0]),
        "tv_k": int(tv_k),
        "ti_k": int(t_i.k),
        "n_templates_per_composition": n_templates_per_composition,
    }


# ---- Main ----


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument(
        "--catalog",
        nargs="+",
        required=True,
        help="One or more catalog JSON files to combine for null-distribution sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/aegir/ontology/null_stats.json",
        help="Where to write the null statistics JSON.",
    )
    parser.add_argument(
        "--ti-cache",
        type=str,
        default=str(DEFAULT_T_I_CACHE),
        help="Cache path for T_I (pickled).",
    )
    parser.add_argument(
        "--input-corpus",
        nargs="+",
        default=DEFAULT_INPUT_CORPUS_PATHS,
        help="Paths to text files comprising the input corpus *I*.",
    )
    parser.add_argument(
        "--per-corpus-lines",
        type=int,
        default=DEFAULT_PER_CORPUS_LINES,
        help="Max lines to load per source corpus.",
    )
    parser.add_argument(
        "--ti-k",
        type=int,
        default=DEFAULT_TI_K,
        help="Number of topics for T_I.",
    )
    parser.add_argument(
        "--null-samples",
        type=int,
        default=DEFAULT_NULL_SAMPLES,
        help="Number of null compositions to sample.",
    )
    parser.add_argument(
        "--null-templates-per-composition",
        type=int,
        default=DEFAULT_NULL_TEMPLATES_PER_COMPOSITION,
        help="Templates per random composition in the null.",
    )
    parser.add_argument(
        "--null-tv-k",
        type=int,
        default=DEFAULT_NULL_TV_K,
        help="Topic count for T_V fits during null sampling.",
    )
    parser.add_argument(
        "--skip-ti-fit",
        action="store_true",
        help="Reuse cached T_I if it exists; only recompute the null.",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="Sentence-transformer for T_I (default: topic_alignment.DEFAULT_ENCODER_MODEL). "
             "Use sentence-transformers/all-mpnet-base-v2 for the canonical mpnet ground.",
    )
    args = parser.parse_args()

    catalogs = [load_catalog(p) for p in args.catalog]
    combined_templates: list[CatalogTemplate] = []
    for c in catalogs:
        combined_templates.extend(c.templates)
    combined = Catalog(version="combined", templates=combined_templates, null_stats={})
    logger.info("combined catalog: %d templates from %d files",
                len(combined.templates), len(catalogs))

    ti_cache = Path(args.ti_cache)
    if args.skip_ti_fit and ti_cache.exists():
        from aegir.ontology.topic_alignment import load_topic_model, get_encoder
        logger.info("loading cached T_I from %s", ti_cache)
        t_i = load_topic_model(ti_cache)
        encoder = get_encoder(t_i.encoder_model)
    else:
        t_i, encoder = fit_T_I(
            args.input_corpus,
            args.per_corpus_lines,
            args.ti_k,
            ti_cache,
            encoder_name=args.encoder,
        )

    logger.info("computing null distribution (%d samples)", args.null_samples)
    stats = compute_null_distribution(
        combined,
        t_i,
        encoder,
        n_samples=args.null_samples,
        n_templates_per_composition=args.null_templates_per_composition,
        tv_k=args.null_tv_k,
    )
    stats["catalog_files"] = args.catalog
    stats["ti_cache"] = str(ti_cache)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("wrote null stats to %s", args.output)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

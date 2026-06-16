"""Readers for the three Data Products + optional on-disk artifacts.

Thin loaders that return raw typed data + presence; the projectors in ``build.py``
turn them into Notes. The domain-adapted (local, "what we KNOW") sources are the
ontology catalog and the deterministic DDL lowering; on-disk corpus/coverage runs feed
the content/topics products when present (graceful skip otherwise). The canonical
SHARED overlay (the ``corpora`` submodule) is layered in by ``load_ontology`` once its
published shape is finalized — TODO marked below; the projection is valid from the
local catalog alone today.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

from aegir.ontology.schema import CatalogTemplate

REPO = Path(__file__).resolve().parents[3]
CATALOG_GLOB = "src/aegir/ontology/catalog/0[1-7]_*.json"


def kb_dir() -> Path:
    """The KB projection root (gitignored, regenerable). Override with AEGIR_KB_DIR."""
    return Path(os.environ.get("AEGIR_KB_DIR") or (REPO / "build" / "dev"))


def load_ontology() -> list[tuple[str, CatalogTemplate]]:
    """``[(family, CatalogTemplate)]`` from the 7 catalog family files (domain-adapted).

    TODO(sync): overlay the canonical published ontology from the ``corpora`` submodule
    (zndx/sdg-corpora) when checked out — the SHARE layer atop this KNOW layer.
    """
    from aegir.ontology.schema import load_catalog
    out: list[tuple[str, CatalogTemplate]] = []
    for f in sorted(glob.glob(str(REPO / CATALOG_GLOB))):
        if "candidate" in f or "combined" in f:
            continue
        fam = Path(f).stem
        for t in load_catalog(f).templates:
            out.append((fam, t))
    return out


def _first(patterns: list[str], env: str) -> Path | None:
    """Env override, else the lexicographically-latest match of the given globs."""
    if os.environ.get(env):
        p = Path(os.environ[env])
        return p if p.exists() else None
    for pat in patterns:
        m = sorted(glob.glob(pat))
        if m:
            return Path(m[-1])
    return None


_ART = "/raid/checkpoints/aegir-artifacts"


def corpus_run() -> Path | None:
    """Latest on-disk ``chapters.parquet`` (the content Data Product), or None."""
    return _first([f"{_ART}/chapters_v*/*/chapters.parquet",
                   f"{_ART}/chapters*/chapters.parquet"], "AEGIR_CORPUS_RUN")


def coverage_run() -> Path | None:
    """Latest on-disk ``topic_coverage.parquet`` (topics), or None."""
    return _first([f"{_ART}/coverage_v*/*/topic_coverage.parquet"], "AEGIR_COVERAGE_RUN")


def _recs(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def corpus_recs() -> tuple[list[dict], Path | None]:
    """(chapter rows, run path) for the content Data Product — read once so the build
    can derive cross-references (term→chapters, topic→chapters) before projecting."""
    r = corpus_run()
    return (_recs(r), r) if r else ([], None)


def coverage_recs() -> tuple[list[dict], Path | None]:
    """(topic rows, run path) for topics — read once for term→topics cross-refs."""
    r = coverage_run()
    return (_recs(r), r) if r else ([], None)

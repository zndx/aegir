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
CATALOG_GLOB = "src/aegir/ontology/catalog/0*.json"  # 0* (incl. 08_derived); candidate/combined filtered below


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
    """Latest on-disk ``chapters.parquet`` (the content Data Product), or None. Prefers the
    C2.5 L2 **natural canonical-deliverable** corpus (path-a ``c3nat`` — de-leaked + domain-mixed +
    natural-named + topical prose) over the older ``sdg_corpus_v*`` / ``chapters_v*`` runs, so the
    content navigator shows the latest REFINED corpus by default (the pre-refinement sdg_corpus_v0_3
    runs read as degraded). ``AEGIR_CORPUS_RUN`` still overrides; if/when the canonical corpus is
    regenerated at scale with --naming natural, point this (or that override) at it."""
    return _first(["/raid/build/aegir/path-a/c3nat/chapters/*/chapters.parquet",  # natural canonical deliverable — preferred
                   f"{_ART}/sdg_corpus_v*/*/chapters.parquet",
                   f"{_ART}/chapters_v*/*/chapters.parquet",
                   f"{_ART}/chapters*/chapters.parquet"], "AEGIR_CORPUS_RUN")


def coverage_run() -> Path | None:
    """Latest on-disk ``topic_coverage.parquet`` (topics), or None."""
    return _first([f"{_ART}/coverage_v*/*/topic_coverage.parquet"], "AEGIR_COVERAGE_RUN")


def _recs(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def corpus_runs() -> list[tuple[Path, str]]:
    """Both registers of the content Data Product as ``(path, register)``. The NATURAL corpus (path-a
    ``c3nat`` — natural physical names + topical prose) is the canonical deliverable; the SEMANTIC corpus
    (``c3`` / legacy ``sdg_corpus_v*`` — the ontology-discourse register) is ALSO valuable training material:
    prose written around ontological constructs elucidates how and why the ontology takes this specific
    materialization. Both surfaces are projected and tagged. Overrides: ``AEGIR_CORPUS_RUN_NATURAL`` /
    ``AEGIR_CORPUS_RUN_SEMANTIC`` (``AEGIR_CORPUS_RUN`` still forces a single natural run)."""
    out: list[tuple[Path, str]] = []
    nat = _first(["/raid/build/aegir/path-a/c3nat/chapters/*/chapters.parquet"],
                 "AEGIR_CORPUS_RUN_NATURAL") or corpus_run()
    sem = _first(["/raid/build/aegir/path-a/c3/chapters/*/chapters.parquet",
                  f"{_ART}/sdg_corpus_v*/*/chapters.parquet",
                  f"{_ART}/chapters_v*/*/chapters.parquet"], "AEGIR_CORPUS_RUN_SEMANTIC")
    ref = _first(["/raid/build/aegir/path-a/refine_scale/chapters.parquet"], "AEGIR_REFINE_CORPUS")
    if nat:
        out.append((nat, "natural"))
    if sem and sem != nat:
        out.append((sem, "semantic"))
    if ref:
        out.append((ref, "refined"))   # refinement-loop corpus — register is PER-RECORD (see corpus_recs)
    return out


def corpus_recs() -> tuple[list[dict], Path | None]:
    """(chapter rows tagged with ``_register``, primary run path) for the content Data Product — reads ALL
    registers (natural ⊕ semantic ⊕ the refined dual-register corpus) so the lineup mixes them; read once for
    the build's cross-references. A record's own ``register`` column wins (the refined corpus is per-record)."""
    runs = corpus_runs()
    recs: list[dict] = []
    for path, reg in runs:
        for rec in _recs(path):
            rec = dict(rec)
            rec["_register"] = rec.get("register") or reg
            recs.append(rec)
    return recs, (runs[0][0] if runs else None)


def coverage_recs() -> tuple[list[dict], Path | None]:
    """(topic rows, run path) for topics — read once for term→topics cross-refs."""
    r = coverage_run()
    return (_recs(r), r) if r else ([], None)


def hierarchy_dir() -> Path:
    """Where ``mediate_hierarchy`` *stages* verified per-category subsumption edges (the
    mediation artifact, promoted INTO the catalog by ``--promote``). Override with
    AEGIR_HIERARCHY_RUN."""
    return Path(os.environ.get("AEGIR_HIERARCHY_RUN") or f"{_ART}/evidence/hierarchy")


def term_hierarchy() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """``(broader, narrower)`` term-id maps read from the CATALOG ``broader`` fields — the
    SOURCE OF TRUTH (promoted from ``mediate_hierarchy``'s HermiT-verified mediation). The
    lineup projects the hierarchy from the ontology, not the evidence side-artifact.
    ``broader[child] = [parents]``, ``narrower[parent] = [children]``. Empty until promoted
    (graceful skip, like the corpus/coverage cross-refs)."""
    broader: dict[str, list[str]] = {}
    narrower: dict[str, list[str]] = {}
    for _fam, t in load_ontology():
        for p in (getattr(t, "broader", None) or []):
            broader.setdefault(t.template_id, []).append(p)
            narrower.setdefault(p, []).append(t.template_id)
    return broader, narrower


def relational_shape() -> dict | None:
    """The newest DDL spine's shape census vs the SchemaPile norms (graceful None when no spine).
    Cheap: base_table_index.n_cols (minus the surrogate pk) + views.parquet widths — no cell scan.
    EMD vs build/schemapile_shape_norms.json when the #139 instrument has run; None-field otherwise."""
    import json as _json
    manifests = sorted(REPO.glob("build/spine_*/*/manifest.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        return None
    run_dir = manifests[0].parent
    try:
        import pyarrow.parquet as _pq
        widths = [max(0, r["n_cols"] - 1)
                  for r in _pq.read_table(run_dir / "base_table_index.parquet").to_pylist()]
        vp = run_dir / "views.parquet"
        view_widths = []
        if vp.exists():
            view_widths = [len(_json.loads(r["columns_json"]))
                           for r in _pq.read_table(vp, columns=["columns_json"]).to_pylist()]
    except Exception:
        return None
    allw = sorted(widths + view_widths)
    if not allw:
        return None
    n = len(allw)
    pct = lambda q: allw[min(n - 1, int(q * n))]  # noqa: E731
    out = {
        "spine_run": run_dir.name,
        "n_tables": n,
        "cols_median": allw[n // 2],
        "cols_p90": pct(0.90),
        "cols_p99": pct(0.99),
        "cols_max": allw[-1],
        "wide_rate": round(sum(1 for x in allw if x >= 20) / n, 4),
        "strata": {"base": len(widths), "view": len(view_widths)},
    }
    norms_path = REPO / "build" / "schemapile_shape_norms.json"
    out["shape_emd"] = None
    if norms_path.exists():
        try:
            norms = _json.loads(norms_path.read_text())
            ref = norms.get("col_count_histogram") or {}
            if ref:
                # discrete 1-Wasserstein over the col-count distributions
                hi = max(max(allw), max(int(k) for k in ref))
                ours = [0.0] * (hi + 1)
                for x in allw:
                    ours[min(x, hi)] += 1 / n
                tot = sum(ref.values())
                theirs = [0.0] * (hi + 1)
                for k, v in ref.items():
                    theirs[min(int(k), hi)] += v / tot
                cum = emd = 0.0
                for a, b in zip(ours, theirs):
                    cum += a - b
                    emd += abs(cum)
                out["shape_emd"] = round(emd, 3)
        except Exception:
            pass
    return out


def ontology_metrology() -> dict | None:
    """The realized ontology's IOF/OQuaRE quality profile — ``ontology_metrology.compute`` (the rigor +
    field-standard metrics) + ``ontology_oquare.oquare`` (the 1-5 quality model + the publish-gate verdict),
    or None if the realized OWL is absent (graceful skip, like the corpus/coverage cross-refs). In-process,
    pure rdflib (no JVM): consumes the realized ``corpora/ontology/sdg-ontology.owl`` + its HermiT certificate."""
    owl = REPO / "corpora" / "ontology" / "sdg-ontology.owl"
    cert = REPO / "corpora" / "ontology" / "HERMIT_CERTIFICATE.md"
    if not owl.exists():
        return None
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from ontology_metrology import compute
        from ontology_oquare import oquare
        m = compute(str(owl))
        oq = oquare(str(owl), str(cert) if cert.exists() else None)
    except Exception:
        return None
    return {
        "definitional_completeness": round(m["definitional_completeness"], 4),
        "bfo_grounded": round(m["bfo_grounded"], 4),
        "realizable_machinery": m["realizable_machinery"],
        "def_annotation_coverage": round(m["def_annotation_coverage"], 4),
        "ar": round(m["ar"], 4), "rr": round(m["rr"], 4), "ir": round(m["ir"], 4),
        "aronto": round(m["aronto"], 4), "dit": m["dit"], "tm": round(m["tm"], 4),
        "n_domain_classes": m["n_domain_classes"], "n_datatype_properties": m["n_datatype_properties"],
        "oquare_aggregate": oq["aggregate"], "oquare_characteristics": oq["characteristics"],
        "oquare_green": oq["gate_green"], "consistent": oq["consistent"],
        # OntoClean Tier-A/B taxonomic-correctness proxies (un-gameable)
        "taxonomic_cleanliness": m["taxonomic_cleanliness"], "ontoclean_violations": m["ontoclean_violations"],
        "subsumption_cycles": m["subsumption_cycles"], "sibling_disjointness": m["sibling_disjointness"],
    }


def sdg_corpus() -> "dict | None":
    """LIVE state of the sdg corpus dir (`just metaflow` accretes it): counts +
    the flow's metrics/congruence when present. Graceful None when the corpus is absent —
    the lineup stays in sync with the pipeline AS IT RUNS (kb-build re-projects)."""
    import json as _json
    root = Path("/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus")
    if not root.exists():
        return None
    out: dict = {
        "root": str(root),
        "passages_derived": len(list((root / "entities").glob("*.json"))),
        "chapters_natural": len(list((root / "chapters").rglob("natural.md"))),
        "chapters_semantic": len(list((root / "chapters").rglob("semantic.md"))),
    }
    for f, key in (("metrics.json", "metrics"), ("congruence.json", "congruence"),
                   ("ontology/structure.json", "structure")):
        fp = root / f
        if fp.exists():
            try:
                d = _json.loads(fp.read_text())
                out[key] = d.get("per_register", d) if key == "congruence" else d
            except Exception:  # noqa: BLE001
                pass
    return out

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


def kb_dir() -> Path:
    """The KB projection root (gitignored, regenerable). Override with AEGIR_KB_DIR."""
    return Path(os.environ.get("AEGIR_KB_DIR") or (REPO / "build" / "dev"))


def template_category(t: CatalogTemplate) -> str:
    """The term's browse category — the derive's axiom PATTERN (``provenance.pattern``),
    the engineered axis that replaced the retired hand-authored families (8df0d23).
    Intermediate domain classes (``define_intermediate_classes`` output — no pattern
    provenance) group under ``intermediate``."""
    prov = t.provenance or {}
    if prov.get("pattern"):
        return str(prov["pattern"])
    return "intermediate"


def relational_category(t: CatalogTemplate) -> str:
    """The table's browse category — the DDL shape the template grounds
    (``provenance.grounds_ddl``: junction / dimension / nested-child / constraint /
    enum / eav); intermediate classes lower to plain ``intermediate`` tables."""
    prov = t.provenance or {}
    g = prov.get("grounds_ddl")
    if isinstance(g, (list, tuple)):
        g = g[0] if g else None
    return str(g) if g else "intermediate"


def load_ontology() -> list[tuple[str, CatalogTemplate]]:
    """``[(category, CatalogTemplate)]`` from the live catalog.

    The category is the template's provenance pattern (``template_category``), NOT the
    catalog filename — the file-stem "family" axis died with the 01-07 retirement, and
    the catalog is a single file now (everything is derived — RH 2026-07-07).

    TODO(sync): overlay the canonical published ontology from the ``corpora`` submodule
    (zndx/sdg-corpora) when checked out — the SHARE layer atop this KNOW layer.
    """
    from aegir.ontology.schema import catalog_files, load_catalog
    out: list[tuple[str, CatalogTemplate]] = []
    for f in catalog_files():
        for t in load_catalog(f).templates:
            out.append((template_category(t), t))
    return out


def load_retired_ontology() -> list[tuple[str, CatalogTemplate]]:
    """``[(family, CatalogTemplate)]`` for the catalog generation the pre-R1 corpus and
    coverage still CITE — the hand-authored 01-07 seed families plus the superseded
    early ``08_derived`` batch — recovered from git history at build time (the parent of
    the commit that deleted the seed files; discovered, not hardcoded).

    The lineup projects these as ARCHIVE term notes so corpus citations RESOLVE (the
    demote-to-archive ruling applied to the ontology axis) instead of dangling. They stop
    being cited — and quietly remain archive tombstones — once the corpus/coverage are
    regenerated against the derived catalog (R1). Graceful ``[]`` outside a git checkout.
    """
    import json as _json
    import subprocess

    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  cwd=str(REPO)).stdout
        except Exception:  # noqa: BLE001
            return ""

    fields = set(CatalogTemplate.__dataclass_fields__)
    out: list[tuple[str, CatalogTemplate]] = []
    seen: set[str] = set()

    def _collect(fam: str, doc: dict) -> None:
        for row in doc.get("templates", []):
            if row.get("template_id") in seen:
                continue
            seen.add(row["template_id"])
            # tolerate schema drift across the history boundary
            out.append((fam, CatalogTemplate(**{k: v for k, v in row.items() if k in fields})))

    # 1) The hand-authored seed families, at the parent of their deletion commit.
    sha = _git("log", "--diff-filter=D", "-1", "--format=%H", "--",
               "src/aegir/ontology/catalog/01_foundation.json").strip()
    if sha:
        for f in _git("ls-tree", "--name-only", f"{sha}^", "src/aegir/ontology/catalog/").split():
            base = f.rsplit("/", 1)[-1]
            if not base.startswith("0") or "candidate" in base or "combined" in base:
                continue
            try:
                _collect(base.removesuffix(".json"), _json.loads(_git("show", f"{sha}^:{f}")))
            except Exception:  # noqa: BLE001
                continue
    # 2) Every superseded generation of the derived catalog (newest-first, so the most
    #    recent superseded definition of an id wins). The corpus was generated across
    #    several promotions; ids replaced within the file's history are cited too.
    #    Sweep BOTH names: the live catalog.json and its pre-rename 08_derived.json era
    #    (the "08_derived" label is kept for the historical generations — it's what
    #    those files were called when the corpus cited them).
    for p in ("src/aegir/ontology/catalog/catalog.json",
              "src/aegir/ontology/catalog/08_derived.json"):
        for h in _git("log", "--format=%H", "--", p).split():
            try:
                _collect("08_derived", _json.loads(_git("show", f"{h}:{p}")))
            except Exception:  # noqa: BLE001
                continue
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


def _fullest_spine() -> "tuple[Path, dict] | None":
    """The REALIZED spine run covering the most tables (mtime-newest on ties) — a partial
    cohort run (e.g. a release subset materialized after the full spine) must not shadow
    the full-catalog run for per-template projection surfaces. Coverage is measured from
    the PARQUET row count (metadata-only read), not the manifest claim — a cohort run has
    been observed carrying the full-run manifest over subset parquets."""
    import json as _json
    best: "tuple[int, Path, dict] | None" = None
    for p in sorted(REPO.glob("build/spine_*/*/manifest.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            man = _json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not man.get("realize"):
            continue
        ddl = p.parent / "ddl_statements.parquet"
        try:
            import pyarrow.parquet as _pq
            n = _pq.ParquetFile(ddl).metadata.num_rows
        except Exception:  # noqa: BLE001
            continue
        if best is None or n > best[0]:
            best = (n, p, man)
    return (best[1], best[2]) if best else None


def spine_manifest() -> dict | None:
    """The fullest realized spine run's manifest (``realize_summary`` — the
    provenance-authenticity dial), tagged with ``_run``; graceful None."""
    found = _fullest_spine()
    if not found:
        return None
    p, man = found
    man["_run"] = f"{p.parent.parent.name}/{p.parent.name}"
    return man


def spine_run() -> dict | None:
    """The fullest deterministic DDL spine run's REALIZED subgraphs, per template.

    Reads ``build/spine_*/<run>/ddl_statements.parquet`` (the realize expansion:
    junction/star/normalized satellites + their intra-subgraph FK edges). This is what
    replaced the retired family-complex FK wiring: cross-entity edges are the constructs
    web's to EARN (Convert 2) — the deterministic spine's edges live INSIDE a template's
    subgraph. Graceful None when no spine run exists.

    Returns ``{"run": <name>, "by_template": {tid: {"tables": {name: [cols]},
    "fks": [{src_table, src_col, dst_table, dst_col}]}}}``.
    """
    import json as _json
    found = _fullest_spine()
    if not found:
        return None
    run_dir = found[0].parent
    ddl = run_dir / "ddl_statements.parquet"
    if not ddl.exists():
        return None
    try:
        import pyarrow.parquet as _pq
        rows = _pq.read_table(
            ddl, columns=["template_id", "table_name", "columns_json", "fks_json"]).to_pylist()
    except Exception:  # noqa: BLE001 — the subgraph web is optional enrichment
        return None
    by_t: dict[str, dict] = {}
    for r in rows:
        e = by_t.setdefault(r["template_id"], {"tables": {}, "fks": []})
        e["tables"][r["table_name"]] = _json.loads(r.get("columns_json") or "[]")
        for fk in _json.loads(r.get("fks_json") or "[]"):
            fk.setdefault("src_table", r["table_name"])   # implicit in the parquet row
            e["fks"].append(fk)
    return {"run": f"{run_dir.parent.name}/{run_dir.name}", "by_template": by_t}


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


def sdg_constructs() -> "dict | None":
    """The generated relational product, VERBATIM (RH ruling: the lineup audits what
    `just metaflow` offers up — wrapping names is obfuscation). One entry per unique table
    name across all constructs; FK targets by real name; provenance = construct pids."""
    import json as _json
    root = Path("/raid/checkpoints/aegir-artifacts/sdg-corpora/corpus")
    cdir = root / "constructs"
    if not cdir.exists():
        return None
    tables: dict = {}
    views: dict = {}
    for cj in sorted(cdir.glob("*.json")):
        try:
            d = _json.loads(cj.read_text())
        except Exception:  # noqa: BLE001
            continue
        pid = cj.stem
        plans = (d.get("key_plan") or {}).get("plans") or {}
        pk_kind = {v.get("table"): v.get("kind") for v in plans.values() if isinstance(v, dict)}
        for t in d.get("tables") or []:
            name = t.get("name")
            if not name:
                continue
            e = tables.setdefault(name, {"columns": [], "pk": t.get("pk"), "fks": [],
                                         "constructs": [], "pk_kind": pk_kind.get(name)})
            if not e["columns"]:
                e["columns"] = [c.get("name") for c in t.get("columns") or []]
                e["fks"] = t.get("fks") or []
            e["constructs"].append(pid)
        for v in d.get("views") or []:
            vn = v.get("name") if isinstance(v, dict) else None
            if vn:
                views.setdefault(vn, {"sql": (v.get("sql") or "")[:400], "construct": pid})
    if not tables:
        return None
    return {"tables": tables, "views": views, "n_constructs": len(list(cdir.glob("*.json")))}


def released_ddl() -> "dict | None":
    """The RELEASED DDL spine — the newest run under ``corpora/ddl/`` (the SHARE record;
    v0.4 = 623 tables lowered from the 623-template released catalog). Feeds the
    current-root relational surface. Graceful None without the submodule."""
    import json as _json
    runs = sorted((REPO / "corpora" / "ddl").glob("*/manifest.json"))
    if not runs:
        return None
    run_dir = runs[-1].parent
    ddl = run_dir / "ddl_statements.parquet"
    if not ddl.exists():
        return None
    try:
        import pyarrow.parquet as _pq
        rows = _pq.read_table(ddl).to_pylist()
    except Exception:  # noqa: BLE001
        return None
    man: dict = {}
    try:
        man = _json.loads((run_dir / "manifest.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    return {"run": run_dir.name, "manifest": man, "rows": rows}


def releases() -> "list[dict]":
    """The RELEASED corpus generations — the sdg-corpora dataset CARDs (the SHARE record),
    oldest→newest. These are the #147 cut-point anchors: the latest release projects to
    ``current``, past releases to ``archive``. Each: ``{version, path, body}`` (the card's
    markdown with its HF YAML header stripped). Graceful ``[]`` without the submodule."""
    import re as _re
    out: list[dict] = []
    for p in sorted((REPO / "corpora" / "corpus").glob("CARD_v*.md")):
        m = _re.match(r"CARD_v([\d.]+)\.md", p.name)
        if not m:
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                body = parts[2].strip()
        out.append({"version": m.group(1), "path": str(p.relative_to(REPO)), "body": body})
    out.sort(key=lambda r: tuple(int(x) for x in r["version"].split(".")))
    return out


def strategy_state() -> "dict | None":
    """The declared strategy (the sdg-strategy submodule's CURRENT manifest) + drift-lite."""
    try:
        from aegir.strategy.manifest import declared, submodule_commit
        man = declared()
        if not man:
            return None
        return {"manifest": man, "commit": submodule_commit()}
    except Exception:  # noqa: BLE001
        return None

"""provenance — value-grain lineage over aegir_hx: the governance-layer answer to "is this
value/noun lineage-backed?"

RH 2026-07-18: the scan-side token bag (a set scraped from build/gittables/*.json at import time)
was a stopgap that comments accidentally elevated to architecture. The PATTERN is Atlas + OpenLineage
on the aegir_hx AGE graph (borrowed from Gaius, enhanced here): harvest jobs EMIT OL RunEvents
(GitTables snapshot dataset → run → pool datasets) and every pooled value lands as token-grain nodes
``(:Token)-[:TOKEN_OF]->(:Dataset)``, so admission questions are answered BY THE GRAPH. The local
JSON artifacts remain the rebuildable truth (Atlas is never a master — [[obsolescence_by_obviation]]);
``backfill()`` re-projects them into the graph at any time.

Token semantics mirror the scan's historical matching exactly (full lowercased value + component
words >= 2 chars) so the graph answer is never stricter than the bag it replaces.

    uv run --no-sync python -m aegir.governance.provenance          # backfill both pool artifacts
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from aegir.governance import graph as G
from aegir.governance import ol

REPO = Path(__file__).resolve().parent.parent.parent.parent
_GT_DIR = REPO / "build" / "gittables"
_CHUNK = 400          # tokens per UNWIND statement (statement-size bound)


def value_tokens(values) -> "set[str]":
    """Full lowercased values + >= 2-char word fragments — the scan-compatible token expansion."""
    toks: "set[str]" = set()
    for pair in values or []:
        v = str(pair[0] if isinstance(pair, (list, tuple)) else pair).lower()
        toks.add(v)
        toks.update(w for w in re.findall(r"[a-z0-9&]+", v) if len(w) >= 2)
    return toks


def _merge_tokens(conn, dataset_qn: str, tokens: "set[str]") -> int:
    """Batched idempotent MERGE of token nodes + TOKEN_OF edges onto an already-merged dataset."""
    toks = sorted(tokens)
    for i in range(0, len(toks), _CHUNK):
        lits = ", ".join(G._lit(t) for t in toks[i:i + _CHUNK])
        G.run(conn,
              f"MATCH (d:Dataset {{qualifiedName: {G._lit(dataset_qn)}}}) "
              f"UNWIND [{lits}] AS t MERGE (tok:Token {{t: t}}) "
              f"MERGE (tok)-[:TOKEN_OF]->(d) RETURN count(*)")
    return len(toks)


def emit_pool_lineage(job_name: str, run_key: str, snapshot: str,
                      pools: "dict[str, list]", run_props: "dict | None" = None) -> dict:
    """One OL RunEvent for a harvest job (gittables snapshot → run → one dataset per pool) + the
    token-grain projection. Idempotent: runId = uuid5(job, run_key); everything MERGEs."""
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ol.PRODUCER}/{job_name}/{run_key}"))
    outputs = [{"namespace": "pools", "name": name,
                "facets": {"pool": {"n_values": len(vals), **(run_props or {})}}}
               for name, vals in sorted(pools.items())]
    event = {"eventType": "COMPLETE",
             "eventTime": datetime.now(timezone.utc).isoformat(),
             "producer": ol.PRODUCER,
             "run": {"runId": run_id},
             "job": {"namespace": "aegir", "name": job_name},
             "inputs": [{"namespace": "gittables", "name": snapshot}],
             "outputs": outputs}
    res = ol.ingest_run_event(event)
    n_tok = 0
    with G.connect() as conn:
        for name, vals in sorted(pools.items()):
            n_tok += _merge_tokens(conn, f"pools:{name}", value_tokens(vals))
    return {**res, "tokens": n_tok}


def provenance_backed(token: str) -> "bool | None":
    """Graph-answered admission: is this lowercased token lineage-backed by any pool dataset?
    None = graph unavailable (caller decides the fallback and MUST say which source answered)."""
    try:
        with G.connect() as conn:
            n = G.scalar(conn, f"MATCH (tok:Token {{t: {G._lit(token.strip().lower())}}}) "
                               f"RETURN count(tok)")
            return bool(n)
    except Exception:  # noqa: BLE001 — down graph is a legitimate offline state, not an error here
        return None


def emit_corpus_lineage(run_dir: "Path | str") -> dict:
    """The corpus MAIN-PATH emission (task #23 — restores what the SemanticCorpusFlow→SdgCorporaFlow
    succession dropped): one OL run per corpus flow run, at CONSTRUCT grain (the current truth — NOT the
    template-era project_atlas_ddl footprint, which projects the vestigial catalog shape).

    Declares the provenance triple from the run-zettel — (window, strategy_id, code commit) + per-stage
    stage_keys — as run-level identity; inputs are the strategy dataset + the pool datasets the rows core
    reads (as published at emission); outputs are the run's ontology/constructs/chapters datasets plus one
    dataset per construct (n_tables/n_views/stage_key facets), PART_OF-linked for walkability. Idempotent.
    """
    run_dir = Path(run_dir)
    zs = sorted((run_dir / "runs").glob("*.json"))
    if not zs:
        raise FileNotFoundError(f"no run-zettel under {run_dir}/runs")
    z = json.loads(zs[-1].read_text())
    versions, rid = z.get("versions", {}), z["id"]
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}

    inputs = [{"namespace": "strategy", "name": versions.get("strategy_id", "undeclared"),
               "facets": {"strategy": {"commit": versions.get("strategy_commit", "")}}}]
    vp = _GT_DIR / "value_profiles.json"
    if vp.exists():
        inputs += [{"namespace": "pools", "name": f"semantic/{st}"} for st in sorted(json.loads(vp.read_text()))]
    dp = _GT_DIR / "differentia_profiles.json"
    if dp.exists():
        inputs += [{"namespace": "pools", "name": f"differentia/{c}"}
                   for c in sorted(json.loads(dp.read_text()).get("pools") or {})]

    constructs = sorted((run_dir / "constructs").glob("*.json"))
    outputs = [
        {"namespace": "corpus", "name": f"{rid}/ontology",
         "facets": {"ontology": {k: v for k, v in (metrics.get("structure") or {}).items()
                                 if isinstance(v, (int, float, str, bool))}}},
        {"namespace": "corpus", "name": f"{rid}/constructs", "facets": {"corpus": {"n": len(constructs)}}},
        {"namespace": "corpus", "name": f"{rid}/chapters",
         "facets": {"corpus": {"n": metrics.get("n_chapters", 0),
                               "prose_chars_median": metrics.get("prose_chars_median", 0)}}},
    ]
    for cf in constructs:
        c = json.loads(cf.read_text())
        outputs.append({"namespace": "corpus", "name": f"{rid}/construct/{cf.stem}",
                        "facets": {"construct": {"n_tables": len(c.get("tables", [])),
                                                 "n_views": len(c.get("views", [])),
                                                 "stage_key": c.get("stage_key", "")}}})

    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ol.PRODUCER}/sdg_corpora_flow/{z.get('metaflow_run_id', rid)}"))
    event = {"eventType": "COMPLETE", "eventTime": datetime.now(timezone.utc).isoformat(),
             "producer": ol.PRODUCER, "run": {"runId": run_id},
             "job": {"namespace": "aegir", "name": "sdg_corpora_flow"},
             "inputs": inputs, "outputs": outputs}
    res = ol.ingest_run_event(event)
    with G.connect() as conn:
        G.merge_node(conn, "Run", {"run_id": run_id},
                     {k: str(v) for k, v in versions.items() if not isinstance(v, dict)})
        for cf in constructs:                             # construct → collection membership (walkability)
            G.merge_edge(conn, "Dataset", {"qualifiedName": f"corpus:{rid}/construct/{cf.stem}"},
                         "PART_OF", "Dataset", {"qualifiedName": f"corpus:{rid}/constructs"})
    return {**res, "zettel": rid}


def backfill() -> dict:
    """Re-project BOTH local pool artifacts into the graph (truth → Atlas, rebuildable any time)."""
    out = {}
    vp = _GT_DIR / "value_profiles.json"
    if vp.exists():
        d = json.loads(vp.read_text())
        lin = json.loads((_GT_DIR / "lineage.json").read_text()) if (_GT_DIR / "lineage.json").exists() else {}
        snap = lin.get("snapshot_sha1", "unknown")
        seg = lin.get("segmentation_version", "seg")
        out["gittables_value_harvest"] = emit_pool_lineage(
            "gittables_value_harvest", f"{snap}:{seg}", snap,
            {f"semantic/{st}": rec.get("values", []) for st, rec in d.items()},
            {"segmentation": seg})
    dp = _GT_DIR / "differentia_profiles.json"
    if dp.exists():
        d = json.loads(dp.read_text())
        meta = d.get("_meta", {})
        snap = meta.get("snapshot_sha1", "unknown")
        harv = meta.get("harvester", "harvester")
        out["differentia_harvest"] = emit_pool_lineage(
            "differentia_harvest", f"{snap}:{harv}", snap,
            {f"differentia/{col}": rec.get("values", []) for col, rec in (d.get("pools") or {}).items()},
            {"harvester": harv})
    return out


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:                                # corpus-run backfill: <run_dir>
        r = emit_corpus_lineage(_sys.argv[1])
        print(f"sdg_corpora_flow[{r['zettel']}]: run {r['run_id'][:8]}… · "
              f"{r['inputs']} inputs · {r['outputs']} datasets · {r['columns']} column edges")
    else:
        for job, res in backfill().items():
            print(f"{job}: run {res['run_id'][:8]}… · {res['outputs']} pool datasets · {res['tokens']} tokens")

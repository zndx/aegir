"""Extended OpenLineage variant over aegir_hx.

Ingests OpenLineage RunEvents into the aegir_hx graph (Marquez-compatible shape:
``(Dataset)-[:INPUT_TO]->(Run)-[:OUTPUTS]->(Dataset)``, ``(Job)-[:EXECUTES]->(Run)``)
and serves lineage queries. The *extension*: OL dataset names may be aegir entity
qualifiedNames (``chapter:…``, ``topic:…``), so lineage rides the same graph as the
Atlas entities + classifications. Caddy fronts this alongside the real Atlas v2
service; Marquez can be pointed at the same events to verify OL conformance.
"""
from __future__ import annotations

from aegir.governance import graph as G

PRODUCER = "https://github.com/zndx/aegir"


def _ds_qn(ds: dict) -> str:
    return f"{ds.get('namespace', 'aegir')}:{ds.get('name', '')}"


def _ds_props(ds: dict) -> dict:
    p = {"namespace": ds.get("namespace", "aegir"), "name": ds.get("name", "")}
    # flatten scalar facet values (e.g. our reGrounding facet) onto the node
    for fk, fv in (ds.get("facets") or {}).items():
        if isinstance(fv, dict):
            for k, v in fv.items():
                if isinstance(v, (int, float, str, bool)) and not k.startswith("_"):
                    p[f"{fk}_{k}"] = v
    return p


def _ingest_column_lineage(conn, ds: dict, out_qn: str) -> int:
    """Project an OpenLineage ``columnLineage`` facet into column-level graph edges.

    Creates ``(Column)`` nodes keyed ``dataset:#col``, ``(Dataset)-[:HAS_COLUMN]->(Column)``
    membership, and ``(Column)-[:DERIVES_FROM]->(Column)`` derivation edges (output→input),
    so view-column ← base-column lineage rides the same aegir_hx graph as the entities.
    Tolerant of the facet's ``fields`` being a map (output_col → spec) or a list; no-op when
    the facet is absent. Returns the number of derivation edges written.
    """
    facet = (ds.get("facets") or {}).get("columnLineage") or {}
    fields = facet.get("fields") or {}
    items = (fields.items() if isinstance(fields, dict)
             else [(f.get("name") or f.get("field"), f) for f in fields])
    out_ns = ds.get("namespace", "aegir")
    n = 0
    for out_col, spec in items:
        if not out_col or not isinstance(spec, dict):
            continue
        out_col_qn = f"{out_qn}#{out_col}"
        G.merge_node(conn, "Column", {"qualifiedName": out_col_qn},
                     {"dataset": out_qn, "column": out_col})
        G.merge_edge(conn, "Dataset", {"qualifiedName": out_qn}, "HAS_COLUMN",
                     "Column", {"qualifiedName": out_col_qn})
        for inf in (spec.get("inputFields") or []):
            in_qn = f"{inf.get('namespace', out_ns)}:{inf.get('name', '')}"
            in_col = inf.get("field") or inf.get("inputField") or inf.get("column") or ""
            in_col_qn = f"{in_qn}#{in_col}"
            G.merge_node(conn, "Column", {"qualifiedName": in_col_qn},
                         {"dataset": in_qn, "column": in_col})
            G.merge_edge(conn, "Dataset", {"qualifiedName": in_qn}, "HAS_COLUMN",
                         "Column", {"qualifiedName": in_col_qn})
            G.merge_edge(conn, "Column", {"qualifiedName": out_col_qn}, "DERIVES_FROM",
                         "Column", {"qualifiedName": in_col_qn})
            n += 1
    return n


def ingest_run_event(event: dict) -> dict:
    """Project an OpenLineage RunEvent into aegir_hx. Idempotent (MERGE)."""
    run = event.get("run") or {}
    job = event.get("job") or {}
    run_id = run.get("runId") or run.get("run_id")
    if not run_id:
        raise ValueError("RunEvent missing run.runId")
    job_name = job.get("name", "unknown")
    job_ns = job.get("namespace", "aegir")
    state = event.get("eventType", "OTHER")

    with G.connect() as conn:
        G.merge_node(conn, "Run", {"run_id": run_id}, {
            "eventType": state, "eventTime": event.get("eventTime", ""),
            "job": job_name, "namespace": job_ns})
        G.merge_node(conn, "Job", {"namespace": job_ns, "name": job_name})
        G.merge_edge(conn, "Job", {"namespace": job_ns, "name": job_name}, "EXECUTES",
                     "Run", {"run_id": run_id})
        n_in = n_out = n_cols = 0
        for ds in event.get("inputs") or []:
            qn = _ds_qn(ds)
            G.merge_node(conn, "Dataset", {"qualifiedName": qn}, _ds_props(ds))
            G.merge_edge(conn, "Dataset", {"qualifiedName": qn}, "INPUT_TO", "Run", {"run_id": run_id})
            n_in += 1
        for ds in event.get("outputs") or []:
            qn = _ds_qn(ds)
            G.merge_node(conn, "Dataset", {"qualifiedName": qn}, _ds_props(ds))
            G.merge_edge(conn, "Run", {"run_id": run_id}, "OUTPUTS", "Dataset", {"qualifiedName": qn})
            n_out += 1
            n_cols += _ingest_column_lineage(conn, ds, qn)
    try:
        # keep the Lineage panel's materialized views near-live (throttled; never raises)
        from aegir.governance import lineage_views as _LV
        _LV.refresh_throttled_async()
    except Exception:  # noqa: BLE001
        pass
    return {"run_id": run_id, "eventType": state, "inputs": n_in, "outputs": n_out, "columns": n_cols}


def run_lineage(run_id: str) -> dict:
    """A run with its input/output datasets (the OL run-graph slice)."""
    rid = G._lit(run_id)
    with G.connect() as conn:
        run = G.run(conn, f"MATCH (r:Run {{run_id: {rid}}}) RETURN r")
        ins = G.run(conn, f"MATCH (d:Dataset)-[:INPUT_TO]->(:Run {{run_id: {rid}}}) RETURN d.qualifiedName")
        outs = G.run(conn, f"MATCH (:Run {{run_id: {rid}}})-[:OUTPUTS]->(d:Dataset) RETURN d.qualifiedName")
    return {"run_id": run_id, "run": run[0] if run else None, "inputs": ins, "outputs": outs}


def dataset_lineage(namespace: str, name: str, max_depth: int = 5) -> dict:
    """Upstream + downstream datasets reachable through runs (OL graph traversal)."""
    qn = G._lit(f"{namespace}:{name}")
    with G.connect() as conn:
        up = G.run(conn,
            f"MATCH (src:Dataset)-[:INPUT_TO]->(:Run)-[:OUTPUTS*0..{max_depth}]->(t:Dataset {{qualifiedName: {qn}}}) "
            f"RETURN DISTINCT src.qualifiedName")
        down = G.run(conn,
            f"MATCH (s:Dataset {{qualifiedName: {qn}}})-[:INPUT_TO]->(:Run)-[:OUTPUTS]->(t:Dataset) "
            f"RETURN DISTINCT t.qualifiedName")
    return {"dataset": f"{namespace}:{name}", "upstream": up, "downstream": down}


def make_router():
    """FastAPI router for the extended OL surface (mounted by the gateway)."""
    from fastapi import APIRouter

    router = APIRouter(tags=["openlineage"])

    @router.post("/api/v1/lineage")          # OpenLineage standard RunEvent ingestion
    async def post_lineage(event: dict):
        return ingest_run_event(event)

    @router.get("/api/lineage/runs/{run_id}")
    async def get_run(run_id: str):
        return run_lineage(run_id)

    @router.get("/api/lineage/datasets/{namespace}/{name}")
    async def get_dataset(namespace: str, name: str):
        return dataset_lineage(namespace, name)

    return router

"""marquez — a Marquez-compatible read API over aegir's governed provenance (RH 2026-07-23).

The OL reference-implementation UI (marquez-web) speaks the Marquez REST API. This router
serves the subset that UI needs — namespaces, jobs, runs, datasets, and the lineage
graph — directly from the aegir_hx projection (Run/Job/Dataset nodes + EXECUTES/
INPUT_TO/OUTPUTS edges, fed by every OL emitter: the flow's own events AND kvasir's
native emission). Mount under /api/v1 and point marquez-web at the gateway "at whim" —
Atlas remains the governance authority; this is the OL-ecosystem read surface.

Marquez nodeId grammar: ``job:<namespace>:<name>`` · ``dataset:<namespace>:<name>``.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aegir.governance import graph as G

router = APIRouter(prefix="/api/v1", tags=["marquez"])


def _rows(cypher: str) -> list:
    with G.connect() as conn:
        return G.run(conn, cypher)


@router.get("/namespaces")
def namespaces() -> dict:
    rows = _rows("MATCH (j:Job) RETURN DISTINCT j.namespace")
    return {"namespaces": [{"name": str(r), "isHidden": False}
                           for r in sorted({str(x) for x in rows if x})]}


@router.get("/namespaces/{ns}/jobs")
def jobs(ns: str) -> dict:
    nsl = G._lit(ns)
    rows = _rows(f"MATCH (j:Job {{namespace: {nsl}}}) RETURN j.name")
    out = []
    for name in sorted({str(r) for r in rows if r}):
        nl = G._lit(name)
        runs = _rows(f"MATCH (:Job {{namespace: {nsl}, name: {nl}}})-[:EXECUTES]->(r:Run) "
                     "RETURN {id: r.run_id, state: r.eventType, at: r.eventTime}")
        latest = sorted(runs, key=lambda r: str(r.get("at", "")))[-1] if runs else None
        out.append({"id": {"namespace": ns, "name": name}, "name": name, "namespace": ns,
                    "type": "BATCH", "simple_name": name,
                    "latestRun": ({"id": latest.get("id"), "state": latest.get("state"),
                                   "createdAt": latest.get("at")} if latest else None)})
    return {"jobs": out, "totalCount": len(out)}


@router.get("/namespaces/{ns}/jobs/{job}/runs")
def job_runs(ns: str, job: str) -> dict:
    rows = _rows(f"MATCH (:Job {{namespace: {G._lit(ns)}, name: {G._lit(job)}}})"
                 "-[:EXECUTES]->(r:Run) "
                 "RETURN {id: r.run_id, state: r.eventType, at: r.eventTime}")
    return {"runs": [{"id": r.get("id"), "state": r.get("state"),
                      "createdAt": r.get("at"), "jobVersion": {"namespace": ns, "name": job}}
                     for r in sorted(rows, key=lambda x: str(x.get("at", "")), reverse=True)],
            "totalCount": len(rows)}


@router.get("/namespaces/{ns}/datasets")
def datasets(ns: str) -> dict:
    rows = _rows("MATCH (d:Dataset) RETURN d.qualifiedName")
    qns = sorted({str(r) for r in rows if r and str(r).startswith(f"{ns}:")})
    return {"datasets": [{"id": {"namespace": ns, "name": q.split(":", 1)[1]},
                          "name": q.split(":", 1)[1], "namespace": ns, "type": "DB_TABLE"}
                         for q in qns],
            "totalCount": len(qns)}


@router.get("/lineage")
def lineage(nodeId: str, depth: int = 2) -> dict:
    """The marquez-web core call: the bipartite job/dataset graph around a node."""
    try:
        kind, ns, name = nodeId.split(":", 2)
    except ValueError as e:
        raise HTTPException(400, "nodeId must be kind:namespace:name") from e
    depth = max(1, min(depth, 5))

    graph: dict[str, dict] = {}

    def _jid(ns_: str, n_: str) -> str:
        return f"job:{ns_}:{n_}"

    def _did(qn: str) -> str:
        ns_, n_ = (qn.split(":", 1) + [""])[:2]
        return f"dataset:{ns_}:{n_}"

    def _node(nid: str, ntype: str, data: dict) -> dict:
        return graph.setdefault(nid, {"id": nid, "type": ntype, "data": data,
                                      "inEdges": [], "outEdges": []})

    def _edge(src: str, dst: str) -> None:
        e = {"origin": src, "destination": dst}
        if e not in graph[src]["outEdges"]:
            graph[src]["outEdges"].append(e)
        if e not in graph[dst]["inEdges"]:
            graph[dst]["inEdges"].append(e)

    def _expand_job(ns_: str, n_: str, d_: int) -> None:
        jid = _jid(ns_, n_)
        _node(jid, "JOB", {"id": {"namespace": ns_, "name": n_}, "name": n_,
                           "namespace": ns_, "type": "BATCH"})
        nsl, nl = G._lit(ns_), G._lit(n_)
        ins = _rows(f"MATCH (d:Dataset)-[:INPUT_TO]->(:Run)<-[:EXECUTES]-"
                    f"(:Job {{namespace: {nsl}, name: {nl}}}) RETURN DISTINCT d.qualifiedName")
        outs = _rows(f"MATCH (:Job {{namespace: {nsl}, name: {nl}}})-[:EXECUTES]->"
                     "(:Run)-[:OUTPUTS]->(d:Dataset) RETURN DISTINCT d.qualifiedName")
        for qn in {str(x) for x in ins if x}:
            did = _did(qn)
            _node(did, "DATASET", {"id": {"namespace": qn.split(":", 1)[0],
                                          "name": qn.split(":", 1)[-1]},
                                   "name": qn.split(":", 1)[-1],
                                   "namespace": qn.split(":", 1)[0], "type": "DB_TABLE"})
            _edge(did, jid)
            if d_ > 1:
                _expand_dataset(qn, d_ - 1)
        for qn in {str(x) for x in outs if x}:
            did = _did(qn)
            _node(did, "DATASET", {"id": {"namespace": qn.split(":", 1)[0],
                                          "name": qn.split(":", 1)[-1]},
                                   "name": qn.split(":", 1)[-1],
                                   "namespace": qn.split(":", 1)[0], "type": "DB_TABLE"})
            _edge(jid, did)
            if d_ > 1:
                _expand_dataset(qn, d_ - 1)

    def _expand_dataset(qn: str, d_: int) -> None:
        did = _did(qn)
        _node(did, "DATASET", {"id": {"namespace": qn.split(":", 1)[0],
                                      "name": qn.split(":", 1)[-1]},
                               "name": qn.split(":", 1)[-1],
                               "namespace": qn.split(":", 1)[0], "type": "DB_TABLE"})
        qnl = G._lit(qn)
        producers = _rows(f"MATCH (j:Job)-[:EXECUTES]->(:Run)-[:OUTPUTS]->"
                          f"(:Dataset {{qualifiedName: {qnl}}}) "
                          "RETURN {ns: j.namespace, n: j.name}")
        consumers = _rows(f"MATCH (:Dataset {{qualifiedName: {qnl}}})-[:INPUT_TO]->"
                          "(:Run)<-[:EXECUTES]-(j:Job) RETURN {ns: j.namespace, n: j.name}")
        for r in producers:
            jid = _jid(str(r.get("ns")), str(r.get("n")))
            _node(jid, "JOB", {"id": {"namespace": r.get("ns"), "name": r.get("n")},
                               "name": r.get("n"), "namespace": r.get("ns"), "type": "BATCH"})
            _edge(jid, did)
            if d_ > 1:
                _expand_job(str(r.get("ns")), str(r.get("n")), d_ - 1)
        for r in consumers:
            jid = _jid(str(r.get("ns")), str(r.get("n")))
            _node(jid, "JOB", {"id": {"namespace": r.get("ns"), "name": r.get("n")},
                               "name": r.get("n"), "namespace": r.get("ns"), "type": "BATCH"})
            _edge(did, jid)
            if d_ > 1:
                _expand_job(str(r.get("ns")), str(r.get("n")), d_ - 1)

    if kind == "job":
        _expand_job(ns, name, depth)
    elif kind == "dataset":
        _expand_dataset(f"{ns}:{name}", depth)
    else:
        raise HTTPException(400, f"unknown node kind {kind!r}")
    return {"graph": list(graph.values())}

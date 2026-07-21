"""Aegir gateway — FastAPI app serving the leaderboard UI + API.

Air-gap posture: read-only. No database writes, no external HTTP calls, no
telemetry. Reads run sidecars from ``config.runs.dir`` (JSON on disk, written
by ``aegir.utils.runs.RunArtifacts``) and the DBpedia GT mappings from
``config.data.gittables_signals_dir``.

Endpoints:

    GET /api/health                   — liveness + versions + dataset reachable
    GET /api/leaderboard              — aggregate row per run; sortable client-side
    GET /api/runs/{run_id}            — full metadata + metrics for one run
    GET /api/runs/{run_id}/plot/{name}
                                      — static Bokeh JSON (consumable by
                                        ``Bokeh.embed.embed_item`` in browser)
    GET /api/ontology/dbpedia-types   — the 120 DBpedia labels from GT JSON
    GET /api/classifications/latest   — per-class F1 breakdown (from latest run)
    GET /*                            — static React bundle (ui/dist)

The static-bundle handler is mounted last so all ``/api/*`` paths win; a
missing ``ui/dist`` directory returns a helpful 404 body pointing the user
at ``just ui-build``.
"""

from __future__ import annotations

import json
import logging
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aegir.config import Config, load_config
from aegir.utils.runs import (
    iter_runs,
    load_run_detail,
    load_run_summary,
)

_log = logging.getLogger("aegir.gateway")


# ── App factory ─────────────────────────────────────────────────


def create_app(cfg: Config | None = None) -> FastAPI:
    """Build the FastAPI app bound to the given config.

    Config is resolved once at import time and cached on ``app.state.cfg``;
    re-create the app (e.g. in tests) to pick up env-var changes.
    """
    cfg = cfg or load_config()
    app = FastAPI(
        title="Aegir Gateway",
        version=_pkg_version(),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.cfg = cfg

    _register_api_routes(app)
    # Extended OpenLineage variant over aegir_hx (/api/v1/lineage, /api/lineage/*).
    from aegir.governance.ol import make_router as _ol_router
    app.include_router(_ol_router())
    _mount_static_bundle(app, cfg)
    _wire_lineup_upkeep(app, cfg)
    return app


def _wire_lineup_upkeep(app: FastAPI, cfg: Config) -> None:
    """Fold the lineup KB upkeep into the gateway (one app, no extra process): on startup
    register the pg_cron jobs (idempotent) and run the ScheduledTaskProcessor that consumes
    ``scheduled_tasks`` → ``aegir.lineup.maintain``. Gated by ``AEGIR_LINEUP_UPKEEP=1`` (set
    in the devenv gateway exec) so tests / TestClient don't touch the DB or scheduler. Every
    step is guarded — a DB outage never stops the gateway serving."""
    import os
    if os.environ.get("AEGIR_LINEUP_UPKEEP") != "1":
        return
    state: dict = {}

    @app.on_event("startup")
    async def _start_upkeep() -> None:
        import asyncio
        try:
            from aegir.lineup import cron
            await asyncio.to_thread(cron.install, cfg.db.url)
        except Exception as e:                              # noqa: BLE001
            _log.warning("lineup cron install skipped: %s", e)
        try:
            from aegir.lineup.processor import ScheduledTaskProcessor
            p = ScheduledTaskProcessor(cfg.db.url)
            state["proc"], state["task"] = p, asyncio.create_task(p.run())
            _log.info("lineup upkeep processor started")
        except Exception as e:                              # noqa: BLE001
            _log.warning("lineup processor not started: %s", e)

    @app.on_event("shutdown")
    async def _stop_upkeep() -> None:
        if state.get("proc"):
            state["proc"].stop()
        if state.get("task"):
            state["task"].cancel()


def _pkg_version() -> str:
    try:
        return version("aegir")
    except PackageNotFoundError:
        return "0.0.0-dev"


# ── /api/health ─────────────────────────────────────────────────


def _register_api_routes(app: FastAPI) -> None:

    @app.get("/api/health")
    def health() -> dict:
        cfg: Config = app.state.cfg
        runs_dir = Path(cfg.runs.dir)
        gittables_dir = Path(cfg.data.gittables_signals_dir)
        runs_count = len(iter_runs(runs_dir))
        return {
            "ok": True,
            "version": _pkg_version(),
            "runs_dir": str(runs_dir),
            "runs_dir_exists": runs_dir.exists(),
            "runs_count": runs_count,
            "gittables_dir": str(gittables_dir),
            "gittables_dir_exists": gittables_dir.exists(),
            "gateway_port": cfg.gateway.port,
        }

    # ── /api/kb — the lineup projection (build/dev/{current,scratch,archive}) ──
    # Thin: delegate to aegir.lineup readers. Read-only; reflects the latest
    # `just kb-build`. Returns 404 with a hint if the projection isn't built.

    @app.get("/api/kb/index")
    def kb_index() -> dict:
        import json as _json

        from aegir.lineup import sources as _S
        p = _S.kb_dir() / "index.json"
        if not p.exists():
            raise HTTPException(404, "KB projection not built — run `just kb-build`")
        return _json.loads(p.read_text())

    @app.post("/api/kb/verify-path")
    def kb_verify_path(payload: dict) -> dict:
        """Path-aware HermiT step (RH 2026-07-21): verify the lineup's axiom universe on the
        fly — the adoption use-case (user-provided ontologies, incremental spec adaptation in
        scratch) verified at the speed of browsing."""
        trail = payload.get("trail") or []
        if not trail:
            raise HTTPException(400, "empty trail")
        try:
            from aegir.lineup.verify_path import verify
            r = verify(trail, budget_s=int(payload.get("budget", 180)))
        except KeyError as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"verification unavailable: {e}") from e
        return {"note_id": f"path/{r['path_id']}", "verdict": r["verdict"]}

    @app.get("/api/kb/note/{note_id:path}")
    def kb_note(note_id: str, root: str | None = None) -> dict:
        # provenance/<vid> ids are synthetic (the live AGE graph, not a projected file) — resolve them to a
        # provenance-kind note the React panel renders as a ReactFlow ego-graph centered on that node.
        if re.match(r"ontology/(?:foreign/\w+|self-census)/class/", note_id):
            # synthetic FOREIGN CLASS panel (RH 2026-07-21: members ARE the primary articles):
            # resolved from the census fragment cache — every class of a censused ontology is
            # navigable, axiom links walk class-to-class, zero projection bloat.
            import json as _json
            from pathlib import Path as _P
            _m = re.match(r"ontology/(?:foreign/(\w+)|self-census)/class/(.+)$", note_id)
            tag, local = (_m.group(1) or "sdg"), _m.group(2)
            base_id = f"ontology/foreign/{tag}" if _m.group(1) else "ontology/self-census"
            frag_p = _P("build/foreign_fragments") / f"{tag}.json"
            if not frag_p.exists():
                try:
                    from aegir.lineup.verify_path import _extract_foreign
                    _extract_foreign(tag)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(503, f"fragment cache for {tag!r} unavailable: {e}") from e
            frag = _json.loads(frag_p.read_text())
            iri = next((k for k in list(frag["frames"]) + list(frag.get("equiv") or {})
                        if k.rsplit("/", 1)[-1].rsplit("#", 1)[-1] == local), None)
            frames = frag["frames"].get(iri or "", [])
            equivs = (frag.get("equiv") or {}).get(iri or "", [])
            # catalog maps, built once: head→tid, tid→(manchester, verbal), reverse USAGE,
            # and tid→SKOS concept local (the reverse of the concept panel's binding)
            cm = getattr(app.state, "class_maps", None)
            if cm is None:
                cm = {"head_tid": {}, "tid_art": {}, "usage": {}, "tid_concept": {}}
                try:
                    _cat = _json.loads(_P("src/aegir/ontology/catalog/catalog.json").read_text())
                    for _t in _cat.get("templates", []):
                        _mt = _t.get("manchester_template") or ""
                        _h = re.search(r"Class:\s*\{(\w+):Class\}", _mt)
                        if _h:
                            cm["head_tid"][_h.group(1)] = _t["template_id"]
                        cm["tid_art"][_t["template_id"]] = (
                            _mt, (_t.get("verbal_template") or ""))
                        for _f in set(re.findall(r"\{(\w+):Class\}", _mt)):
                            if not (_h and _f == _h.group(1)):
                                cm["usage"].setdefault(_f, []).append(_t["template_id"])
                    from aegir.ontology import domain_index as DI
                    for _k in DI.load_skos():
                        _loc2 = str(_k).rsplit("#", 1)[-1]
                        for _i, _ch in enumerate(_loc2):
                            if _ch == "_":
                                cm["tid_concept"].setdefault(_loc2[_i + 1:].lower(), _loc2)
                except Exception:  # noqa: BLE001
                    pass
                app.state.class_maps = cm
            cov = _json.loads(_P(f"build/{tag}_coverage.json").read_text())
            grp = next((g for g in cov.get("pathways", {}).get("groups", [])
                        if any(m.get("local") == local for m in g.get("members", []))), None)

            def _wl_iris(txt: str) -> str:
                def _one(m: "re.Match") -> str:
                    iri = m.group(1)
                    loc = iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                    if tag == "sdg" and "signals.zndx.org" not in iri:
                        return f"`{loc[:40]}`"          # backbone (BFO/CCO) — not our article
                    return f"[[{base_id}/class/{loc}|{loc[:40]}]]"
                return re.sub(r"<([^>]+)>", _one, txt)

            ax_parts = ([f"- EquivalentTo {_wl_iris(f)}" for f in equivs[:12]]
                        + [f"- SubClassOf {_wl_iris(f)}" for f in frames[:24]])
            ax_rows = "\n".join(ax_parts) or \
                      "- _(no extracted axioms — outside the renderer's construct family)_"
            tid2 = cm["head_tid"].get(local)
            catalog_md = ""
            if tid2:
                mt2, vt2 = cm["tid_art"].get(tid2, ("", ""))
                vfill = re.sub(r"\{(\w+)\}",
                               lambda m: re.sub(r"(?<!^)(?=[A-Z])", " ", m.group(1)).lower(), vt2)
                conc = cm["tid_concept"].get(tid2.lower())
                catalog_md = (
                    "**Defining template.** "
                    + f"[[ontology/term/{tid2}|{tid2}]]"
                    + (f" · SKOS concept [[lexicon/concept/{conc}|{conc}]]" if conc else "")
                    + "\n\n"
                    + (f"> {vfill}\n\n" if vfill else "")
                    + (f"```manchester\n{mt2.strip()}\n```\n\n" if mt2 else ""))
            used_by = cm["usage"].get(local, [])
            usage_md = ""
            if used_by:
                usage_md = ("**Referenced by** (as a restriction filler): " + " · ".join(
                    f"[[ontology/term/{u}|{u}]]" for u in sorted(used_by)[:10])
                    + (f" _…+{len(used_by) - 10}_" if len(used_by) > 10 else "") + "\n\n")
            grp_line = ""
            if grp:
                gid = f"{base_id}/group/{grp['slug']}"
                grp_line = f" (group [[{gid}|{grp.get('group', '')[:36]}]])"
            term_line = ""
            if False and tag == "sdg":
                # head → template_id (the move-2 lesson: heads are NOT always snake(template_id))
                hm = getattr(app.state, "head_map", None)
                if hm is None:
                    hm = {}
                    try:
                        _cat = _json.loads(_P("src/aegir/ontology/catalog/catalog.json").read_text())
                        for _t in _cat.get("templates", []):
                            _h = re.search(r"Class:\s*\{(\w+):Class\}",
                                           _t.get("manchester_template") or "")
                            if _h:
                                hm[_h.group(1)] = _t["template_id"]
                    except Exception:  # noqa: BLE001
                        pass
                    app.state.head_map = hm
                tid = hm.get(local)
                if tid and (_P("build/dev/scratch/ontology/term") / f"{tid}.md").exists():
                    term_line = (f"**Live term.** [[ontology/term/{tid}|{tid}]] — the catalog "
                                 "template surface for this class.\n\n")
            body = (f"**`{local}`** — a class of {tag.upper()}{grp_line}.\n\n"
                    + (catalog_md if tag == "sdg" else "")
                    + "**Axioms (realized OWL, extracted).**\n" + ax_rows + "\n\n"
                    + (usage_md if tag == "sdg" else "")
                    + f"_Source: build/foreign_fragments/{tag}.json (census extraction; "
                    "constructs outside the renderer degrade honestly)._\n")
            return {"id": note_id, "name": note_id, "title": f"{local[:52]} · {tag}",
                    "kind": "lexicon-construct", "data_product": "ontology", "root": "scratch",
                    "body": body,
                    "links": ([f"{base_id}/group/{grp['slug']}"] if grp else [])}
        if note_id.startswith("path/"):
            # synthetic path-run note (RH 2026-07-21): verification results join the lineup
            # WITHOUT waiting for a projection rebuild — resolved straight from build/path_runs.
            import json as _json
            from pathlib import Path as _P
            rp = _P("build/path_runs") / (note_id.split("/", 1)[1] + ".json")
            if not rp.exists():
                raise HTTPException(404, f"no path run {note_id!r}")
            r = _json.loads(rp.read_text())
            vd = r.get("verdict", {})
            ok = vd.get("consistent")
            body = (f"**Path verification** — HermiT over the axioms found before this panel "
                    f"in the lineup (armed signature backbone; isolated, never production).\n\n"
                    f"**Verdict: {'CONSISTENT ✓' if ok else 'INCONSISTENT ✗'}** · "
                    f"unsat {len(vd.get('unsat') or [])} · path `{r['path_id']}` · "
                    f"kasten `{r['version']}`\n\n"
                    + ("**Unsatisfiable.** " + " · ".join(f"`{u}`" for u in vd.get("unsat") or [])
                       + "\n\n" if vd.get("unsat") else "")
                    + "**Inputs (left of me).**\n"
                    + "\n".join(f"- [[{t}]]" for t in r.get("trail", [])) + "\n\n"
                    + f"_native templates {r['inputs'].get('native_templates', 0)} · foreign "
                    + ", ".join(f"{k}:{v}" for k, v in (r['inputs'].get('foreign') or {}).items())
                    + f" · {r['inputs'].get('foreign_axioms', 0)} foreign axioms · lineage "
                    + f"{r.get('lineage')}_\n")
            return {"id": note_id, "name": note_id, "title": f"path verify · {r['path_id'][:8]}",
                    "kind": "path-run", "data_product": "ontology", "root": "scratch",
                    "body": body, "links": r.get("trail", [])}
        if re.fullmatch(r"provenance/\d+", note_id):
            # SYNTHETIC ego ids are the vid form ONLY (provenance/<digits>) — provenance/sources
            # and provenance/source/* are PROJECTED notes and must fall through to the index
            # (RH 2026-07-21: the ego route was shadowing the Sources panels)
            rest = note_id.split("/", 1)[1]
            return {"id": note_id, "name": note_id, "title": "Provenance", "kind": "provenance",
                    "data_product": "training", "ego_focal": int(rest) if rest.isdigit() else None,
                    "body": "", "links": []}
        import json as _json

        from aegir.lineup import notes as _N
        from aegir.lineup import sources as _S
        kb = _S.kb_dir()
        idx_p = kb / "index.json"
        if not idx_p.exists():
            raise HTTPException(404, "KB projection not built — run `just kb-build`")
        idx = _json.loads(idx_p.read_text())
        # Roots are refs (git-style): the same surface id (e.g. lens/terms) exists per root —
        # current = the latest release kasten, scratch = trunk, archive = past. Prefer the
        # requested root's instance; fall back to any (cross-root trails keep resolving).
        hits = [n for n in idx["notes"] if n["id"] == note_id]
        if not hits:
            raise HTTPException(404, f"no KB note {note_id!r}")
        match = next((n for n in hits if n["root"] == root), hits[0])
        return _N.read_note(kb, match["relpath"])

    # ── /api/leaderboard ───────────────────────────────────────

    @app.get("/api/leaderboard")
    def leaderboard() -> dict:
        cfg: Config = app.state.cfg
        rows = []
        for run_dir in iter_runs(Path(cfg.runs.dir)):
            try:
                rows.append(load_run_summary(run_dir))
            except Exception as exc:  # skip broken run dirs rather than 500
                _log.warning("skip %s: %s", run_dir, exc)
        return {"rows": rows, "count": len(rows)}

    # ── /api/runs/{run_id} ─────────────────────────────────────

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        cfg: Config = app.state.cfg
        run_dir = Path(cfg.runs.dir) / run_id
        if not run_dir.exists() or not (run_dir / "metadata.json").exists():
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return load_run_detail(run_dir)

    # ── /api/viz/{app}/embed — live HoloViews viz bootstrap (lineup_app, runs_app, …) ────────
    @app.get("/api/viz/{app_name}/embed")
    def viz_embed(app_name: str, request: Request) -> PlainTextResponse:
        """Return the ``bokeh.embed.server_document`` bootstrap for a live HoloViews app served by the
        bokeh server behind this gateway's reverse proxy at ``/viz/*`` (lineup_app, runs_app, …). The
        React ``<PanelView>`` injects it; BokehJS loads from the bokeh server (renders GraphRenderer
        correctly, unlike npm ``@bokeh/bokehjs``). All query args (lens / run_id / plot / …) are
        forwarded to the app session; ``resources=None`` makes the autoload pull every resource from
        the (same-origin, proxied) bokeh server → no CDN, air-gapped."""
        if not _is_safe_plot_name(app_name):
            raise HTTPException(status_code=400, detail="invalid app name")
        from bokeh.embed import server_document
        args = dict(request.query_params) or None   # forward all query args to the bokeh session
        script = server_document(f"/viz/{app_name}", arguments=args, resources=None)
        return PlainTextResponse(script, media_type="text/html")

    # ── /api/provenance/ego — first-order neighborhood of a lineage node (instance-level, navigable) ──
    @app.get("/api/provenance/ego")
    def provenance_ego(focal: int | None = None) -> dict:
        """The 1-hop ego-graph of a provenance node (AGE internal id) for the lineup Provenance panel. No
        ``focal`` → a seed anchor (Dataset/Run/Chapter). Returns the focal + its neighbors (capped), each
        with a derived display name + edge type + direction; the React panel renders it and opens a node's
        OWN ego-graph on click (panel-trail). Degrades gracefully if aegir_hx is down."""
        from aegir.governance import graph as G
        cap = 40
        # MATERIALIZED-FIRST (RH 2026-07-21): panel-shaped lineage views make the ego two
        # indexed selects; live cypher is the fallback (fresh DB / mid-refresh), which also
        # kicks a background ensure+refresh so the next click is fast.
        try:
            from aegir.governance import lineage_views as LV
            mv = LV.ego(focal, cap=cap)
            if mv is not None:
                return mv
            LV.refresh_throttled_async()
        except Exception:  # noqa: BLE001
            pass

        def _disp(label: str, mp: dict | None) -> str:
            mp = mp or {}
            for k in ("name", "title", "chapter_id", "template_id", "run_id", "dataset", "qualifiedName"):
                if mp.get(k):
                    return str(mp[k])
            if mp.get("topic_id") is not None:
                return f"topic {mp['topic_id']}"
            return label

        try:
            with G.connect() as conn:
                if focal is None:
                    for lab in ("Dataset", "Run", "Chapter"):
                        r = G.run(conn, f"MATCH (n:{lab}) RETURN {{vid: id(n)}} LIMIT 1")
                        if r:
                            focal = int(r[0]["vid"])
                            break
                    if focal is None:
                        return {"focal": None, "neighbors": [], "truncated": False}
                fv = int(focal)
                f = G.run(conn, f"MATCH (n) WHERE id(n) = {fv} RETURN {{label: labels(n)[0], mp: properties(n)}}")
                if not f:
                    raise HTTPException(404, f"no provenance node {fv}")
                nb = G.run(conn, f"MATCH (n)-[r]-(m) WHERE id(n) = {fv} AND labels(m)[0] <> 'vertex' "
                                 "RETURN {mid: id(m), mlabel: labels(m)[0], etype: type(r), "
                                 f"out: (startNode(r) = n), mp: properties(m)}} LIMIT {cap + 1}")
                nb = [x for x in nb if not str(x.get("etype", "")).startswith("__rdbms")]
                truncated = len(nb) > cap
                nb = nb[:cap]
                fl = f[0]
                return {
                    "focal": {"vid": fv, "label": fl["label"], "name": _disp(fl["label"], fl.get("mp"))},
                    "neighbors": [{"vid": int(x["mid"]), "label": x["mlabel"],
                                   "name": _disp(x["mlabel"], x.get("mp")), "edge": x["etype"],
                                   "out": bool(x["out"])} for x in nb],
                    "truncated": truncated, "cap": cap,
                }
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 — graph down/restarting; the panel shows an empty state
            return {"focal": None, "neighbors": [], "error": type(e).__name__, "truncated": False}

    # ── /api/ontology/dbpedia-types ────────────────────────────

    @app.get("/api/ontology/dbpedia-types")
    def ontology_dbpedia_types() -> dict:
        cfg: Config = app.state.cfg
        gt_path = Path(cfg.data.gittables_signals_dir) / "gittables_gt.json"
        if not gt_path.exists():
            # Accept the absence gracefully — Ontologies panel renders an
            # empty-state banner rather than failing. M2 brings a real import flow.
            return {"source": str(gt_path), "exists": False, "types": []}
        mappings = json.loads(gt_path.read_text()).get("mappings", {})
        counts: dict[str, int] = {}
        for v in mappings.values():
            counts[v] = counts.get(v, 0) + 1
        types = [
            {"label": k, "count": counts[k]}
            for k in sorted(counts.keys())
        ]
        return {
            "source": str(gt_path),
            "exists": True,
            "total_columns": sum(counts.values()),
            "num_types": len(types),
            "types": types,
        }

    # ── /api/stats ─────────────────────────────────────────────

    @app.get("/api/stats")
    def stats() -> dict:
        """Compact summary for the Landing page stat cards.

        Matches the Atelier Landing pattern (4 cards: Status / Runs / Tasks
        / Terms). Field names are stable so the React UI can lock them in.
        """
        cfg: Config = app.state.cfg
        runs_dir = Path(cfg.runs.dir)
        run_dirs = iter_runs(runs_dir)

        try:
            from aegir.data.table_dataset import TASK_NUM_CLASSES
            num_tasks = len(TASK_NUM_CLASSES)
            # Terms = union of all label vocabulary sizes across registered
            # tasks. A rough "how much world does the model know?" indicator;
            # more honest than summing across overlapping ontologies.
            num_terms = sum(v for v in TASK_NUM_CLASSES.values() if v)
        except Exception:
            num_tasks = 0
            num_terms = 0

        latest_run = None
        if run_dirs:
            try:
                latest_run = load_run_summary(run_dirs[0])
            except Exception:
                pass

        return {
            "service": {
                "ok": True,
                "version": _pkg_version(),
                "runs_dir_exists": runs_dir.exists(),
            },
            "runs": {
                "count": len(run_dirs),
                "latest": latest_run,
            },
            "tasks": {
                "count": num_tasks,
            },
            "terms": {
                "count": num_terms,
                "note": "sum over registered task vocabularies — overlaps not deduplicated",
            },
        }

    # ── /api/classifications/latest ────────────────────────────

    @app.get("/api/classifications/latest")
    def classifications_latest() -> dict:
        """Return the most recent run's summary metrics.

        M1: per-class F1 breakdown is not persisted in sidecars (only micro +
        macro aggregates), so this endpoint returns a clearly-labeled minimal
        payload: run_id, micro, macro, num_classes. M2 will extend the metrics
        schema with the per-class vector so this endpoint can render a bar
        chart. The *shape* of the response stabilizes here.
        """
        cfg: Config = app.state.cfg
        runs = iter_runs(Path(cfg.runs.dir))
        if not runs:
            return {"run_id": None, "message": "no runs yet — train a model to populate"}
        latest = load_run_summary(runs[0])
        return {
            "run_id": latest.get("run_id"),
            "task": latest.get("task"),
            "model_size": latest.get("model_size"),
            "micro_f1": latest.get("best_val_micro_f1"),
            "macro_f1": latest.get("best_val_macro_f1"),
            "num_classes": _task_num_classes(latest.get("task")),
            "per_class_f1": None,  # M2: will populate with a 120-length list
        }


    # ── /api/classifications/catalog ──────────────────────────
    #
    # Apache Atlas vernacular: a Classification (sometimes called a Tag) is a
    # named type that can be attached to Entities, optionally inherits from
    # parent Classifications, and carries Attributes (typed fields). Atelier
    # uses the same shape (see ``atelier/ui/src/types/canvas.ts``). Aegir
    # exposes an Atlas-compatible catalog so our training corpus metadata can
    # round-trip with an existing Atlas deployment without reshape.

    @app.get("/api/classifications/catalog")
    def classifications_catalog() -> dict:
        """Read-only Atlas-style classification tree.

        M1: seeded from the registered task label spaces — each task becomes a
        top-level classification whose children are its labels. This is the
        minimal honest view ("here are the labels we train on"). M2 brings an
        editable catalog with BFO-grounded Atlas types and per-entity
        tagging.
        """
        cfg: Config = app.state.cfg
        del cfg  # reserved for future use (Atlas endpoint sync)

        try:
            from aegir.data.table_dataset import TASK_NUM_CLASSES
        except Exception:
            return {"classifications": [], "source": "unavailable"}

        children: list[dict] = []
        for task_id, num in TASK_NUM_CLASSES.items():
            children.append(
                {
                    "name": task_id,
                    "kind": "task",
                    "num_labels": num,
                    "parent": "AegirTask",
                    # Atlas-style: an Entity tagged with this classification
                    # is a column labeled under this task's vocabulary.
                    "attributes": [
                        {"name": "model_size", "type": "string"},
                        {"name": "run_id", "type": "string"},
                        {"name": "predicted_label", "type": "string"},
                    ],
                }
            )
        return {
            "classifications": [
                {
                    "name": "AegirTask",
                    "kind": "root",
                    "description": (
                        "Root classification for any Aegir-trained task. "
                        "Subclasses correspond to registered task vocabularies."
                    ),
                    "children": children,
                }
            ],
            "source": "registered_tasks",
            "editable": False,  # M2 unblocks write paths
        }

    # ── /api/ontology/vocabulary ───────────────────────────────

    @app.get("/api/ontology/vocabulary")
    def ontology_vocabulary() -> dict:
        """The common ICE/BFO-grounded training ontology, as presently defined.

        M1: returns a minimal skeleton sketching our upper-level alignment
        with Basic Formal Ontology (BFO) and the Common Core Ontologies (CCO).
        M2 will replace this with the real pretraining ontology artifact
        once the Stage-2 ontology-extraction pipeline lands.
        """
        return {
            "name": "Aegir ICE/BFO training ontology (skeleton)",
            "upper": "BFO:Entity",
            "nodes": [
                {"id": "BFO:Continuant", "parent": "BFO:Entity", "kind": "upper"},
                {"id": "BFO:Occurrent", "parent": "BFO:Entity", "kind": "upper"},
                {"id": "BFO:Object", "parent": "BFO:Continuant", "kind": "upper"},
                {"id": "BFO:Process", "parent": "BFO:Occurrent", "kind": "upper"},
                {"id": "BFO:Role", "parent": "BFO:Continuant", "kind": "upper"},
                {"id": "CCO:Agent", "parent": "BFO:Object", "kind": "mid"},
                {"id": "CCO:InformationBearingEntity", "parent": "BFO:Object", "kind": "mid"},
                {
                    "id": "ICE:DataElement",
                    "parent": "CCO:InformationBearingEntity",
                    "kind": "mid",
                    "description": (
                        "Aegir's domain anchor — a coherent data element "
                        "spanning one or more table columns."
                    ),
                },
            ],
            "note": (
                "M1 skeleton; M2 expands via the ontology-extraction pipeline "
                "(docs/current/src/pretraining.md, Stage 2)."
            ),
        }

    # ── /api/ontology/subsume ─────────────────────────────────

    @app.post("/api/ontology/subsume")
    async def ontology_subsume(payload: dict) -> dict:
        """Predict subsumption from a user-supplied vocab into our ontology.

        Body: ``{"terms": [{"name": "...", "description": "...", ...}, ...]}``

        M1 response is a deterministic skeleton: each input term maps to
        ``ICE:DataElement`` with low confidence and a ``"reasoning": "M1 stub — replace with BERTSubs-based predictor"`` note.
        The response *shape* stabilizes here so the React side of the
        upload/review flow can be built out without waiting for the real
        predictor. M2 replaces the stub body with a BERTSubs-style
        contextual-embedding subsumption predictor (Chen et al., "Contextual
        Semantic Embeddings for Ontology Subsumption Prediction").
        """
        terms = payload.get("terms", []) if isinstance(payload, dict) else []
        if not isinstance(terms, list):
            raise HTTPException(status_code=400, detail="'terms' must be a list")
        max_terms = 1000
        if len(terms) > max_terms:
            raise HTTPException(
                status_code=413,
                detail=f"too many terms ({len(terms)} > {max_terms}); batch and retry",
            )

        suggestions: list[dict] = [
            _subsume_stub(t) for t in terms
            if isinstance(t, dict) and str(t.get("name", "")).strip()
        ]
        return {
            "suggestions": suggestions,
            "count": len(suggestions),
            "predictor": "stub-m1",
        }

    # ── /api/p5/runs ──────────────────────────────────────────
    #
    # P5 GRPO/RLVR run discovery. Walks ``cfg.p5.output_dir`` for
    # ``checkpoint-N`` subdirs, returns one row per checkpoint with
    # the run's RunMetadata sidecar + a count of SAE feature
    # records spilled to that checkpoint. The UI uses this to
    # render the run list and pick a run to subscribe to.

    @app.get("/api/p5/runs")
    def p5_runs() -> dict:
        cfg: Config = app.state.cfg
        p5_dir = Path(cfg.p5.output_dir)
        if not p5_dir.exists():
            return {
                "rows": [],
                "p5_dir": str(p5_dir),
                "p5_dir_exists": False,
            }

        rows = _list_p5_checkpoints(p5_dir, cfg.p5.metadata_filename, cfg.p5.sae_log_filename)
        return {
            "rows": rows,
            "count": len(rows),
            "p5_dir": str(p5_dir),
            "p5_dir_exists": True,
        }

    # ── /api/p5/sae/stream ────────────────────────────────────
    #
    # Server-sent-events stream of SAE feature records as the
    # trainer spills them. The training process writes
    # ``checkpoint-N/sae_features.jsonl`` on every save event;
    # this endpoint tails the latest file and emits one SSE
    # ``data:`` line per JSON record. When the trainer rolls a
    # new checkpoint we emit a synthetic ``event: checkpoint``
    # marker so the UI can reset its plot state.
    #
    # Cross-worktree: the training process runs in this repo's
    # worktree, the UI in another worktree, both share the
    # ``p5.output_dir`` filesystem path. No additional IPC needed.

    @app.get("/api/p5/sae/stream")
    async def p5_sae_stream() -> StreamingResponse:
        cfg: Config = app.state.cfg
        p5_dir = Path(cfg.p5.output_dir)
        live_filename = cfg.p5.sae_live_log_filename
        snapshot_filename = cfg.p5.sae_log_filename

        async def event_gen():
            import asyncio
            # Prefer the run-root live JSONL when present (updated
            # every ``sae_live_spill_every_n_steps`` GRPO steps);
            # fall back to the latest checkpoint's snapshot when
            # the trainer hasn't started spilling live yet.
            live_path = p5_dir / live_filename
            last_source = ""  # "live" | "snapshot:<name>" | ""
            last_pos = 0
            heartbeat_every = 20  # cycles of poll_seconds → ~10s
            tick = 0
            poll_seconds = 0.5
            while True:
                if live_path.exists():
                    if last_source != "live":
                        last_source = "live"
                        last_pos = 0
                        yield (
                            f"event: source\n"
                            f"data: {json.dumps({'source': 'live', 'path': str(live_path)})}\n\n"
                        )
                    yielded = 0
                    try:
                        with live_path.open() as f:
                            f.seek(last_pos)
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                yield f"data: {line}\n\n"
                                yielded += 1
                            last_pos = f.tell()
                    except FileNotFoundError:
                        last_source = ""
                        last_pos = 0
                else:
                    ck = _latest_p5_checkpoint(p5_dir)
                    if ck is None:
                        yield "event: idle\ndata: {\"reason\": \"no live log and no checkpoints yet\"}\n\n"
                        await asyncio.sleep(2.0)
                        continue
                    src = f"snapshot:{ck.name}"
                    if last_source != src:
                        last_source = src
                        last_pos = 0
                        step = _checkpoint_step(ck)
                        yield (
                            f"event: checkpoint\n"
                            f"data: {json.dumps({'step': step, 'checkpoint': ck.name})}\n\n"
                        )
                    sae_path = ck / snapshot_filename
                    if sae_path.exists():
                        try:
                            with sae_path.open() as f:
                                f.seek(last_pos)
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    yield f"data: {line}\n\n"
                                last_pos = f.tell()
                        except FileNotFoundError:
                            pass
                tick += 1
                if tick % heartbeat_every == 0:
                    yield "event: heartbeat\ndata: {}\n\n"
                await asyncio.sleep(poll_seconds)

        return StreamingResponse(event_gen(), media_type="text/event-stream")


# Lightweight keyword → (iri, label, path) map used by the stub subsumption
# predictor. Intentionally shallow — a real BERTSubs-style model will replace
# this in M2. The point of the heuristic is to give the UI *meaningful*
# suggestions on common vocab (Customer, Invoice, Order, ...) without waiting
# for the embedding predictor, so users can exercise the review flow end-to-end.
_BFO_PATH_OBJECT = [
    "BFO:Entity",
    "BFO:Continuant",
    "BFO:Object",
    "CCO:Agent",
]
_BFO_PATH_INFO = [
    "BFO:Entity",
    "BFO:Continuant",
    "BFO:Object",
    "CCO:InformationBearingEntity",
    "ICE:DataElement",
]
_BFO_PATH_PROCESS = [
    "BFO:Entity",
    "BFO:Occurrent",
    "BFO:Process",
]
_BFO_PATH_ROLE = [
    "BFO:Entity",
    "BFO:Continuant",
    "BFO:Role",
]

_KEYWORD_HINTS: list[tuple[tuple[str, ...], dict]] = [
    # Agents / people / organizations — roughly BFO:Object/CCO:Agent.
    (
        ("customer", "client", "person", "patient", "employee", "user",
         "account_holder", "agent", "vendor", "supplier", "provider", "party"),
        {"iri": "CCO:Agent", "label": "Agent", "confidence": 0.6, "path": _BFO_PATH_OBJECT},
    ),
    # Processes / transactions — BFO:Process.
    (
        ("transaction", "order", "purchase", "payment", "shipment", "delivery",
         "encounter", "visit", "session", "event", "process"),
        {"iri": "BFO:Process", "label": "Process", "confidence": 0.55, "path": _BFO_PATH_PROCESS},
    ),
    # Roles — BFO:Role (a specifically dependent continuant borne by an object).
    (
        ("role", "job_title", "position", "title", "designation", "rank"),
        {"iri": "BFO:Role", "label": "Role", "confidence": 0.55, "path": _BFO_PATH_ROLE},
    ),
    # Information artefacts — documents / records / identifiers.
    (
        ("invoice", "receipt", "document", "record", "report", "statement",
         "id", "identifier", "number", "code", "reference", "file", "log"),
        {"iri": "ICE:DataElement", "label": "Data Element", "confidence": 0.5,
         "path": _BFO_PATH_INFO},
    ),
]


def _checkpoint_step(checkpoint_dir: Path) -> int:
    """Parse ``N`` out of ``checkpoint-N``. Returns -1 on a malformed
    directory name so callers can sort robustly."""
    name = checkpoint_dir.name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("-", 1)[1])
    except (ValueError, IndexError):
        return -1


def _latest_p5_checkpoint(p5_dir: Path) -> Path | None:
    """Return the highest-numbered ``checkpoint-N`` directory under
    ``p5_dir``, or ``None`` if no valid checkpoints exist. Mirrors
    ``aegir.rl.checkpointing.latest_checkpoint`` so the gateway has
    no run-time dependency on the rl package."""
    if not p5_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for entry in p5_dir.iterdir():
        if not entry.is_dir():
            continue
        step = _checkpoint_step(entry)
        if step >= 0:
            candidates.append((step, entry))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _list_p5_checkpoints(p5_dir: Path, metadata_filename: str,
                         sae_log_filename: str) -> list[dict]:
    """Walk ``p5_dir`` and return one row per ``checkpoint-N`` dir
    with the run's metadata + a count of SAE feature records spilled
    to that checkpoint. Skips dirs whose metadata sidecar is missing
    or unreadable rather than 500-ing on a partial run."""
    rows: list[dict] = []
    if not p5_dir.exists():
        return rows
    for entry in sorted(p5_dir.iterdir()):
        step = _checkpoint_step(entry)
        if step < 0:
            continue
        meta_path = entry / metadata_filename
        try:
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
        except Exception:  # noqa: BLE001 — partial write tolerated
            metadata = None
        sae_path = entry / sae_log_filename
        n_sae = 0
        if sae_path.exists():
            try:
                with sae_path.open() as f:
                    n_sae = sum(1 for line in f if line.strip())
            except OSError:
                n_sae = 0
        rows.append({
            "step": step,
            "checkpoint": entry.name,
            "metadata": metadata,
            "n_sae_records": n_sae,
            "checkpoint_dir": str(entry),
        })
    return rows


def _subsume_stub(term: dict) -> dict:
    """Best-effort keyword-based ontology subsumption — stand-in for BERTSubs.

    Matches the term's name against a small keyword table that maps common
    data-modeling vocabulary to the three main branches of BFO via CCO. When
    a match is found, returns that candidate at its tabled confidence; when
    not, returns ICE:DataElement at low confidence (the catch-all for any
    column that carries information but isn't recognizably an agent, process
    or role).

    The shape of the return matches the fully-learned M2 predictor so the
    UI's upload/review flow stays stable when the real model drops in.
    """
    name = str(term.get("name", "")).strip()
    lower = name.lower()
    candidate: dict | None = None
    for keywords, tmpl in _KEYWORD_HINTS:
        if any(k in lower for k in keywords):
            candidate = dict(tmpl)
            break
    if candidate is None:
        candidate = {
            "iri": "ICE:DataElement",
            "label": "Data Element",
            "confidence": 0.05,
            "path": _BFO_PATH_INFO,
        }
        reasoning = (
            "No keyword match — defaulting to ICE:DataElement at low "
            "confidence. M2 BERTSubs predictor will score this against the "
            "full ontology neighborhood."
        )
    else:
        reasoning = (
            f"Matched keyword heuristic for {candidate['label']!r}. "
            "Placeholder predictor; the M2 scorer will refine confidence "
            "using contextual embeddings (BERTSubs)."
        )
    return {
        "term": name,
        "candidates": [candidate],
        "reasoning": reasoning,
    }


def _task_num_classes(task: str | None) -> int | None:
    """Look up the registered num_classes for a task id. Best-effort."""
    if not task:
        return None
    try:
        from aegir.data.table_dataset import TASK_NUM_CLASSES
        return TASK_NUM_CLASSES.get(task)
    except Exception:
        return None


def _is_safe_plot_name(name: str) -> bool:
    """Allow only basic slugs — blocks directory traversal via plot name."""
    return bool(name) and all(c.isalnum() or c in "_-" for c in name) and len(name) <= 64


# ── static bundle ───────────────────────────────────────────────


def _mount_static_bundle(app: FastAPI, cfg: Config) -> None:
    dist = Path(cfg.ui.dist_dir).resolve()

    if not dist.exists():
        # Graceful 404 at GET /: tell the user how to build the UI.
        @app.get("/", response_class=PlainTextResponse)
        def missing_bundle_root() -> PlainTextResponse:
            return PlainTextResponse(
                f"Aegir gateway is up, but the UI bundle is not built.\n"
                f"Expected: {dist}\n"
                f"Run: just ui-build (or: cd ui && pnpm install && pnpm build)\n",
                status_code=404,
            )
        return

    # Serve the built SPA. ``html=True`` means directory requests land on
    # index.html, which react-router-dom handles client-side.
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")


# Module-level app for ``uvicorn aegir.gateway.app:app``.
app = create_app()

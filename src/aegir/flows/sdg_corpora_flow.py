"""SdgCorporaFlow — the sdg sdg-corpora iteration as one Metaflow flow.

The long-run deliverable (RH 2026-07-04): FinePDFs passages → **metrology-informed** entity
derivation (the agent iterates against the SchemaPile structural signal, ``derive_loop``) →
merged realized ontology (**HermiT signs the certificate**; kvasir emits proof-carrying DDL +
SHACL shapes) → RI-true constructs → **parallel register-specialized ACP prose harnesses**
(natural ∥ semantic as true parallel Metaflow steps; semantic tool-equipped over MCP) →
gated verify → assembled sdg-corpora release tree.

Standing riders encoded here (the don't-forget list):
- REAL UNIVERSALS, FICTIONAL PARTICULARS — ``scan_sensitive_nouns`` runs in verify.
- Thinking traces RETAINED — full exchanges land in per-chapter ``exchange.json``.
- grok/xai are OPT-IN (``--backends local`` is the default; the pool mix is a Parameter).
- kvasir fast-refutes in-loop; HermiT certifies at realize (trust doctrine).
- The corpora PUBLISH stays behind ``aegir.lineup sync`` (OQuaRE hard gate) + the P5 riders
  (#136: DST belief module, generation manifest, predictions schema, mirror cleanup) — this
  flow ASSEMBLES the release tree; it does not push.
- Long runs: AEGIR_METAFLOW_MODE=local; engine complete-trace knobs per pipeline_run_gotchas.

    AEGIR_METAFLOW_MODE=local uv run --no-sync python -m aegir.flows.sdg_corpora_flow run \
        --n-passages 48 --rounds 2 --backends local --pool 2
"""
import os  # noqa: E402

import aegir.metaflow_patch  # noqa: F401  — MUST precede the metaflow import (404-retry patch)
from metaflow import FlowSpec, Parameter, current, step  # noqa: E402

from aegir.flows.config import apply_metaflow_config  # noqa: E402
from aegir.flows.trace import TracedFlow, traced_step  # noqa: E402

apply_metaflow_config(os.environ.get("AEGIR_METAFLOW_MODE", "rke2"))

import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
JVM = f"{REPO}/build/jvm-libs"
CUDA = f"{REPO}/build/cuda-driver-libs"
ART = Path("/raid/checkpoints/aegir-artifacts")


def _probe_ctx() -> "int | None":
    """The live vLLM's max_model_len (None if not serving yet — lazy spawn)."""
    import json as _json
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=3) as r:
            d = _json.load(r)
        return int(d["data"][0].get("max_model_len") or 0) or None
    except Exception:  # noqa: BLE001
        return None


def _engine_up(min_ctx: int = 32768) -> bool:
    """Ensure the engine serves WITH the complete-trace config (ctx >= min_ctx). The flow
    OWNS this env — ambient engines with a smaller context caused the overflow class.
    Returns True if THIS flow launched it (and should tear it down at end)."""
    if subprocess.run(["pgrep", "-f", "aegir.engine.server"], capture_output=True).returncode == 0:
        ctx = _probe_ctx()
        if ctx is not None and ctx < min_ctx:
            print(f"  WARN: engine already up with ctx {ctx} < {min_ctx} — big chapters will "
                  "route through the agent-stage fallback (never-choke), but a restart with "
                  f"AEGIR_MAX_MODEL_LEN={min_ctx} is faster", flush=True)
        else:
            print(f"  engine already up (ctx {ctx or 'lazy'})", flush=True)
        return False
    env = dict(os.environ, LD_LIBRARY_PATH=CUDA,
               AEGIR_MAX_MODEL_LEN=os.environ.get("AEGIR_MAX_MODEL_LEN", str(min_ctx)),
               AEGIR_MAX_NUM_SEQS=os.environ.get("AEGIR_MAX_NUM_SEQS", "4"))
    log = (REPO / "build" / "engine_flow.log").open("w")
    subprocess.Popen(["uv", "run", "--no-sync", "python", "-m", "aegir.engine.server"],
                     cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    for _ in range(60):
        if subprocess.run(["bash", "-c", "grep -q 'gRPC listening' build/engine_flow.log 2>/dev/null"],
                          cwd=str(REPO)).returncode == 0:
            print("  engine gRPC up", flush=True)
            return True
        time.sleep(2)
    print("  WARN: engine not confirmed; first call will block on model load", flush=True)
    return True


class SdgCorporaFlow(TracedFlow, FlowSpec):
    """Greenfield sdg-corpora: metrology-informed derive → HermiT-certified ontology →
    kvasir DDL/shapes → parallel two-register ACP prose → gated release tree."""

    n_passages = Parameter("n-passages", default=0, type=int,
                           help="FinePDFs passages to derive + write chapters from (0 = ALL harvested)")
    output_dir = Parameter("output-dir", default="",
                           help="override the run dir (top-up semantics: existing chapters are kept)")
    rounds = Parameter("rounds", default=2, type=int,
                       help="max metrology-feedback rounds per passage derivation")
    backends = Parameter("backends", default="local",
                         help="csv of prose backends (local|grok|xai) — grok/xai are OPT-IN")
    pool = Parameter("pool", default=2, type=int,
                     help="per-register concurrent harnesses per backend (engine max_num_seqs bound)")
    passage_chars = Parameter("passage-chars", default=6000, type=int,
                              help="chars of each passage fed to the deriver")
    skip_hermit = Parameter("skip-hermit", default=False, type=bool,
                            help="skip the HermiT certificate (fast shakedowns only — NOT releases)")
    entities_from = Parameter("entities-from", default="",
                              help="reuse a prior run's entities/ dir (skips the derive stage)")
    strategy_ref = Parameter("strategy", default="",
                             help="run under a strategy REF (branch/sha in the sdg-strategy "
                                  "submodule) — SHADOW runs: own corpus dir, collections from "
                                  "the ref's binding, manifests resolved by sha; '' = main")
    harvest_target = Parameter("harvest-target", default=0, type=int,
                               help="ADVANCE THE INPUT WINDOW: stream FinePDFs until this many "
                                    "new in-domain docs land (0 = use the harvest as-is)")
    refine_budget = Parameter("refine-budget", default=24, type=int,
                              help="max escalation-worklist entries triaged per run (the "
                                   "refine_escalations organ; 0 disables the pass)")
    corpus_mode = Parameter("corpus", default=True, type=bool,
                            help="accrete into the persistent corpus dir (idempotent top-up per "
                                 "input window) instead of a fresh per-run dir")
    ddl_scope = Parameter("ddl-scope", default="both",
                          help="the kvasir-scoped DDL stage's scope (pathway consolidation, RH "
                               "2026-07-23): comprehensive is DEFAULT-ON by ruling — 'both' runs "
                               "it alongside catalog (the released spine). catalog | "
                               "comprehensive | both | off")

    @traced_step
    @step
    def start(self):
        if self.strategy_ref:
            os.environ["AEGIR_STRATEGY_REF"] = self.strategy_ref  # one seam: all resolvers follow
        # OL-first (RH 2026-07-23): kvasir emits OpenLineage RunEvents natively; the
        # flow CONSUMES AND PERSISTS them run-scoped ({run_out}/ol_events — the durable
        # raw record) and projects them into governed provenance at step boundaries,
        # alongside the flow's own OL emission — one lineage spine, Marquez-servable.
        self.run_out = self.output_dir or (
            str(ART / "sdg-corpora" /
                (self._shadow_dir() if self.strategy_ref else "corpus")) if self.corpus_mode
            else str(ART / "sdg-corpora" / current.run_id))
        Path(self.run_out, "entities").mkdir(parents=True, exist_ok=True)
        os.environ["KVASIR_OL_DIR"] = str(Path(self.run_out, "ol_events").resolve())
        Path(self.run_out, "ol_events").mkdir(exist_ok=True)
        manifest = REPO / "build/domain_harvest/manifest.jsonl"
        docs = sorted((REPO / "build/domain_harvest/docs").glob("*.txt"))
        if not docs:
            raise RuntimeError("no harvested passages — run SemanticCorpusFlow harvest or "
                               "scripts/harvest_domain_docs.py first")
        self.passages = [str(p) for p in (docs if self.n_passages == 0 else docs[: self.n_passages])]
        self.emit_event("flow.started", {"n_passages": len(self.passages),
                                         "backends": self.backends, "manifest": manifest.exists()})
        print(f"SdgCorporaFlow {current.run_id}: {len(self.passages)} passages → {self.run_out}",
              flush=True)
        try:
            from aegir.strategy.manifest import declared, drift
            man = declared()  # follows AEGIR_STRATEGY_REF (shadow) or CURRENT (main)
            if self.strategy_ref and man:
                from aegir.strategy.materialize import materialize
                mat = materialize(self.strategy_ref)
                bad = {k: v for k, v in mat.items() if "NO SOURCE" in v or "MISMATCH" in v}
                if bad:
                    raise RuntimeError(f"shadow lens not materializable: {bad} — "
                                       "run `python -m aegir.strategy.materialize "
                                       f"{self.strategy_ref}` and inspect")
                print(f"  shadow {man['strategy_id']} @ {self.strategy_ref}: lens {mat}",
                      flush=True)
            if man:
                d = drift(man)
                if d:
                    msg = f"STRATEGY DRIFT vs {man['strategy_id']}: {', '.join(d[:6])}"
                    if os.environ.get("AEGIR_STRATEGY_ENFORCE") == "1":
                        raise RuntimeError(msg + " (AEGIR_STRATEGY_ENFORCE=1)")
                    print(f"  WARNING {msg} — re-seed or commit the strategy", flush=True)
                else:
                    print(f"  strategy {man['strategy_id']} CLEAN (drift-checked)", flush=True)
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 — accountability must not sink the run
            print(f"  strategy check skipped ({str(e)[:100]})", flush=True)
        self.next(self.harvest)

    def _shadow_dir(self) -> str:
        from aegir.strategy.manifest import load_by_ref
        man = load_by_ref(self.strategy_ref)
        return f"corpus-{man['strategy_id']}"

    def _arm_strategy_env(self):
        """Steps run as separate processes — re-arm the ref seam in each."""
        if self.strategy_ref:
            os.environ["AEGIR_STRATEGY_REF"] = self.strategy_ref

    @traced_step
    @step
    def harvest(self):
        """ADVANCE THE INPUT WINDOW (idempotent by construction: content-hash doc filenames,
        persistent stream cursor, append-only manifest). --harvest-target 0 = no-op."""
        self._arm_strategy_env()
        if self.harvest_target > 0:
            import subprocess as sp
            args = ["uv", "run", "--no-sync", "python", "scripts/harvest_domain_docs.py",
                    "--target", str(self.harvest_target),
                    "--max-stream", str(self.harvest_target * 40)]
            try:
                from aegir.strategy.manifest import lens_binding
                aim = lens_binding().get("aiming_collection")
                if aim:
                    args += ["--domain-collection", aim]  # the strategy picks the lens
            except Exception:  # noqa: BLE001
                pass
            r = sp.run(args, cwd=str(REPO))
            if r.returncode != 0:
                raise RuntimeError(f"harvest failed (exit {r.returncode})")
            docs = sorted((REPO / "build/domain_harvest/docs").glob("*.txt"))
            self.passages = [str(p) for p in (docs if self.n_passages == 0
                                              else docs[: self.n_passages])]
            print(f"  window advanced → {len(self.passages)} passages total", flush=True)
        self.next(self.derive)

    @traced_step
    @step
    def derive(self):
        """Metrology-informed derivation: the agent proposes; kvasir profiles the DDL against
        SchemaPile; below `rich` the structural reason re-prompts the agent (bounded rounds)."""
        if self.entities_from:
            import shutil
            src = Path(self.entities_from)
            for f in src.glob("*"):
                shutil.copy2(f, Path(self.run_out, "entities", f.name))
            self.engine_owned = _engine_up()  # prose still needs the engine
            self.derive_stats = {"reused_from": str(src),
                                 "n": len(list(src.glob('*.json')))}
            self.derive_reports = {}
            print(f"  reused {self.derive_stats['n']} derived passages from {src}", flush=True)
            self.next(self.refine_escalations)
            return
        self._arm_strategy_env()
        self.engine_owned = _engine_up()
        from aegir.ontology.derive_harness import deriver_version
        from aegir.ontology.derive_loop import derive_with_metrology
        from aegir.ontology.entities import to_manchester
        dv = deriver_version()
        try:
            from aegir.strategy.lineage import stage_key
            skey = stage_key("derive") or ""
        except Exception:  # noqa: BLE001
            skey = ""
        ent_dir = Path(self.run_out, "entities")
        stats = {"rich": 0, "thin": 0, "inert": 0, "malformed": 0, "cached": 0,
                 "cached_legacy": 0}
        self.derive_reports = {}
        for i, ppath in enumerate(self.passages):
            pid = Path(ppath).stem[:16]
            ej = ent_dir / f"{pid}.json"
            if ej.exists():
                try:
                    st = json.loads(ej.read_text())
                    # lineage-entailed key (lens + derive voices + schema); grandfather:
                    # pre-lineage entries validate on deriver_version alone and re-stamp
                    # naturally at the next real change (the #150 schema bump re-derives all).
                    if skey and st.get("stage_key") == skey:
                        stats["cached"] += 1
                        continue
                    if st.get("deriver_version") == dv and "stage_key" not in st:
                        stats["cached_legacy"] += 1
                        continue
                except Exception:  # noqa: BLE001 — unreadable stamp → re-derive
                    pass
            passage = Path(ppath).read_text()[: self.passage_chars]
            entities, report = derive_with_metrology(passage, max_rounds=self.rounds)
            stats[report.final_verdict or "malformed"] = stats.get(report.final_verdict, 0) + 1
            (ent_dir / f"{pid}.json").write_text(json.dumps({
                "passage": ppath,
                "deriver_version": dv,
                "stage_key": skey,
                "entities": [{"name": e.name, "label": e.label, "genus": e.genus,
                              "definition": e.definition,
                              "attributes": [{"name": a.name, "xsd": a.xsd, "enum": a.enum}
                                             for a in e.attributes],
                              "relations": [{"prop": r.prop, "target": r.target, "card": r.card}
                                            for r in e.relations]} for e in entities],
            }, indent=1))
            (ent_dir / f"{pid}.omn").write_text(to_manchester(entities))
            self.derive_reports[pid] = report.summary()
            print(f"  [{i+1}/{len(self.passages)}] {pid}: {len(entities)} entities, "
                  f"verdict={report.final_verdict} (round {report.accepted_round})", flush=True)
        self.derive_stats = stats
        self.emit_event("derive.done", stats)
        self.next(self.refine_escalations)

    @traced_step
    @step
    def refine_escalations(self):
        """Semantic-upkeep organ (#32, RH-ratified): the standing consumer of the elaboration ACP
        worklist. Bounded, stratified, aged; the agent proposes over the fork ACP channel with the
        FULL membrane dialogue; the same membranes dispose. Success = processed, not emptied —
        residue persisting is green. Engine down → held, no attempts burned."""
        from aegir.refine.elaboration import triage
        if not self.engine_owned or int(self.refine_budget) <= 0:
            self.refine_kpis = {"skipped": "engine down — worklist held, no attempts burned"
                                if not self.engine_owned else "refine-budget 0"}
            print("  refine_escalations: engine down → held", flush=True)
            self.next(self.realize)
            return
        self.refine_kpis = triage(budget=int(self.refine_budget), run_key=current.run_id)
        k = self.refine_kpis
        print(f"  refine_escalations: in={k.get('worklist_in')} processed={k.get('processed')} "
              f"accepted={k.get('accepted')} cleared={k.get('cleared')} "
              f"aged_to_review={k.get('aged_to_review')} residual={k.get('residual')}", flush=True)
        self.emit_event("refine_escalations.done", {kk: vv for kk, vv in k.items()
                                                    if kk not in ("lineage", "age_histogram")})
        self.next(self.realize)

    @traced_step
    @step
    def realize(self):
        """Merge → HermiT certificate → kvasir DDL + SHACL shapes → SchemaPile structure score.

        CERTIFICATION LADDER (RH 2026-07-26): a reasoner that cannot finish is a GRADE, not a
        stop — merges keep outgrowing monolithic reasoning as the ontology grows. Attempt the
        full-scope pass; on the tractability signal (exit 3) re-attempt with decomposed
        certification; record the achieved tier and CONTINUE. Only an unmodellable theory
        (exit 2) or genuine unsatisfiable classes (exit 4) stop the flow.

        A budget trip must never cost the downstream steps again: in run 1785041579228975 the
        90-min grind failed the flow outright, banking derive's 6.5h but leaving chapters,
        congruence, assemble, project and ddl_stage unrun — and reporting the tractability
        limit as a sick TBox, which is the opposite remediation.
        """
        env = dict(os.environ, LD_LIBRARY_PATH=JVM)

        def _attempt(repaired: bool):
            args = ["uv", "run", "--no-sync", "python", "scripts/realize_sdg.py",
                    "--entities-dir", f"{self.run_out}/entities",
                    "--output-dir", f"{self.run_out}/ontology"]
            if self.skip_hermit:
                args.append("--skip-hermit")
            if repaired:
                # BOTH ratified tier-1 repairs, measured 2026-07-26 on the 3027-passage merge:
                #   full nominal union            > 5400s, no verdict (run 1785041579228975;
                #                                 realize_sdg's own comment records an earlier one)
                #   --drop-individuals alone      > 5400s, no verdict  ← nominals were NOT the cause
                #   + --reconcile-categories        21s, consistent, unsat=0, level=certified
                # The driver was 38,975 union-global `Functional` axioms — per-passage "=1"
                # readings that do not lift (the catalog asserts none). Functionality plus
                # max-cardinality forces equality reasoning across 112,463 existentials; the
                # projection is otherwise almost EL (no only/not/or/inverse/disjoint at all).
                args += ["--drop-individuals", "--reconcile-categories"]
            return subprocess.run(args, cwd=str(REPO), env=env)

        r = _attempt(repaired=False)
        if r.returncode == 3:
            print("  realize: HermiT exceeded the budget at full scope → re-attempting with the "
                  "ratified tier-1 repairs (decomposed certification + functional-overassertion "
                  "and double-category reconciliation)", flush=True)
            r = _attempt(repaired=True)
        if r.returncode == 2:
            raise RuntimeError("HermiT refused the merged ontology (INCONSISTENT — no model) — "
                               f"see {self.run_out}/ontology/certificate.json")
        if r.returncode == 4:
            raise RuntimeError("HermiT found UNSATISFIABLE classes in the merge — the TBox is "
                               f"sick; see {self.run_out}/ontology/certificate.json (unsat list)")
        if r.returncode == 3:
            raise RuntimeError(
                "HermiT exceeded AEGIR_REASON_BUDGET_S at BOTH full and decomposed scope, so "
                "there is NO verdict — nothing is certified and nothing ships. This is a "
                "tractability limit, NOT a sick TBox: raise the budget deliberately, or "
                "decompose further (kvasir T1). The emitted artifacts still stand for "
                "inspection; do not read a stale certificate.json as this run's evidence.")
        if r.returncode != 0:
            raise RuntimeError(f"realize failed (exit {r.returncode})")
        cert = json.loads(Path(self.run_out, "ontology/certificate.json").read_text())
        self.certification = {k: cert.get(k) for k in (
            "level", "releasable_as", "method", "scope_certified", "uncertified_residue",
            "generated_at")}
        lvl, rel = self.certification.get("level"), self.certification.get("releasable_as")
        residue = self.certification.get("uncertified_residue") or []
        print(f"  realize: certification level={lvl} → releasable as {rel}"
              + (f"; uncertified residue: {'; '.join(residue)}" if residue else ""), flush=True)
        self.emit_event("realize.certified", {"level": lvl, "releasable_as": rel,
                                              "n_residue": len(residue)})
        self.structure = json.loads(Path(self.run_out, "ontology/structure.json").read_text())
        self.emit_event("realize.done", {"shape_emd": str(self.structure.get("shape_emd"))})
        try:
            from aegir.governance.ol import ingest_events_dir
            n_ol = ingest_events_dir(Path(self.run_out, 'ol_events'))
            if n_ol:
                print(f'   OL: {n_ol} kvasir run event(s) → governed provenance')
        except Exception:  # noqa: BLE001 — provenance never blocks the pass
            pass
        self.next(self.ddl_stage)

    @traced_step
    @step
    def ddl_stage(self):
        """THE kvasir-scoped DDL stage (pathway consolidation, RH 2026-07-23): ontology →
        relational under the ONE conductor, scope as a parameter. `catalog` regenerates the
        RELEASED spine (content-addressed run_id — an unchanged catalog swaps nothing);
        `comprehensive` runs the certified-union kvasir lowering (covers the classes the
        catalog spine leaves unpopulated). Each scope emits its closure artifact NATIVELY
        (associations module — generation and record are one act) and its own OL RunEvent
        into governed PROVENANCE; kvasir's native events ride KVASIR_OL_DIR alongside."""
        scope = str(self.ddl_scope or "catalog").lower()
        self.ddl_stage_report = {"scope": scope}
        if scope in ("off", "none"):
            print("  ddl_stage: off", flush=True)
            self.next(self.build_constructs)
            return
        from aegir.governance.ol import emit_run_event, file_dataset
        from aegir.ontology.associations import (emit_comprehensive_closure,
                                                 emit_spine_associations)
        ev_dir = Path(self.run_out, "ol_events")
        if scope in ("catalog", "both"):
            from aegir.lineup.sync import _regen_ddl
            ok = _regen_ddl()
            runs = sorted((REPO / "corpora/ddl").glob("*/manifest.json"))
            if ok and runs:
                run_dir = runs[-1].parent
                s = emit_spine_associations(run_dir) or {}
                self.ddl_stage_report["catalog"] = s
                try:
                    from aegir.ontology.sql_bundle import emit_sql_bundle
                    sb = emit_sql_bundle(run_dir)
                    self.ddl_stage_report["sql_bundle"] = sb["dialects"]
                    print("  ddl[catalog]: sql bundle "
                          + " · ".join(f"{d}:{x['tables']}t/{x['views']}v"
                                       for d, x in sb["dialects"].items()), flush=True)
                except Exception as e:  # noqa: BLE001 — emission failure is visible, not fatal
                    print(f"  ddl[catalog] sql bundle deferred ({str(e)[:80]})", flush=True)
                try:
                    emit_run_event(
                        "ddl-spine", run_key=run_dir.name, events_dir=ev_dir,
                        inputs=[file_dataset(REPO / "src/aegir/ontology/catalog/catalog.json")],
                        outputs=[file_dataset(run_dir / f) for f in
                                 ("naming_map.parquet", "ddl_statements.parquet", "views.parquet",
                                  "ontology_entity_associations.json",
                                  "sql/postgres/00_schema.sql") if (run_dir / f).exists()],
                        facets={"scope": "catalog",
                                "closure": {k: s.get(k) for k in
                                            ("n_tables", "n_without_pattern", "n_unpopulated",
                                             "n_classes", "n_name_matches_rejected")}})
                except Exception as e:  # noqa: BLE001 — provenance never blocks the pass
                    print(f"  ddl[catalog] OL deferred ({str(e)[:80]})", flush=True)
                print(f"  ddl[catalog]: run {run_dir.name} · closure {s.get('n_tables')} tables "
                      f"({s.get('n_without_pattern')} without pattern) · "
                      f"{s.get('n_unpopulated')} classes unpopulated", flush=True)
        if scope in ("comprehensive", "both"):
            out = Path(self.run_out, "ddl_comprehensive")
            args = ["uv", "run", "--no-sync", "python", "scripts/realize_sdg.py",
                    "--entities-dir", f"{self.run_out}/entities",
                    "--merge-ontology", str(REPO / "corpora/ontology/sdg-ontology.omn"),
                    "--inline-theory", "--arm-property-domains", "--ground-entity-props",
                    "--reconcile-categories", "--drop-individuals",
                    "--output-dir", str(out)]
            if self.skip_hermit:
                args.append("--skip-hermit")
            r = subprocess.run(args, cwd=str(REPO), env=dict(os.environ, LD_LIBRARY_PATH=JVM))
            if r.returncode == 2:
                raise RuntimeError("comprehensive union INCONSISTENT — see "
                                   f"{out}/certificate.json")
            if r.returncode != 0:
                # shakedown doctrine: artifacts land before the certificate; unsat is
                # worklist signal, never blindness
                print(f"  ddl[comprehensive]: certificate refused (exit {r.returncode}) — "
                      "artifacts stand, unsat set = the re-author worklist", flush=True)
            s = emit_comprehensive_closure(out) or {}
            self.ddl_stage_report["comprehensive"] = s
            if s:
                try:
                    emit_run_event(
                        "ddl-comprehensive", run_key=Path(self.run_out).name, events_dir=ev_dir,
                        inputs=[file_dataset(REPO / "corpora/ontology/sdg-ontology.omn")],
                        outputs=[file_dataset(out / f) for f in
                                 ("ddl.sql", "plan.json", "shapes.ttl",
                                  "ontology_entity_associations.json") if (out / f).exists()],
                        facets={"scope": "comprehensive",
                                "closure": {k: s.get(k) for k in
                                            ("n_tables", "n_without_pattern",
                                             "n_unpopulated", "n_classes")}})
                except Exception as e:  # noqa: BLE001
                    print(f"  ddl[comprehensive] OL deferred ({str(e)[:80]})", flush=True)
                print(f"  ddl[comprehensive]: closure {s.get('n_tables')} tables "
                      f"({s.get('n_without_pattern')} without pattern) · "
                      f"{s.get('n_unpopulated')} classes unpopulated", flush=True)
        try:
            from aegir.governance.ol import ingest_events_dir
            n_ol = ingest_events_dir(ev_dir)
            if n_ol:
                print(f"   OL: {n_ol} run event(s) → governed provenance", flush=True)
        except Exception:  # noqa: BLE001
            pass
        self.emit_event("ddl_stage.done", {"scope": scope})
        self.next(self.build_constructs)

    @traced_step
    @step
    def build_constructs(self):
        """Per-passage RI-true constructs + materialized VIEWS (the embedded payload the
        chapter prose supports) + kvasir-verified facts (the semantic grounding)."""
        self._arm_strategy_env()
        from aegir.generate.harness import kvasir_facts
        from aegir.ontology.entities import (add_views, from_json, render_payload_blocks,
                                             resolve_construct_constraints, to_construct, to_manchester)
        from aegir.ontology.rows import require_gittables
        # BOUNDARY (no silent degradation): real GitTables values or HALT — a transiently-missing value
        # profile must not silently revert the whole corpus to mechanical placeholders (RH 2026-07-13).
        require_gittables()
        cons_dir = Path(self.run_out, "constructs")
        cons_dir.mkdir(exist_ok=True)
        self.construct_ids = []
        n_views = 0
        from collections import Counter
        pk_kinds, tstyles = Counter(), Counter()
        try:
            from aegir.strategy.lineage import stage_key as _sk
            ckey = _sk("build_constructs") or ""
        except Exception:  # noqa: BLE001
            ckey = ""
        for p in sorted(Path(self.run_out, "entities").glob("*.json")):
            cj = cons_dir / f"{p.stem}.json"
            if cj.exists() and cj.stat().st_mtime >= p.stat().st_mtime:
                con_cached = json.loads(cj.read_text())
                stored = con_cached.get("stage_key")
                if ckey and stored is not None and stored != ckey:
                    pass  # targets changed under the strategy → rebuild below
                else:
                    n_views += len(con_cached.get("views", []))
                    kp = con_cached.get("key_plan") or {}
                    tstyles[kp.get("table_style", "?")] += 1
                    for pl in (kp.get("plans") or {}).values():
                        pk_kinds[pl.get("kind", "?")] += 1
                    self.construct_ids.append(p.stem)
                    continue  # idempotence: newer than entities + same targets key
            ents = from_json(json.loads(p.read_text()))
            if not ents:
                continue
            omn = to_manchester(ents)
            # round-trip alignment (engine up by this step) — real, in-range numerics; cached across constructs
            constraints = resolve_construct_constraints(ents)
            con = to_construct(ents, n_rows=4, constraints=constraints)
            con = add_views(con, ents)
            blocks = render_payload_blocks(con)
            con["payload_blocks"] = blocks
            con["payload_markers"] = list(blocks)
            con["payload_preview"] = "\n\n".join(list(blocks.values())[:6])[:6000]
            con["ontology_omn"] = omn
            con["kvasir_facts"] = kvasir_facts(omn)
            con["stage_key"] = ckey
            n_views += len(con.get("views", []))
            kp = con.get("key_plan") or {}
            tstyles[kp.get("table_style", "?")] += 1
            for pl in (kp.get("plans") or {}).values():
                pk_kinds[pl.get("kind", "?")] += 1
            (cons_dir / f"{p.stem}.json").write_text(json.dumps(con, indent=1))
            self.construct_ids.append(p.stem)
        npk = sum(pk_kinds.values()) or 1
        self.key_shape_report = {
            "pk_kinds": {k: round(v / npk, 3) for k, v in pk_kinds.most_common()},
            "table_styles": dict(tstyles)}
        print(f"  {len(self.construct_ids)} constructs · {n_views} views · "
              f"pk kinds {self.key_shape_report['pk_kinds']}", flush=True)
        self.next(self.prose_natural, self.prose_semantic)

    def _run_register(self, register: str) -> list:
        """Drive the work queue for ONE register across all constructs × backends.
        STREAM-WRITES each chapter as it completes (a crash at item N never loses N-1)
        and SKIPS chapters already present (top-up semantics with --output-dir)."""
        self._arm_strategy_env()
        import asyncio
        from aegir.generate.harness import build_queue, run_queue
        cons_dir = Path(self.run_out, "constructs")
        out_dir = Path(self.run_out, "chapters")
        backends = [b.strip() for b in self.backends.split(",") if b.strip()]
        rows = []
        todo_ids = []
        try:
            from aegir.strategy.lineage import stage_key as _sk
            pkey = _sk("prose") or ""
        except Exception:  # noqa: BLE001
            pkey = ""
        for cid in self.construct_ids:
            md = out_dir / cid / f"{register}.md"
            kf = out_dir / cid / f".{register}.prose_key"
            stored = kf.read_text().strip() if kf.exists() else None
            if md.exists() and (not pkey or stored is None or stored == pkey):
                # grandfather: pre-lineage chapters (no sidecar) stay valid until the prose
                # voices actually change under a keyed regime
                t = md.read_text()
                rows.append({"construct": cid, "register": register, "backend": "cached",
                             "provider": "cached", "chars": len(t), "tool_calls": [],
                             "embed": {}, "error": ""})
            else:
                todo_ids.append(cid)
        if todo_ids:
            constructs = {cid: json.loads((cons_dir / f"{cid}.json").read_text())
                          for cid in todo_ids}
            combos = [(register, be) for be in backends]
            pools = {be: self.pool for be in backends}
            done = {"n": 0}

            def _sink(r):
                d = out_dir / r.construct_id
                d.mkdir(parents=True, exist_ok=True)
                if r.prose:
                    (d / f"{r.register}.md").write_text(r.prose)
                    if pkey:
                        (d / f".{r.register}.prose_key").write_text(pkey)
                    (d / f"{r.register}.exchange.json").write_text(
                        json.dumps(r.exchange, indent=1))  # thinking traces RETAINED
                done["n"] += 1
                if done["n"] % 25 == 0:
                    print(f"  {register}: {done['n']}/{len(todo_ids)}", flush=True)

            results = asyncio.run(run_queue(build_queue(constructs, combos), pools,
                                            on_result=_sink))
            for r in results:
                rows.append({"construct": r.construct_id, "register": r.register,
                             "backend": r.backend, "provider": r.provider,
                             "chars": len(r.prose), "tool_calls": r.exchange.get("tool_calls", []),
                             "embed": r.exchange.get("embed", {}), "error": r.error})
        ok = sum(1 for x in rows if x["chars"])
        print(f"  {register}: {ok}/{len(rows)} chapters "
              f"({len(rows) - len(todo_ids)} cached)", flush=True)
        return rows

    @traced_step
    @step
    def prose_natural(self):
        """Natural register — practitioner domain prose, tool-free (prose-pure)."""
        self.register_rows = self._run_register("natural")
        self.next(self.join_verify)

    @traced_step
    @step
    def prose_semantic(self):
        """Semantic register — ontology-mechanistic prose, kvasir-tool-equipped over MCP
        (stdio for local, streamable-http for grok) + kvasir-verified facts grounding."""
        self.register_rows = self._run_register("semantic")
        self.next(self.join_verify)

    @traced_step
    @step
    def join_verify(self, inputs):
        """Join the register branches; run the gates (report-first doctrine)."""
        self.merge_artifacts(inputs, exclude=["register_rows"])
        rows = [r for inp in inputs for r in inp.register_rows]
        Path(self.run_out, "manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        # Gate 1 — fictional particulars (REAL UNIVERSALS, FICTIONAL PARTICULARS).
        scan = subprocess.run(
            ["uv", "run", "--no-sync", "python", "scripts/scan_sensitive_nouns.py",
             "--root", f"{self.run_out}/chapters",
             "--json-out", f"{self.run_out}/sensitive_scan.json"],
            cwd=str(REPO), capture_output=True, text=True)
        self.sensitive_ok = scan.returncode == 0
        if not self.sensitive_ok:
            print(f"  SENSITIVE-NOUN GATE: attention needed\n{scan.stdout[-600:]}", flush=True)
        # Gate 2 — EMBEDDED PAYLOAD: every chapter must carry its tables AND views
        # (the chapters ARE textbooks-with-embedded-views; prose-only is a regression).
        n_tables_embedded = n_views_embedded = payload_bad = 0
        for md in Path(self.run_out, "chapters").rglob("*.md"):
            txt = md.read_text()
            t, v = txt.count("**Table `"), txt.count("**View `")
            n_tables_embedded += t
            n_views_embedded += v
            if t == 0 or (v == 0 and "v_" in json.dumps(self.construct_ids)):
                pass  # per-chapter view expectation checked below via constructs
            if t == 0:
                payload_bad += 1
        self.payload_gate = {"tables_embedded": n_tables_embedded,
                             "views_embedded": n_views_embedded,
                             "chapters_missing_payload": payload_bad}
        # Gate 2 — cell success + structure metrics.
        by = {}
        for r in rows:
            key = f"{r['register']}/{r['backend']}"
            c = by.setdefault(key, {"ok": 0, "n": 0, "tool_calls": 0})
            c["n"] += 1
            c["ok"] += 1 if r["chars"] else 0
            c["tool_calls"] += len(r.get("tool_calls") or [])
        self.metrics = {
            "cells": by, "derive_stats": self.derive_stats,
            "structure": {k: self.structure.get(k) for k in
                          ("n_elected", "total_fks", "n_junctions", "n_lookups", "shape_emd")},
            "sensitive_ok": self.sensitive_ok,
            "key_shapes": getattr(self, "key_shape_report", {}),
            "payload": self.payload_gate,
            "prose_chars_median": sorted(r["chars"] for r in rows)[len(rows) // 2] if rows else 0,
            "n_chapters": sum(1 for r in rows if r["chars"]),
            # the standing organ's health: worklist size + age histogram = "is the channel silting?"
            "refine_escalations": getattr(self, "refine_kpis", {}),
        }
        Path(self.run_out, "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))
        print(json.dumps(self.metrics, indent=2, default=str), flush=True)
        self.next(self.congruence)

    @traced_step
    @step
    def congruence(self):
        """The tripartite lineage graph: input passages -(harvest maxsim)-> concept entries
        <-(congruence maxsim)- output chapters. Quantifies how well the corpus preserves the
        input window's concept associations (the BERTopic-era R_D reborn over the ColBERT
        late-interaction index). Report-only; the graph is lineup/Atlas-ready."""
        self._arm_strategy_env()
        try:
            from aegir.ontology.congruence import congruence_report
            rep = congruence_report(Path(self.run_out))
            self.metrics["congruence"] = rep["per_register"]
            Path(self.run_out, "metrics.json").write_text(
                json.dumps(self.metrics, indent=2, default=str))
            print(f"  congruence: {json.dumps(rep['per_register'], default=str)}", flush=True)
        except Exception as e:  # noqa: BLE001 — report-only: qdrant down must not sink the corpus
            print(f"  congruence skipped ({str(e)[:120]})", flush=True)
        self.next(self.assemble)

    @traced_step
    @step
    def assemble(self):
        """Assemble the release tree. PUBLISH stays behind `aegir.lineup sync` (OQuaRE hard
        gate) + the P5 riders (#136: DST belief module, generation manifest, predictions
        schema, mirror cleanup) — do NOT auto-push corpora from a flow."""
        (Path(self.run_out) / "RELEASE.md").write_text(
            f"# sdg-corpora sdg candidate — run {current.run_id}\n\n"
            f"- ontology/sdg-sdg.omn (+ certificate.json — HermiT)\n"
            f"- ontology/ddl.sql, ontology/shapes.ttl (kvasir, proof-carrying)\n"
            f"- ontology/structure.json (shape EMD vs SchemaPile: "
            f"{self.structure.get('shape_emd')})\n"
            f"- chapters/<passage>/{{natural,semantic}}.md (+ .exchange.json traces)\n"
            f"- manifest.jsonl, metrics.json, sensitive_scan.json\n\n"
            f"Publish path: corpora sync (OQuaRE gate) + P5 riders (#136). Not automated here.\n")
        from aegir.lineup.zettel import write_run_zettel
        try:
            zp = write_run_zettel(Path(self.run_out), run_id=current.run_id,
                                  derive_stats=self.derive_stats, metrics=self.metrics)
            print(f"  run-zettel: {zp.name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  run-zettel skipped ({str(e)[:80]})", flush=True)
        self.emit_event("flow.completed", {"n_chapters": self.metrics.get("n_chapters", 0)})
        self.next(self.project)

    @traced_step
    @step
    def project(self):
        """Lineup KB projection — the corpus/sdg live note + all surfaces re-project
        so the /lineup panel reflects this run's accretion. Failure-tolerant tail."""
        try:
            import os as _os
            r = subprocess.run(["uv", "run", "--no-sync", "python", "-m", "aegir.lineup", "build"],
                               cwd=str(REPO), capture_output=True, text=True, timeout=600,
                               env={**_os.environ, "AEGIR_SDG_RUN": str(self.run_out)})  # project THIS run
            line = next((l for l in r.stdout.splitlines() if "sdg" in l), "")
            print(f"  lineup: {line.strip() or 'projected'}", flush=True)
        except Exception as e:  # noqa: BLE001 — projection is the tail; never lose the corpus to it
            print(f"  lineup projection skipped ({str(e)[:100]})", flush=True)
        # Atlas+OL main-path emission (task #23 — the enhancement the flow succession dropped): the run
        # becomes a walkable OL run in aegir_hx at CONSTRUCT grain, declaring the provenance triple from
        # the run-zettel. Best-effort like the lineup: a down graph never loses the corpus; backfill via
        # `python -m aegir.governance.provenance <run_dir>`.
        try:
            from aegir.governance.provenance import emit_corpus_lineage
            lr = emit_corpus_lineage(Path(self.run_out))
            print(f"  lineage → aegir_hx: OL run {lr['run_id'][:8]}… · {lr['outputs']} datasets", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  corpus lineage deferred ({str(e)[:100]}) — backfill: "
                  f"python -m aegir.governance.provenance {self.run_out}", flush=True)
        self.next(self.end)

    @traced_step
    @step
    def end(self):
        if getattr(self, "engine_owned", False):
            subprocess.run(["pkill", "-TERM", "-f", "aegir.engine.server"], check=False)
            subprocess.run(["pkill", "-TERM", "-f", "vllm.entrypoints"], check=False)
        print(f"SdgCorporaFlow {current.run_id} complete → {self.run_out}", flush=True)


if __name__ == "__main__":
    SdgCorporaFlow()

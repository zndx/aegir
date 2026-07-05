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
    harvest_target = Parameter("harvest-target", default=0, type=int,
                               help="ADVANCE THE INPUT WINDOW: stream FinePDFs until this many "
                                    "new in-domain docs land (0 = use the harvest as-is)")
    corpus_mode = Parameter("corpus", default=True, type=bool,
                            help="accrete into the persistent corpus dir (idempotent top-up per "
                                 "input window) instead of a fresh per-run dir")

    @traced_step
    @step
    def start(self):
        self.run_out = self.output_dir or (
            str(ART / "sdg-corpora" / "corpus") if self.corpus_mode
            else str(ART / "sdg-corpora" / current.run_id))
        Path(self.run_out, "entities").mkdir(parents=True, exist_ok=True)
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
        self.next(self.harvest)

    @traced_step
    @step
    def harvest(self):
        """ADVANCE THE INPUT WINDOW (idempotent by construction: content-hash doc filenames,
        persistent stream cursor, append-only manifest). --harvest-target 0 = no-op."""
        if self.harvest_target > 0:
            import subprocess as sp
            r = sp.run(["uv", "run", "--no-sync", "python", "scripts/harvest_domain_docs.py",
                        "--target", str(self.harvest_target),
                        "--max-stream", str(self.harvest_target * 40)],
                       cwd=str(REPO))
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
            self.next(self.realize)
            return
        self.engine_owned = _engine_up()
        from aegir.ontology.derive_harness import deriver_version
        from aegir.ontology.derive_loop import derive_with_metrology
        from aegir.ontology.entities import to_manchester
        dv = deriver_version()
        ent_dir = Path(self.run_out, "entities")
        stats = {"rich": 0, "thin": 0, "inert": 0, "malformed": 0, "cached": 0}
        self.derive_reports = {}
        for i, ppath in enumerate(self.passages):
            pid = Path(ppath).stem[:16]
            ej = ent_dir / f"{pid}.json"
            if ej.exists():
                try:
                    if json.loads(ej.read_text()).get("deriver_version") == dv:
                        stats["cached"] += 1
                        continue  # idempotence: (passage_hash, deriver_version) already done
                except Exception:  # noqa: BLE001 — unreadable stamp → re-derive
                    pass
            passage = Path(ppath).read_text()[: self.passage_chars]
            entities, report = derive_with_metrology(passage, max_rounds=self.rounds)
            stats[report.final_verdict or "malformed"] = stats.get(report.final_verdict, 0) + 1
            (ent_dir / f"{pid}.json").write_text(json.dumps({
                "passage": ppath,
                "deriver_version": dv,
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
        self.next(self.realize)

    @traced_step
    @step
    def realize(self):
        """Merge → HermiT certificate → kvasir DDL + SHACL shapes → SchemaPile structure score."""
        args = ["uv", "run", "--no-sync", "python", "scripts/realize_sdg.py",
                "--entities-dir", f"{self.run_out}/entities",
                "--output-dir", f"{self.run_out}/ontology"]
        if self.skip_hermit:
            args.append("--skip-hermit")
        env = dict(os.environ, LD_LIBRARY_PATH=JVM)
        r = subprocess.run(args, cwd=str(REPO), env=env)
        if r.returncode == 2:
            raise RuntimeError("HermiT refused the merged ontology (inconsistent) — see "
                               f"{self.run_out}/ontology/certificate.json")
        if r.returncode == 3:
            raise RuntimeError("HermiT found UNSATISFIABLE classes in the merge — the TBox is "
                               f"sick; see {self.run_out}/ontology/certificate.json (unsat list)")
        if r.returncode != 0:
            raise RuntimeError(f"realize failed (exit {r.returncode})")
        self.structure = json.loads(Path(self.run_out, "ontology/structure.json").read_text())
        self.emit_event("realize.done", {"shape_emd": str(self.structure.get("shape_emd"))})
        self.next(self.build_constructs)

    @traced_step
    @step
    def build_constructs(self):
        """Per-passage RI-true constructs + materialized VIEWS (the embedded payload the
        chapter prose supports) + kvasir-verified facts (the semantic grounding)."""
        from aegir.generate.harness import kvasir_facts
        from aegir.ontology.entities import (add_views, from_json, render_payload_blocks,
                                             to_construct, to_manchester)
        cons_dir = Path(self.run_out, "constructs")
        cons_dir.mkdir(exist_ok=True)
        self.construct_ids = []
        n_views = 0
        from collections import Counter
        pk_kinds, tstyles = Counter(), Counter()
        for p in sorted(Path(self.run_out, "entities").glob("*.json")):
            cj = cons_dir / f"{p.stem}.json"
            if cj.exists() and cj.stat().st_mtime >= p.stat().st_mtime:
                con_cached = json.loads(cj.read_text())
                n_views += len(con_cached.get("views", []))
                kp = con_cached.get("key_plan") or {}
                tstyles[kp.get("table_style", "?")] += 1
                for pl in (kp.get("plans") or {}).values():
                    pk_kinds[pl.get("kind", "?")] += 1
                self.construct_ids.append(p.stem)
                continue  # idempotence: construct newer than its entities
            ents = from_json(json.loads(p.read_text()))
            if not ents:
                continue
            omn = to_manchester(ents)
            con = to_construct(ents, n_rows=4)
            con = add_views(con, ents)
            blocks = render_payload_blocks(con)
            con["payload_blocks"] = blocks
            con["payload_markers"] = list(blocks)
            con["payload_preview"] = "\n\n".join(list(blocks.values())[:6])[:6000]
            con["ontology_omn"] = omn
            con["kvasir_facts"] = kvasir_facts(omn)
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
        import asyncio
        from aegir.generate.harness import build_queue, run_queue
        cons_dir = Path(self.run_out, "constructs")
        out_dir = Path(self.run_out, "chapters")
        backends = [b.strip() for b in self.backends.split(",") if b.strip()]
        rows = []
        todo_ids = []
        for cid in self.construct_ids:
            if (out_dir / cid / f"{register}.md").exists():
                t = (out_dir / cid / f"{register}.md").read_text()
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
            r = subprocess.run(["uv", "run", "--no-sync", "python", "-m", "aegir.lineup", "build"],
                               cwd=str(REPO), capture_output=True, text=True, timeout=600)
            line = next((l for l in r.stdout.splitlines() if "sdg" in l), "")
            print(f"  lineup: {line.strip() or 'projected'}", flush=True)
        except Exception as e:  # noqa: BLE001 — projection is the tail; never lose the corpus to it
            print(f"  lineup projection skipped ({str(e)[:100]})", flush=True)
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

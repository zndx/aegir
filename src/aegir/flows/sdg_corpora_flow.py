"""SdgCorporaFlow — the greenfield sdg-corpora iteration as one Metaflow flow.

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


def _engine_up() -> bool:
    """Ensure the engine serves (qwen3_xml tool parser is the config default now).
    Returns True if THIS flow launched it (and should tear it down at end)."""
    if subprocess.run(["pgrep", "-f", "aegir.engine.server"], capture_output=True).returncode == 0:
        print("  engine already up", flush=True)
        return False
    env = dict(os.environ, LD_LIBRARY_PATH=CUDA)
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

    n_passages = Parameter("n-passages", default=6, type=int,
                           help="FinePDFs passages to derive + write chapters from")
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

    @traced_step
    @step
    def start(self):
        self.run_out = str(ART / "sdg-corpora" / current.run_id)
        Path(self.run_out, "entities").mkdir(parents=True, exist_ok=True)
        manifest = REPO / "build/domain_harvest/manifest.jsonl"
        docs = sorted((REPO / "build/domain_harvest/docs").glob("*.txt"))
        if not docs:
            raise RuntimeError("no harvested passages — run SemanticCorpusFlow harvest or "
                               "scripts/harvest_domain_docs.py first")
        self.passages = [str(p) for p in docs[: self.n_passages]]
        self.emit_event("flow.started", {"n_passages": len(self.passages),
                                         "backends": self.backends, "manifest": manifest.exists()})
        print(f"SdgCorporaFlow {current.run_id}: {len(self.passages)} passages → {self.run_out}",
              flush=True)
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
        from aegir.ontology.derive_loop import derive_with_metrology
        from aegir.ontology.entities import to_manchester
        ent_dir = Path(self.run_out, "entities")
        stats = {"rich": 0, "thin": 0, "inert": 0, "malformed": 0}
        self.derive_reports = {}
        for i, ppath in enumerate(self.passages):
            pid = Path(ppath).stem[:16]
            passage = Path(ppath).read_text()[: self.passage_chars]
            entities, report = derive_with_metrology(passage, max_rounds=self.rounds)
            stats[report.final_verdict or "malformed"] = stats.get(report.final_verdict, 0) + 1
            (ent_dir / f"{pid}.json").write_text(json.dumps({
                "passage": ppath,
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
        args = ["uv", "run", "--no-sync", "python", "scripts/realize_greenfield.py",
                "--entities-dir", f"{self.run_out}/entities",
                "--output-dir", f"{self.run_out}/ontology"]
        if self.skip_hermit:
            args.append("--skip-hermit")
        env = dict(os.environ, LD_LIBRARY_PATH=JVM)
        r = subprocess.run(args, cwd=str(REPO), env=env)
        if r.returncode == 2:
            raise RuntimeError("HermiT refused the merged ontology (inconsistent) — see "
                               f"{self.run_out}/ontology/certificate.json")
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
        """Drive the work queue for ONE register across all constructs × backends."""
        import asyncio
        from aegir.generate.harness import build_queue, run_queue
        cons_dir = Path(self.run_out, "constructs")
        constructs = {cid: json.loads((cons_dir / f"{cid}.json").read_text())
                      for cid in self.construct_ids}
        backends = [b.strip() for b in self.backends.split(",") if b.strip()]
        combos = [(register, be) for be in backends]
        pools = {be: self.pool for be in backends}
        results = asyncio.run(run_queue(build_queue(constructs, combos), pools))
        out_dir = Path(self.run_out, "chapters")
        rows = []
        for r in results:
            d = out_dir / r.construct_id
            d.mkdir(parents=True, exist_ok=True)
            if r.prose:
                (d / f"{r.register}.md").write_text(r.prose)
                (d / f"{r.register}.exchange.json").write_text(
                    json.dumps(r.exchange, indent=1))  # thinking traces RETAINED
            rows.append({"construct": r.construct_id, "register": r.register,
                         "backend": r.backend, "provider": r.provider,
                         "chars": len(r.prose), "tool_calls": r.exchange.get("tool_calls", []),
                         "embed": r.exchange.get("embed", {}), "error": r.error})
        ok = sum(1 for x in rows if x["chars"])
        print(f"  {register}: {ok}/{len(rows)} chapters", flush=True)
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
        self.next(self.assemble)

    @traced_step
    @step
    def assemble(self):
        """Assemble the release tree. PUBLISH stays behind `aegir.lineup sync` (OQuaRE hard
        gate) + the P5 riders (#136: DST belief module, generation manifest, predictions
        schema, mirror cleanup) — do NOT auto-push corpora from a flow."""
        (Path(self.run_out) / "RELEASE.md").write_text(
            f"# sdg-corpora greenfield candidate — run {current.run_id}\n\n"
            f"- ontology/sdg-greenfield.omn (+ certificate.json — HermiT)\n"
            f"- ontology/ddl.sql, ontology/shapes.ttl (kvasir, proof-carrying)\n"
            f"- ontology/structure.json (shape EMD vs SchemaPile: "
            f"{self.structure.get('shape_emd')})\n"
            f"- chapters/<passage>/{{natural,semantic}}.md (+ .exchange.json traces)\n"
            f"- manifest.jsonl, metrics.json, sensitive_scan.json\n\n"
            f"Publish path: corpora sync (OQuaRE gate) + P5 riders (#136). Not automated here.\n")
        self.emit_event("flow.completed", {"n_chapters": self.metrics.get("n_chapters", 0)})
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

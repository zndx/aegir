"""SemanticCorpusFlow — the end-to-end content-first pipeline as one Metaflow flow.

Each stage is a ``@traced_step`` (Step→OTel→NiFi) that shells out to the proven script with the right
environment; artifacts flow between steps as run-dir paths on ``self``. Steps run LOCALLY (not as K8s pods)
so they keep GPU / vLLM-engine / JVM / qdrant / postgres access — the Metaflow service plane on RKE2 only
provides versioned run metadata + artifact storage + the UI. The flow owns the engine lifecycle (up for
derive/chapter-gen, down at the end) and sets ``LD_LIBRARY_PATH`` per step (jvm-libs for the membrane,
cuda-driver-libs for the engine).

    import aegir.metaflow_patch
    just metaflow            # = python -m aegir.flows.semantic_corpus_flow run --n-docs 8 ...
"""
import os  # noqa: E402

import aegir.metaflow_patch  # noqa: F401  — MUST precede the metaflow import (404-retry patch)
from metaflow import FlowSpec, Parameter, current, step  # noqa: E402

from aegir.flows.config import apply_metaflow_config  # noqa: E402
from aegir.flows.trace import TracedFlow, traced_step  # noqa: E402

# rke2 = service plane on RKE2 (versioned UI); local = local metadata + /raid datastore (no service plane,
# runs the full DAG on the GPUs without K8s). AEGIR_METAFLOW_MODE selects; rke2 is the default.
apply_metaflow_config(os.environ.get("AEGIR_METAFLOW_MODE", "rke2"))

import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
JVM = f"{REPO}/build/jvm-libs"
CUDA = f"{REPO}/build/cuda-driver-libs"
ART = Path("/raid/checkpoints/aegir-artifacts")


def _run(args: "list[str]", *, ld: str = "", extra_env: "dict | None" = None) -> None:
    """Run a pipeline stage as `uv run --no-sync python <args>` with optional LD_LIBRARY_PATH; raise on fail."""
    env = dict(os.environ)
    if ld:
        env["LD_LIBRARY_PATH"] = ld
    if extra_env:
        env.update(extra_env)
    cmd = ["uv", "run", "--no-sync", "python", *args]
    print(f"  $ {' '.join(cmd)}  (LD={ld or '-'})", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed (exit {r.returncode}): {' '.join(args[:2])}")


def _engine_up() -> None:
    """Launch the vLLM engine detached (cuda-driver-libs) if not already serving; wait for the gRPC port."""
    if subprocess.run(["pgrep", "-f", "aegir.engine.server"], capture_output=True).returncode == 0:
        print("  engine already up", flush=True)
        return
    env = dict(os.environ, LD_LIBRARY_PATH=CUDA)
    log = (REPO / "build" / "engine_flow.log").open("w")
    subprocess.Popen(["uv", "run", "--no-sync", "python", "-m", "aegir.engine.server"],
                     cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if subprocess.run(["bash", "-c", "grep -q 'gRPC listening' build/engine_flow.log 2>/dev/null"],
                          cwd=str(REPO)).returncode == 0:
            print("  engine gRPC up", flush=True)
            return
        time.sleep(2)
    print("  WARN: engine gRPC not confirmed; proceeding (first Complete will block on model load)", flush=True)


def _engine_down() -> None:
    subprocess.run(["pkill", "-TERM", "-f", "aegir.engine.server"], check=False)
    subprocess.run(["pkill", "-TERM", "-f", "vllm.entrypoints"], check=False)


class SemanticCorpusFlow(TracedFlow, FlowSpec):
    """Content-first regen: harvest LIMS docs → derive ontology → membrane/promote → realized DDL →
    content-first chapters (meaningful views) → verify → lineup → Atlas. One versioned generation."""

    n_docs = Parameter("n-docs", default=8, type=int, help="harvested docs to derive + write chapters from")
    domain = Parameter("domain", default="Laboratory Information Management", help="SKOS domain subtree")
    mix = Parameter("mix", default="engine/instruct:1.0", help="chapter-gen LLM mix (local Qwen3.6 default)")
    oversample = Parameter("oversample", default=12, type=int, help="harvest stream multiplier for the domain gate")
    max_tokens = Parameter("max-tokens", default=24000, type=int,
                           help="chapter-gen output cap; needs a 32768-ctx engine for complete reasoning traces")

    @traced_step
    @step
    def start(self):
        self.run_out = str(ART / "flow" / current.run_id)
        Path(self.run_out).mkdir(parents=True, exist_ok=True)
        self.emit_event("flow.started", {"n_docs": self.n_docs, "domain": self.domain})
        print(f"SemanticCorpusFlow run {current.run_id} → {self.run_out}", flush=True)
        self.next(self.build_domain_index)

    @traced_step
    @step
    def build_domain_index(self):
        # SKOS vocab is the source-of-truth in-repo; (re)build the ColBERT/Qdrant domain index over it.
        _run(["scripts/build_domain_index.py", "build"])
        self.next(self.harvest)

    @traced_step
    @step
    def harvest(self):
        _run(["scripts/harvest_domain_docs.py", "--domain", self.domain,
              "--target", str(self.n_docs), "--max-stream", str(self.n_docs * self.oversample)])
        self.next(self.derive)

    @traced_step
    @step
    def derive(self):
        _engine_up()
        _run(["scripts/derive_ontology.py", "--from-harvest", "--n-docs", str(self.n_docs),
              "--write-candidates"], ld=JVM)
        self.next(self.promote)

    @traced_step
    @step
    def promote(self):
        _run(["scripts/promote_candidates.py"], ld=JVM)   # CPU re-gate → aggregate HermiT → combined.json
        self.next(self.build_spine)

    @traced_step
    @step
    def build_spine(self):
        _run(["scripts/build_ddl_spine.py", "--realize", "--output-dir", f"{self.run_out}/spine"])
        self.next(self.generate_chapters)

    @traced_step
    @step
    def generate_chapters(self):
        _engine_up()
        _run(["scripts/generate_chapter.py", "--from-harvest", "--n-chapters", str(self.n_docs),
              "--mix", self.mix, "--realize-schemas", "--max-tokens", str(self.max_tokens),
              "--output", f"{self.run_out}/chapters"])
        self.chapters_run = next((str(p.parent) for p in
                                 Path(f"{self.run_out}/chapters").glob("*/chapters.parquet")), "")
        self.next(self.verify)

    @traced_step
    @step
    def verify(self):
        if self.chapters_run:
            _run(["scripts/verify_chapters.py", "--chapters-run", self.chapters_run])  # content-first: no audit
        self.next(self.project)

    @traced_step
    @step
    def project(self):
        # lineup KB projection (now incl. 08_derived) + Atlas relational/lineage projection.
        try:
            _run(["-m", "aegir.lineup", "build"])
        except Exception as e:  # noqa: BLE001 — projection is the tail; never lose the corpus to it
            print(f"  lineup build skipped ({e})", flush=True)
        try:
            _run(["scripts/project_atlas_ddl.py", "--per-family", "2"])
        except Exception as e:  # noqa: BLE001 — Atlas may be down (devenv services not restarted)
            print(f"  Atlas projection skipped ({e})", flush=True)
        self.next(self.end)

    @traced_step
    @step
    def end(self):
        _engine_down()
        self.emit_event("flow.completed", {"chapters_run": getattr(self, "chapters_run", "")})
        print(f"SemanticCorpusFlow {current.run_id} complete · chapters: {getattr(self, 'chapters_run', '')}", flush=True)


if __name__ == "__main__":
    SemanticCorpusFlow()

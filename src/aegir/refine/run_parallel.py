"""Parallel, multi-provider agent-mediated corpus refinement — the Grok objective.

Fans out N refinement workers (one per provider) over the live corpus. Each is a ``run_scale --worker`` process
pinned to a proposer backend — ``local`` (the vLLM engine) | ``grok`` (Grok Build, UNMETERED subscription via
ACP) | ``xai-api`` (metered XAI_API_KEY) — striding DISJOINT chapters (worker i takes chapters i, i+N, i+2N…).

Why processes: each worker gets its own JVM, so the HermiT membrane never contends across workers; and a
``grok`` worker hits the subscription WS relay while a ``local`` worker hits the engine — different backends, so
they run TRULY in parallel without sharing the engine's seq budget. The hx (``raw.exchange``) + OL/AGE lineage
is append-safe + idempotent across workers, and every exchange is tagged with its provider. After all workers
exit, the orchestrator merges the per-worker manifests, emits the corpus parquet, and re-projects the lineup.

Run (engine up for a ``local`` worker; ``grok login`` done once for a ``grok`` worker):
    LD_LIBRARY_PATH=build/jvm-libs uv run --no-sync python -m aegir.refine.run_parallel \
        --providers local,grok --n-chapters 120 --realize
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from aegir.refine.run_scale import _emit_corpus_parquet, _project_lineup

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="local",
                    help="comma list, one WORKER per provider: local | grok | xai-api (e.g. local,grok)")
    ap.add_argument("--n-chapters", type=int, default=60, help="upper bound on TOTAL chapters across workers")
    ap.add_argument("--n-templates", type=int, default=3)
    ap.add_argument("--realize", action="store_true",
                    help="realized schemas (EAV/junction/star) — broad relational-construct space")
    ap.add_argument("--token-target", type=int, default=0, help="per-worker token target (0 = run to --n-chapters)")
    ap.add_argument("--out", default=os.environ.get("AEGIR_REFINE_OUT", "/raid/build/aegir/path-a/refine_scale"))
    a = ap.parse_args()
    providers = [p.strip() for p in a.providers.split(",") if p.strip()]
    n = len(providers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"PARALLEL refine: {n} worker(s) {providers} over ≤{a.n_chapters} chapters (stride {n}), "
          f"realize={a.realize} → {out}", flush=True)

    procs = []
    for i, prov in enumerate(providers):
        cmd = [sys.executable, "-m", "aegir.refine.run_scale", "--worker",
               "--backend", prov, "--stride", str(n), "--worker-id", str(i),
               "--n-chapters", str(a.n_chapters), "--n-templates", str(a.n_templates), "--out", str(out)]
        if a.realize:
            cmd.append("--realize")
        if a.token_target:
            cmd += ["--token-target", str(a.token_target)]
        logf = open(out / f"worker_{i}_{prov}.log", "w")
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env={**os.environ}, cwd=str(ROOT))
        procs.append((i, prov, p, logf))
        print(f"  → worker {i} [{prov}] pid {p.pid}  (log: worker_{i}_{prov}.log)", flush=True)

    t0 = time.time()
    for i, prov, p, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"  ✓ worker {i} [{prov}] exited rc={rc}  ({time.time() - t0:.0f}s)", flush=True)

    # merge the per-worker manifests into the canonical one, then finalize ONCE (the workers skipped this)
    merged: list[dict] = []
    for i, _prov, _p, _l in procs:
        wm = out / f"manifest_w{i}.jsonl"
        if wm.exists():
            merged += [json.loads(ln) for ln in wm.read_text().splitlines() if ln.strip()]
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in merged) + "\n")
    n_parq = _emit_corpus_parquet(out)
    _project_lineup()

    promoted = sum(1 for m in merged if m.get("outcome") == "promote")
    toks = sum(m.get("tokens", 0) for m in merged)
    nex = sum(m.get("exchanges", 0) for m in merged)
    print(f"\nPARALLEL RUN done: {promoted}/{len(merged)} promoted across {providers}, ~{toks} corpus tokens, "
          f"{nex} exchanges → raw.exchange, {n_parq} surface-records → chapters.parquet (lineup re-projected), "
          f"{time.time() - t0:.0f}s. → {out}", flush=True)
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

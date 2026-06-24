"""The overnight agentic MIXED-REGISTER scale run — the refinement loop over the live corpus.

Iterates N distinct live chapters (``live_draft`` over template windows), each refined by the membrane-gated
loop with the REAL agent (scaffold agency over the tables) + DUAL-REGISTER output (natural + semantic prose
over the same RI-true tables), and EVERY agent exchange + COMMIT run-event captured into the hx
(``raw.exchange``) + OL/AGE lineage plane. Resilient: a per-chapter failure (agent/gate/engine) is logged to
the manifest and skipped — it never kills the run; ``--start`` resumes from an offset after an engine hiccup.

Honest scope: live ontology tables are already value-clean (the mixing is provably non-contaminating — see
audit_mixing), so on live data the loop's work is prose + admissibility confirmation + lineage; the scaffold
agency fires only where a real defect exists. The product is the dual-register corpus + its full provenance.

Run (overnight, engine up):
    LD_LIBRARY_PATH=build/jvm-libs uv run --no-sync python -m aegir.refine.run_scale --n-chapters 200
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from aegir.refine.lineage import LineageRecorder
from aegir.refine.live import live_draft
from aegir.refine.loop import run_refinement

ROOT = Path(__file__).resolve().parents[3]
FORK_PY = ROOT / "components" / "oss-mistral-cli" / ".venv" / "bin" / "python"


def _subproc(construct: dict, mode: str, register: str, feedback: dict, backend: str = "local") -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["AEGIR_FORK_DIR"] = str(ROOT / "components" / "oss-mistral-cli")
    env["AEGIR_PROPOSE_BACKEND"] = backend   # local engine | grok (Grok Build, unmetered) | xai-api (metered)
    req = json.dumps({"construct": construct, "mode": mode, "register": register, "feedback": feedback})
    p = subprocess.run([str(FORK_PY), "-m", "aegir.refine._propose"], input=req,
                       capture_output=True, text=True, env=env, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-400:])
    return json.loads(p.stdout)


def _refine_one(offset: int, n_templates: int, out_dir: str, realize: bool = False, backend: str = "local") -> dict:
    rec = LineageRecorder()

    def scaffold(c, obj, fb):
        d = _subproc(c, "edits", "natural", fb, backend)
        rec.record_exchange(d.get("_exchange", {}), source_context={"objective": obj, "mode": "edits"})
        return d.get("edits", [])

    def pnat(c, fb=None):
        d = _subproc(c, "prose", "natural", fb or {}, backend)
        rec.record_exchange(d.get("_exchange", {}), source_context={"objective": "fix_prose", "register": "natural"})
        return d.get("prose", "")

    def psem(c):
        d = _subproc(c, "prose", "semantic", {}, backend)
        rec.record_exchange(d.get("_exchange", {}), source_context={"objective": "dual_register", "register": "semantic"})
        return d.get("prose", "")

    ch = live_draft(n_templates=n_templates, offset=offset, realize=realize)
    res = run_refinement(ch, propose_fn=pnat, scaffold_propose=scaffold, dual_prose_fn=psem,
                         register="natural", lineage=rec, max_iters=8, commit_dir=out_dir)
    toks = 0
    for _, p in res.get("surfaces", []):
        try:
            c = json.loads(Path(p).read_text())
            toks += len(c.get("prose", "")) + sum(
                len(str(cell.get("value", ""))) + len(col.get("name", ""))
                for t in c.get("tables", []) for col in t.get("columns", []) for cell in col.get("cells", []))
        except Exception:  # noqa: BLE001
            pass
    return {"template_id": ch["template_id"], "offset": offset, "outcome": res["outcome"],
            "surfaces": [r for r, _ in res["surfaces"]], "exchanges": len(rec.exchange_ids),
            "lineage": bool(res.get("lineage")), "tokens": toks // 4, "metrics": res["refined_metrics"]}


def _emit_corpus_parquet(out: Path) -> int:
    """Build a ``chapters.parquet`` (the lineup's content format) from the dual-register surface files, so the
    refined corpus surfaces in the register-mix lineup — one record per surface, register per-record."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    recs = []
    for f in sorted((out / "current").glob("*.json")):
        if f.name.endswith(".metrics.json"):
            continue
        try:
            c = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        recs.append({"chapter_id": f.stem, "response_text": c.get("prose", ""),
                     "template_ids": [str(t) for t in (c.get("template_ids") or [])],
                     "family": c.get("family"), "register": c.get("register", "natural"),
                     "model": "engine-refine", "target_topic_id": None})
    if recs:
        pq.write_table(pa.Table.from_pylist(recs), str(out / "chapters.parquet"))
    return len(recs)


def _project_lineup() -> None:
    """Re-project the KB lineup so the refined dual-register corpus surfaces (best-effort)."""
    try:
        subprocess.run(["uv", "run", "--no-sync", "python", "-m", "aegir.lineup", "build"],
                       cwd=str(ROOT), timeout=900, check=False)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chapters", type=int, default=50, help="upper bound on chapters")
    ap.add_argument("--n-templates", type=int, default=3)
    ap.add_argument("--realize", action="store_true",
                    help="realized schemas (EAV/junction/star) — broad relational-construct space + ~3-4x tokens/chapter")
    ap.add_argument("--token-target", type=int, default=0,
                    help="stop once accumulated corpus tokens reach this (0 = run to --n-chapters)")
    ap.add_argument("--start", type=int, default=0, help="resume from this chapter index after a hiccup")
    ap.add_argument("--backend", default="local",
                    help="proposer backend: local (engine) | grok (Grok Build, unmetered) | xai-api (metered)")
    ap.add_argument("--stride", type=int, default=1, help="parallel: process every Nth chapter (N = #workers)")
    ap.add_argument("--worker-id", type=int, default=0, help="parallel: this worker's offset within the stride")
    ap.add_argument("--worker", action="store_true",
                    help="worker mode: per-worker manifest + skip parquet/projection (run_parallel finalizes)")
    ap.add_argument("--out", default=os.environ.get("AEGIR_REFINE_OUT", "/raid/build/aegir/path-a/refine_scale"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    mpath = out / (f"manifest_w{a.worker_id}.jsonl" if a.worker else "manifest.jsonl")
    if mpath.exists():
        manifest = [json.loads(ln) for ln in mpath.read_text().splitlines() if ln.strip()]
    t0 = time.time()
    total_tokens = sum(m.get("tokens", 0) for m in manifest)
    tag = f"w{a.worker_id}/{a.backend}"
    for i in range(a.worker_id, a.n_chapters, a.stride):   # strided so parallel workers cover disjoint chapters
        if i < a.start:
            continue
        try:
            r = _refine_one(i * a.n_templates, a.n_templates, str(out), realize=a.realize, backend=a.backend)
            total_tokens += r.get("tokens", 0)
            print(f"[{tag} {i + 1}/{a.n_chapters}] {r['template_id']} {r['outcome']} tok~{r.get('tokens', 0)} "
                  f"(cum ~{total_tokens}) ex={r['exchanges']} lineage={r['lineage']}", flush=True)
        except Exception as e:  # noqa: BLE001 — a chapter failure must never kill the run
            r = {"offset": i * a.n_templates, "index": i, "error": str(e)[:200]}
            print(f"[{tag} {i + 1}/{a.n_chapters}] FAILED: {str(e)[:120]}", flush=True)
        manifest.append(r)
        mpath.write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
        if not a.worker and (i + 1) % 10 == 0:   # standalone keeps the parquet current; orchestrator finalizes
            _emit_corpus_parquet(out)
        if a.token_target and total_tokens >= a.token_target:
            print(f"[{tag}] token target {a.token_target} reached (~{total_tokens} tokens) — stopping.", flush=True)
            break
    promoted = sum(1 for m in manifest if m.get("outcome") == "promote")
    nex = sum(m.get("exchanges", 0) for m in manifest)
    if a.worker:                                  # run_parallel merges the per-worker manifests + finalizes
        print(f"\n[{tag}] worker done: {promoted} promoted, ~{total_tokens} tokens, {nex} exchanges, "
              f"{time.time() - t0:.0f}s.", flush=True)
        return 0
    n_parq = _emit_corpus_parquet(out)        # the refined corpus in the lineup's content format
    _project_lineup()                          # re-project so the new dual-register content surfaces
    nlin = sum(1 for m in manifest if m.get("lineage"))
    print(f"\nSCALE RUN done: {promoted} promoted, ~{total_tokens} corpus tokens, {nex} exchanges → "
          f"raw.exchange, {nlin} run-events, {n_parq} surface-records → chapters.parquet (lineup re-projected), "
          f"{time.time() - t0:.0f}s. → {out}")
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

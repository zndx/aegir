"""PathATrainingFlow — Path A (World-v3-augmentation continued-pretrain) as one Metaflow flow.

Wraps the proven S1–S3 scripts so the whole data-value-isolation experiment is one CI-gated command
(``just train-path-a``): tokenize is a precondition (binidx prebuilt via tokenize_corpus_binidx.py), then a
foreach over α-arms continue-pretrains a warm-started RWKV-7 (each arm pinned to its own GPU — run with
``--max-workers N`` to parallelize), then byte-perplexity eval (general + relational) across baseline +
every arm, then a metrics ledger. Each ``@traced_step`` is Step→OTel; the trainer self-re-execs for the
nix CUDA libs, so the flow just passes ``CUDA_VISIBLE_DEVICES`` + the alloc knob.

    import aegir.metaflow_patch
    AEGIR_METAFLOW_MODE=local just train-path-a --alphas 0,0.02 --max-workers 2
"""
import os  # noqa: E402

import aegir.metaflow_patch  # noqa: F401  — MUST precede the metaflow import (404-retry patch)
from metaflow import FlowSpec, Parameter, current, step  # noqa: E402

from aegir.flows.config import apply_metaflow_config  # noqa: E402
from aegir.flows.trace import TracedFlow, traced_step  # noqa: E402

apply_metaflow_config(os.environ.get("AEGIR_METAFLOW_MODE", "rke2"))

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def _run(args: "list[str]", *, env_extra: "dict | None" = None) -> None:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = ["uv", "run", "--no-sync", "python", *args]
    print(f"  $ {' '.join(cmd)}  ({env_extra or ''})", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed (exit {r.returncode}): {' '.join(args[:2])}")


def _cuda_env(gpu) -> dict:
    """CUDA env for a GPU subprocess: pin the device + put libcuda (cuda-driver-libs) and the nix cudart
    (AEGIR_CUDA_HOME/lib64) on LD_LIBRARY_PATH (APPEND — keep nix glibc). The trainer self-re-execs for this
    too, but the eval doesn't, so set it here for both."""
    ch = os.environ.get("AEGIR_CUDA_HOME", "")
    ld = f"{ch}/lib64:{REPO}/build/cuda-driver-libs:" + os.environ.get("LD_LIBRARY_PATH", "")
    return {"CUDA_VISIBLE_DEVICES": str(gpu), "LD_LIBRARY_PATH": ld,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


class PathATrainingFlow(TracedFlow, FlowSpec):
    """Continue-pretrain RWKV-7 ± augmentation across α-arms; eval general+relational; ledger. Path A."""

    load = Parameter("load", default="/raid/datasets/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth",
                     help="warm-start checkpoint")
    base = Parameter("base", default="/raid/build/aegir/path-a/base/subsample_100k", help="base binidx prefix")
    aug = Parameter("aug", default="/raid/build/aegir/path-a/aug/pathA_aug", help="augmentation binidx prefix")
    alphas = Parameter("alphas", default="0,0.02", help="comma-separated augmentation proportions (arms)")
    tokens = Parameter("tokens", default=110_000_000, type=int)
    ctx_len = Parameter("ctx-len", default=2048, type=int)
    batch = Parameter("batch", default=4, type=int)
    lr = Parameter("lr", default=6e-5, type=float)
    out_root = Parameter("out", default="/raid/build/aegir/path-a/flow")
    general_data = Parameter("general-data", default="/raid/datasets/fineweb-edu")
    relational_data = Parameter("relational-data", default="/raid/build/aegir/path-a/eval_aug")

    @traced_step
    @step
    def start(self):
        alphas = [float(x) for x in str(self.alphas).split(",") if x.strip() != ""]
        # one GPU per arm (parallel under --max-workers); arms keyed by α
        self.arms = [{"alpha": a, "gpu": i} for i, a in enumerate(alphas)]
        Path(self.out_root).mkdir(parents=True, exist_ok=True)
        self.emit_event("path_a.start", {"alphas": alphas, "tokens": self.tokens})
        print(f"PathA {current.run_id}: arms={self.arms} tokens={self.tokens}", flush=True)
        self.next(self.train, foreach="arms")

    @traced_step
    @step
    def train(self):
        arm = self.input
        out = f"{self.out_root}/arm_a{arm['alpha']}"
        args = ["scripts/continue_pretrain_rwkv7.py", "--load", self.load, "--base", self.base,
                "--alpha", str(arm["alpha"]), "--tokens", str(self.tokens), "--ctx-len", str(self.ctx_len),
                "--batch", str(self.batch), "--lr", str(self.lr), "--out", out]
        if arm["alpha"] > 0:
            args += ["--aug", self.aug]
        _run(args, env_extra=_cuda_env(arm["gpu"]))
        self.arm = {"alpha": arm["alpha"], "ckpt": f"{out}/final.pth"}
        self.next(self.evaluate)

    @traced_step
    @step
    def evaluate(self, inputs):
        # params are available directly; only the per-arm `arm` artifact diverges → read via inputs.
        ckpts = {"baseline": self.load}
        for inp in inputs:
            ckpts[f"a{inp.arm['alpha']}"] = inp.arm["ckpt"]
        scores: dict = {}
        for name, ck in ckpts.items():
            scores[name] = {}
            for reg, dd in (("general", self.general_data), ("relational", self.relational_data)):
                oj = f"{self.out_root}/eval_{name}_{reg}.json"
                _run(["scripts/eval_rwkv7_baseline.py", "--checkpoint", ck, "--data-dir", dd,
                      "--max-bytes", "600000", "--ctx-len", str(self.ctx_len), "--out", oj],
                     env_extra=_cuda_env(0))
                try:
                    scores[name][reg] = json.loads(Path(oj).read_text()).get("bits_per_byte")
                except Exception:  # noqa: BLE001
                    scores[name][reg] = None
        self.scores = scores
        self.next(self.end)

    @traced_step
    @step
    def end(self):
        ledger = {"alphas": str(self.alphas), "tokens": self.tokens, "ctx_len": self.ctx_len,
                  "batch": self.batch, "lr": self.lr, "bits_per_byte": self.scores}
        Path(f"{self.out_root}/metrics_ledger.json").write_text(json.dumps(ledger, indent=1))
        self.emit_event("path_a.done", {"scores": self.scores})
        print(f"PathATrainingFlow {current.run_id} complete · bits/byte: {self.scores}", flush=True)


if __name__ == "__main__":
    PathATrainingFlow()

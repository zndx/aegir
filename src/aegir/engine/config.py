"""Engine config — the capability→model registry (the engine owns this; workloads name capabilities).

PROVISIONAL (per [[provisional_scaffolding_not_goals]]): the single `instruct` capability + the chosen
model are scaffolding for the Semantic-Layer-Upkeep loop, not a fixed design. Add capabilities (embedding,
reasoning, …) and let model selection evolve with workloads — the point of the capability indirection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# vLLM runs in a DEDICATED venv (isolated from aegir's patched main env): the gRPC server runs in aegir's
# main venv (torch-2.9.1/cu128, flash-attn/mamba untouched); only vLLM subprocesses use THIS python.
# PROVISIONAL: default to the sibling agent-evals `.venv-vllm` — the env already PROVEN on this machine to
# serve Qwen3.6-35B-A3B (vLLM 0.19.0, torch 2.10.0+cu129, native qwen3_5_moe). cu129 runs on the system's
# 12.8 driver via CUDA minor-version compat (verified: 6 GPUs, alloc OK). aegir's own `.venv-vllm` is
# vLLM 0.22.1/torch cu130 — a *major* CUDA mismatch the 12.8 driver rejects, so it is NOT the default.
# (Override with AEGIR_VLLM_PYTHON; to make aegir self-contained, recreate `.venv-vllm` mirroring cu129.)
_SIBLING_VLLM = Path(__file__).resolve().parents[4] / "agent-evals" / ".venv-vllm" / "bin" / "python"
_OWN_VLLM = Path(__file__).resolve().parents[3] / ".venv-vllm" / "bin" / "python"
VLLM_PYTHON = os.environ.get(
    "AEGIR_VLLM_PYTHON", str(_SIBLING_VLLM if _SIBLING_VLLM.exists() else _OWN_VLLM))

# The HF cache dir that directly contains the `models--*` trees (note: this layout has NO `/hub` subdir).
DEFAULT_HF_HUB_CACHE = os.environ.get("AEGIR_HF_HUB_CACHE", "/raid/cache/rch/huggingface")

ENGINE_GRPC_PORT = int(os.environ.get("AEGIR_ENGINE_PORT", "50151"))
VLLM_BASE_PORT = int(os.environ.get("AEGIR_VLLM_BASE_PORT", "8100"))
FIRST_GPU = int(os.environ.get("AEGIR_ENGINE_GPU0", "0"))
# Federation (roadmap): if set (host:port), the client delegates capability requests to a remote
# (e.g. Gaius) engine instead of the local one.
FEDERATE_TARGET = os.environ.get("AEGIR_ENGINE_FEDERATE", "")


@dataclass
class ModelSpec:
    model: str                              # vLLM model id (resolved from DEFAULT_HF_HUB_CACHE) or a local path
    tensor_parallel_size: int = 4           # FP8 35B-A3B fits TP=4 (~9GB/GPU) on the 6×24GB; configurable
    gpu_memory_utilization: float = 0.90
    extra_args: list[str] = field(default_factory=list)


# capability → model. Qwen3.6-35B-A3B-FP8 (FP8 ⇒ fits the 6×24GB; tensor-parallel across the local GPUs).
CAPABILITY_MODELS: dict[str, ModelSpec] = {
    "instruct": ModelSpec(
        model=os.environ.get("AEGIR_INSTRUCT_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8"),
        tensor_parallel_size=int(os.environ.get("AEGIR_ENGINE_TP", "4")),
        gpu_memory_utilization=float(os.environ.get("AEGIR_ENGINE_GPU_MEM", "0.90")),
        # Sized for THINKING-trace retention (Qwen3.x reasons verbosely; the trace is a corpus value-add,
        # so we keep it and give it room) WITHOUT OOM: 16384 ctx × 8 seqs has the SAME KV footprint as the
        # proven-safe 8192×16 (product 131072) and a SMALLER sampler warmup (8-wide logits vs the 256
        # default that OOM'd). The native 262144 ctx would demand a huge KV cache; 16384 holds long traces.
        # Raise AEGIR_MAX_MODEL_LEN for even longer readouts (lower AEGIR_MAX_NUM_SEQS in step to hold KV).
        # Tool-calling MUST be enabled or agentic clients (vibe-acp etc.) that send tool_choice="auto" get a
        # vLLM 400 ("auto tool choice requires --enable-auto-tool-choice and --tool-call-parser"). Qwen3.6
        # emits the XML tool format → `qwen3_xml` (measured 2026-07-04: `hermes` silently EATS the calls —
        # the model plans the call in content, tool_calls=[]). Override via AEGIR_TOOL_PARSER.
        extra_args=["--max-model-len", os.environ.get("AEGIR_MAX_MODEL_LEN", "16384"),
                    "--max-num-seqs", os.environ.get("AEGIR_MAX_NUM_SEQS", "8"),
                    "--enable-auto-tool-choice",
                    "--tool-call-parser", os.environ.get("AEGIR_TOOL_PARSER", "qwen3_xml")],
    ),
}

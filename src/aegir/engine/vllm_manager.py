"""The engine's **sole** vLLM client. Launches/manages an OpenAI-compatible vLLM server per capability
on local GPUs (tensor-parallel), health-waits, and runs `Complete` by calling it INTERNALLY. The vLLM
endpoint is never exposed outside the engine — workloads reach inference only through the gRPC `Complete`.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from aegir.engine.config import (CAPABILITY_MODELS, DEFAULT_HF_HUB_CACHE, FIRST_GPU,
                                  VLLM_BASE_PORT, VLLM_PYTHON, ModelSpec)

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_LOG_DIR = Path(os.environ.get("AEGIR_ENGINE_LOG_DIR", "/tmp/aegir-engine"))


@dataclass
class Endpoint:
    capability: str
    spec: ModelSpec
    port: int
    gpu_ids: list[int]
    proc: subprocess.Popen | None = None
    healthy: bool = False
    log_path: Path | None = None


class VllmManager:
    """On-demand vLLM endpoints, one per capability. Thread-safe ensure()."""

    def __init__(self) -> None:
        self._ep: dict[str, Endpoint] = {}
        self._lock = threading.Lock()
        self._next_gpu = FIRST_GPU
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    def ensure(self, capability: str) -> Endpoint:
        if capability not in CAPABILITY_MODELS:
            raise KeyError(f"unknown capability {capability!r}; known: {list(CAPABILITY_MODELS)}")
        with self._lock:
            ep = self._ep.get(capability)
            if ep and ep.proc and ep.proc.poll() is None and ep.healthy:
                return ep
            if ep is None or (ep.proc and ep.proc.poll() is not None):
                ep = self._launch(capability)
                self._ep[capability] = ep
            self._wait_healthy(ep)
            return ep

    def _launch(self, capability: str) -> Endpoint:
        spec = CAPABILITY_MODELS[capability]
        tp = spec.tensor_parallel_size
        gpu_ids = list(range(self._next_gpu, self._next_gpu + tp))
        self._next_gpu += tp
        port = VLLM_BASE_PORT + len(self._ep)
        log_path = _LOG_DIR / f"vllm_{capability}_{port}.log"
        env = dict(os.environ)
        env["HF_HUB_CACHE"] = DEFAULT_HF_HUB_CACHE
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        # The vLLM venv is FOREIGN to aegir (cu129, built against the system glibc). Do NOT inherit
        # aegir's nix/devenv LD_LIBRARY_PATH: its gcc-15 libstdc++ requires GLIBC_2.38 the system libc
        # lacks, which breaks the foreign torch import. Use ONLY the libcuda unmask (proven sufficient
        # for CUDA init), and scrub Python cross-venv vars so the foreign interpreter uses its own site.
        env["LD_LIBRARY_PATH"] = f"{_REPO}/build/cuda-driver-libs"
        for _k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(_k, None)
        cmd = [
            VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.model, "--served-model-name", capability,
            "--port", str(port), "--host", "127.0.0.1",
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
            *spec.extra_args,
        ]
        fh = open(log_path, "w")
        # start_new_session=True puts vLLM (and the TP worker processes it spawns) in its own process
        # group, so shutdown() can kill the WHOLE tree — killing only the parent orphans the workers,
        # which keep holding GPU VRAM (observed: 4×~24GB leaked after a parent kill).
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        return Endpoint(capability=capability, spec=spec, port=port, gpu_ids=gpu_ids,
                        proc=proc, log_path=log_path)

    def _wait_healthy(self, ep: Endpoint, timeout: float = 900.0) -> None:
        url = f"http://127.0.0.1:{ep.port}/health"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if ep.proc and ep.proc.poll() is not None:
                raise RuntimeError(f"vLLM for {ep.capability!r} exited (rc={ep.proc.returncode}); "
                                   f"see {ep.log_path}\n{self._tail(ep.log_path)}")
            try:
                if httpx.get(url, timeout=2.0).status_code == 200:
                    ep.healthy = True
                    return
            except Exception:  # noqa: BLE001 — not up yet
                pass
            time.sleep(3.0)
        raise TimeoutError(f"vLLM for {ep.capability!r} not healthy in {timeout}s; see {ep.log_path}")

    @staticmethod
    def _tail(path: Path | None, n: int = 25) -> str:
        if not path or not path.exists():
            return ""
        return "\n".join(path.read_text(errors="ignore").splitlines()[-n:])

    def complete(self, capability: str, prompt: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.7,
                 json_schema: str = "") -> dict:
        ep = self.ensure(capability)
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": prompt})
        body = {"model": capability, "messages": msgs,
                "max_tokens": max_tokens, "temperature": temperature}
        if json_schema:
            # vLLM structured output enforced AT the engine (Atelier convergence proposal 2026-07-03).
            # NB the top-level guided_json extra-body form is SILENTLY IGNORED by vLLM 0.19.0 (Atelier
            # verified live: required fields came back missing) — the OpenAI-style response_format is
            # the enforcing path on this exact venv. Their correction, adopted before first use.
            import json as _json
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "constrained", "schema": _json.loads(json_schema),
                                "strict": True},
            }
        t0 = time.time()
        url = f"http://127.0.0.1:{ep.port}/v1/chat/completions"
        # RETAIN thinking: Qwen3.x reasons verbosely and the trace is a corpus value-add (Cerebras-style
        # reasoning retention), so we do NOT pass enable_thinking=False. Generous read timeout — long
        # traces are expected and acceptable (the workload waits). reasoning_content is populated when
        # the model/parser separates it; otherwise the trace is retained inline in content.
        r = httpx.post(url, json=body, timeout=900.0)
        # A caller may request prompt+max_tokens beyond the model window; vLLM 400s. CLAMP to fit and retry
        # once rather than surface an opaque INTERNAL — the error carries the exact counts (this bit
        # reauthor_unsat: max_tokens=16000 + a batched prompt > the 16384 window). Reasoning may still truncate
        # (finish_reason='length'), which the caller already handles; a hard context 400 must not be fatal.
        if r.status_code == 400 and "maximum context length" in r.text.lower():
            m = re.search(r"maximum context length is (\d+).*?\((\d+) in the messages", r.text, re.S)
            if m:
                ctx, msg_toks = int(m.group(1)), int(m.group(2))
                fit = max(256, ctx - msg_toks - 64)
                if fit < body["max_tokens"]:
                    body["max_tokens"] = fit
                    r = httpx.post(url, json=body, timeout=900.0)
        r.raise_for_status()
        d = r.json()
        choice = d["choices"][0]
        msg = choice["message"]
        usage = d.get("usage", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        # Qwen3.x wraps its trace in <think>…</think>; the OPENING tag is template-injected (absent from the
        # output) and the CLOSING tag is present. If no vLLM reasoning parser separated it, split on </think>
        # here so the trace is RETAINED SEPARATELY (Cerebras-style) instead of bleeding into the answer —
        # verified against the live output. Defers to vLLM's reasoning_content when a parser does populate it.
        if not reasoning and "</think>" in content:
            head, _, tail = content.partition("</think>")
            reasoning = head.replace("<think>", "").strip()
            content = tail.strip()
        return {"text": content, "model": ep.spec.model,
                "reasoning_content": reasoning,
                "finish_reason": choice.get("finish_reason") or "",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_ms": (time.time() - t0) * 1000.0}

    def status(self) -> list[Endpoint]:
        return list(self._ep.values())

    def shutdown(self) -> None:
        for ep in self._ep.values():
            if ep.proc and ep.proc.poll() is None:
                # Kill the whole process group (see start_new_session in _launch) so TP workers die too.
                try:
                    os.killpg(os.getpgid(ep.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    ep.proc.kill()

"""A ``dspy.LM``-shaped adapter over the Aegir capability/gRPC engine.

**Strict layering** (the engine is the SOLE vLLM client; workloads talk to the gRPC engine, never to
vLLM directly) means dspy-based workloads — notably ``scripts/generate_chapter.py`` — should reach the
model through the engine, not a raw OpenAI/vLLM endpoint. ``EngineLM`` is the bridge: it is duck-typed to
exactly the slice ``generate_chapter`` uses of a ``dspy.LM`` —

  * ``lm(messages=[{"role": "user", "content": ...}]) -> [{"text": ..., "reasoning_content": ...}]``
  * ``lm.history[-1]`` carrying ``{"usage": {...}, "cost": ...}`` for ``usage_and_cost``

— so it is a drop-in without pulling dspy in here. The **complete thinking trace** (``reasoning_content``,
separated by the engine's ``</think>`` split) is returned for retention via the existing pattern
(``response_reasoning`` column + ``raw.exchange``), NOT reduced to a token count. Genuine stats
(prompt/completion tokens, latency) ride in ``history``; cost is ``$0`` — the engine is local, so a run
never charges the budget cap.
"""
from __future__ import annotations


class EngineLM:
    """Minimal ``dspy.LM`` stand-in that completes via ``aegir.engine.client`` (gRPC → engine → vLLM)."""

    def __init__(self, capability: str = "instruct", *, max_tokens: int = 16000,
                 temperature: float = 0.4, timeout: float = 1800.0):
        self.capability = capability
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.model = f"engine/{capability}"
        self.kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        self.history: list[dict] = []

    def __call__(self, prompt: "str | None" = None, messages: "list | None" = None, **kw):
        from aegir.engine.client import complete_detailed
        system, user = "", prompt or ""
        for m in messages or []:
            role = m.get("role")
            if role == "system":
                system = m.get("content", "")
            elif role == "user":
                user = m.get("content", "")
        out = complete_detailed(
            user, capability=self.capability, system_prompt=system,
            max_tokens=int(kw.get("max_tokens", self.max_tokens)),
            temperature=float(kw.get("temperature", self.temperature)),
            timeout=self.timeout,
        )
        # litellm-shaped history record so generate_chapter.usage_and_cost reads tokens + (zero) cost.
        # reasoning_tokens is intentionally absent — the trace is retained as TEXT, not tallied.
        self.history.append({
            "model": self.model,
            "usage": {"prompt_tokens": int(out.get("prompt_tokens") or 0),
                      "completion_tokens": int(out.get("completion_tokens") or 0)},
            "cost": 0.0,                       # local engine — no API spend against the budget cap
            "finish_reason": out.get("finish_reason", ""),
            "latency_ms": out.get("latency_ms", 0.0),
        })
        # The complete thinking trace rides in reasoning_content for downstream retention.
        return [{"text": out.get("text", ""), "reasoning_content": out.get("reasoning_content", "")}]

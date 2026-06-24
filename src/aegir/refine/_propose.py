"""Fork-venv PROPOSE subprocess — the real ``vibe-acp`` agent proposing prose OR structured table edits.

Runs in the FORK's isolated venv (imports ``acp`` + drives ``vibe-acp``); invoked by ``loop.mint`` over a
subprocess boundary (the agent's env and the membrane's env are disjoint — the stdio protocol boundary IS the
env boundary, ``pipeline_run_gotchas`` #6). Reads ``{construct, mode, register}`` on stdin:
  * ``mode="prose"``  → returns ``{"prose": ...}`` in the requested register (natural / semantic) — dual-register.
  * ``mode="edits"``  → returns ``{"edits": [{op, table, column, concept}]}`` — the agent's SCAFFOLD agency over
    the tables (inc-2). The agent only PROPOSES edits; ``scaffold.apply_edits`` disposes them RI-safe.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile

FORK = os.environ.get("AEGIR_FORK_DIR", "/home/rch/local/src/zndx/aegir/components/oss-mistral-cli")


def _render_tables(construct: dict) -> str:
    lines = []
    for t in construct.get("tables", []):
        keys = {t.get("pk")} | {fk["col"] for fk in t.get("fks", [])}
        cols = ", ".join(f"{c['name']}{'*' if c['name'] in keys else ''} [{c.get('concept', '?')}]"
                         for c in t["columns"])
        lines.append(f"- table {t['name']} (columns, *=key: {cols})")
        for col in t["columns"]:
            vals = [str(c["value"]) for c in col["cells"][:4]]
            lines.append(f"    {col['name']}: {', '.join(vals)}")
    return "\n".join(lines)


def _prompt(construct: dict, mode: str, register: str, feedback: dict) -> str:
    if mode == "edits":
        return (
            "You are repairing the TABLES of a relational dataset chapter. Some NON-KEY columns hold "
            "placeholder or out-of-domain values. Propose structured edits to fix them: a `synth_column` edit "
            "re-synthesizes a non-key column from a domain concept. NEVER edit key columns (marked *). Output "
            "ONLY a JSON array, each item {\"op\":\"synth_column\",\"table\":...,\"column\":...,\"concept\":...} "
            "— no prose, no markdown fence.\n\n"
            f"Gate feedback: {json.dumps({k: feedback.get(k) for k in ('disjointness_violations', 'placeholder_rate')})}\n\n"
            "Tables:\n" + _render_tables(construct))
    style = ("a natural, topical practitioner voice (no ontology jargon — write as a data engineer describing "
             "a real dataset)" if register == "natural" else
             "a precise ontology-grounded technical voice (it is fine to reference the underlying concepts and "
             "how the schema materializes them)")
    return (f"Write 2-3 short paragraphs of accurate prose describing the tables below in {style}. Name each "
            "table, its columns, and representative values, and explain the foreign-key relationship. Use only "
            "the values shown. No preamble, no markdown headers — prose only.\n\nTables:\n"
            + _render_tables(construct))


def _parse_edits(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, list) else []
    except Exception:  # noqa: BLE001
        return []


def _strip_reasoning(text: str) -> str:
    return text.rsplit("</think>", 1)[1].strip() if "</think>" in text else text.strip()


def _agent_spec(home: str):
    """``(spec, config_toml | None, model_id, provider)`` for the proposer backend (AEGIR_PROPOSE_BACKEND):

    * ``local`` (default) → vibe-acp → the local vLLM engine.
    * ``grok`` → the authed ``grok`` CLI (Grok Build) as the ACP agent over stdio — the **UNMETERED
      subscription** (the Grok WS relay; one-time ``grok login``). No vibe config; the CLI carries the session.
    * ``xai-api`` → vibe-acp → the **METERED** xAI API (``XAI_API_KEY``).

    Grok / xai-api are OPT-IN — never the default (standing constraint). ``config_toml`` is None for grok
    (its agent isn't vibe-acp)."""
    from aegir.refine.acp import grok_acp_spec, vibe_acp_spec
    backend = os.environ.get("AEGIR_PROPOSE_BACKEND", "local").lower()
    if backend in ("grok", "grok-build"):
        return grok_acp_spec(home), None, os.environ.get("AEGIR_GROK_MODEL", "grok"), "grok-build"
    if backend in ("xai", "xai-api"):
        if not os.environ.get("XAI_API_KEY"):
            raise RuntimeError("AEGIR_PROPOSE_BACKEND=xai-api but XAI_API_KEY is unset (metered xAI API)")
        model = os.environ.get("AEGIR_XAI_MODEL", "grok-4.3")
        toml = ('active_model = "remote"\nsystem_prompt_id = "aegir_writer"\n'
                '[[providers]]\nname = "xai"\napi_base = "https://api.x.ai/v1"\n'
                'api_style = "openai"\napi_key_env_var = "XAI_API_KEY"\n'
                f'[[models]]\nname = "{model}"\nprovider = "xai"\nalias = "remote"\n')
        return vibe_acp_spec(FORK, home), toml, model, "xai-api"
    toml = ('active_model = "local"\nsystem_prompt_id = "aegir_writer"\n'
            '[[providers]]\nname = "local-vllm"\napi_base = "http://127.0.0.1:8100/v1"\n'
            'api_style = "openai"\n[[models]]\nname = "instruct"\nprovider = "local-vllm"\nalias = "local"\n')
    return vibe_acp_spec(FORK, home), toml, "instruct", "engine"


async def _run(construct: dict, mode: str, register: str, feedback: dict) -> dict:
    from aegir.refine.acp import BaseACPClient
    home = tempfile.mkdtemp(prefix="propose_home_")
    spec, cfg_toml, model_id, provider = _agent_spec(home)
    if cfg_toml:
        with open(os.path.join(home, "config.toml"), "w") as f:
            f.write(cfg_toml)
    prompt = _prompt(construct, mode, register, feedback)
    async with BaseACPClient(spec, fs_root=home) as c:
        r = await c.prompt(prompt, timeout=420)
    text = _strip_reasoning(r.text)
    out = {"edits": _parse_edits(text)} if mode == "edits" else {"prose": text}
    # the RAW exchange (full response incl. reasoning) for the aegir-side hx/OL lineage capture
    out["_exchange"] = {"prompt": prompt, "response": r.text, "reasoning": r.thoughts,
                        "model": model_id, "provider": provider}
    return out


def main() -> None:
    req = json.load(sys.stdin)
    out = asyncio.run(_run(req["construct"], req.get("mode", "prose"),
                           req.get("register", "natural"), req.get("feedback", {})))
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()

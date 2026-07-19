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


def _elaboration_prompt(construct: dict, feedback: dict) -> str:
    """The escalation-triage session (#32): the templates here already FAILED the single-shot loop —
    the value of this channel is the FULL membrane dialogue in one context. Every recorded reason is
    presented; the agent must address each explicitly."""
    dialogue = construct.get("dialogue") or []
    dlg = "\n".join(f"  round {i}: {r}" for i, r in enumerate(dialogue, 1)) or "  (none recorded)"
    return (
        "You rephrase a formal ontology axiom as sentences for a technical textbook on relational data "
        "modeling. This template was ESCALATED: previous attempts were rejected by deterministic gates "
        "for the reasons below. Read every reason; your rephrasings must avoid all of them.\n\n"
        f"AXIOM SENTENCE (the base):\n{construct.get('base', '')}\n\n"
        f"MANCHESTER SOURCE (authoritative for quantifiers and direction):\n{construct.get('manchester', '')}\n\n"
        f"REJECTION DIALOGUE:\n{dlg}\n\n"
        + ("Hard rules: (1) Preserve EVERY curly-brace placeholder exactly — identical spelling, "
           "identical braces, each exactly once. "
           if "{" in construct.get("base", "") else
           "Hard rules: (1) This base contains NO placeholders — your rephrasings must contain NO "
           "curly braces anywhere; write every class name as plain words exactly as the base does. ")
        + "(2) NO NEW CLAIMS: every statement must be entailed by the axiom; "
        "quantifiers and cardinalities unchanged. (3) DIRECTION: the sentence's subject is the class "
        "being defined — who bears/does/undergoes what must match the axiom exactly; open with the "
        "subject placeholder. (4) One sentence per rephrasing, 30–400 characters, distinct syntactic "
        f"styles. Produce {construct.get('n', 3)} rephrasings, each on its own line beginning exactly "
        "'REPHRASING: '; any thinking stays OUTSIDE those lines.")


def _parse_elaborations(text: str) -> list:
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*REPHRASING:\s*(.+?)\s*$", line)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _prompt(construct: dict, mode: str, register: str, feedback: dict) -> str:
    if mode == "elaborations":
        return _elaboration_prompt(construct, feedback)
    if mode == "edits":
        return (
            "You are repairing the TABLES of a relational dataset chapter. Some NON-KEY columns hold "
            "placeholder or out-of-domain values. Propose structured edits to fix them: a `synth_column` edit "
            "re-synthesizes a non-key column from a domain concept. NEVER edit key columns (marked *). Output "
            "ONLY a JSON array, each item {\"op\":\"synth_column\",\"table\":...,\"column\":...,\"concept\":...} "
            "— no prose, no markdown fence.\n\n"
            f"Gate feedback: {json.dumps({k: feedback.get(k) for k in ('disjointness_violations', 'placeholder_rate')})}\n\n"
            "Tables:\n" + _render_tables(construct))
    concepts = sorted({(col.get("concept") or "").replace("_", " ").strip()
                       for t in construct.get("tables", []) for col in t.get("columns", [])
                       if col.get("concept") and len(col.get("concept", "")) > 2})
    if register == "natural":
        frame = ("one section of a professional technical reference — a compliance handbook, governance "
                 "framework, or operational guide — in dense, evidence-anchored prose")
        teach = ("Teach the DOMAIN: explain what these things are, why they matter, and how they work in "
                 "practice, drawing on the data below as concrete examples woven into the prose.")
    else:
        frame = ("one section of an ontology-grounded technical reference explaining how a domain is modelled, "
                 "in precise conceptual prose")
        teach = ("Teach the CONCEPTS and how the schema materializes them: explain the entities, their "
                 "relationships, and how the records below instantiate them.")
    anchor = (construct.get("style_anchor") or "").strip()
    anchor_block = ("\n\nEcho the REGISTER and information density of this real corpus passage — its formality, "
                    "sentence rhythm, and how densely it packs evidence — as ONE style exemplar among many, while "
                    "writing about the data below. Do NOT adopt its subject matter or copy it:\n"
                    "«" + anchor[:1000] + "»") if anchor else ""
    return (
        f"You are writing {frame}. The subject is: {', '.join(concepts[:12]) or 'the data below'}.\n\n"
        f"{teach} The data is EVIDENCE, not the subject — do NOT open with 'The X table', do NOT enumerate "
        "columns one by one, do NOT narrate a schema. Cite specific values only where they illustrate a point. "
        "Write 4-6 substantial, flowing paragraphs. Begin with the subject matter — no preamble, no markdown "
        "headers." + anchor_block + "\n\nData to draw on:\n" + _render_tables(construct))


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
    if mode == "edits":
        out = {"edits": _parse_edits(text)}
    elif mode == "elaborations":
        out = {"elaborations": _parse_elaborations(text)}
    else:
        out = {"prose": text}
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

"""Fork-venv PROPOSE subprocess — the real ``vibe-acp`` agent rewriting a chapter's prose.

Runs in the FORK's isolated venv (it imports ``acp`` + drives ``vibe-acp``); invoked by ``loop.mint`` over a
subprocess boundary because the agent's env (acp/vibe) and the membrane's env (deeponto/HermiT) are disjoint
— the stdio protocol boundary IS the env boundary (see ``pipeline_run_gotchas`` #6). Reads a chapter construct
as JSON on stdin, returns ``{"prose": ...}`` on stdout. The agent only PROPOSES prose; the caller re-gates it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

FORK = os.environ.get("AEGIR_FORK_DIR", "/home/rch/local/src/zndx/aegir/components/oss-mistral-cli")


def _render_tables(construct: dict) -> str:
    lines = []
    for t in construct.get("tables", []):
        cols = ", ".join(c["name"] for c in t["columns"])
        lines.append(f"- table {t['name']} (columns: {cols})")
        for col in t["columns"]:
            vals = [str(c["value"]) for c in col["cells"][:4]]
            lines.append(f"    {col['name']}: {', '.join(vals)}")
        for fk in t.get("fks", []):
            lines.append(f"    FK: {fk['col']} -> {fk['ref_table']}.{fk['ref_col']}")
    return "\n".join(lines)


def _strip_reasoning(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


async def _run(construct: dict) -> str:
    from aegir.refine.acp import BaseACPClient, vibe_acp_spec
    home = tempfile.mkdtemp(prefix="vibe_home_")
    with open(os.path.join(home, "config.toml"), "w") as f:
        f.write('active_model = "local"\n'
                '[[providers]]\nname = "local-vllm"\napi_base = "http://127.0.0.1:8100/v1"\n'
                'api_style = "openai"\n'
                '[[models]]\nname = "instruct"\nprovider = "local-vllm"\nalias = "local"\n')
    prompt = (
        "You are writing one section of a technical chapter about a relational dataset. Write 2-3 short "
        "paragraphs of clear, accurate prose describing the tables below: name each table, its columns, and "
        "representative values, and explain the foreign-key relationship. Use only the values shown — invent "
        "nothing. No preamble, no markdown headers, no tables — prose only.\n\nTables:\n"
        + _render_tables(construct))
    async with BaseACPClient(vibe_acp_spec(FORK, home), fs_root=home) as c:
        r = await c.prompt(prompt, timeout=420)
    return _strip_reasoning(r.text)


def main() -> None:
    construct = json.load(sys.stdin)
    prose = asyncio.run(_run(construct))
    json.dump({"prose": prose}, sys.stdout)


if __name__ == "__main__":
    main()

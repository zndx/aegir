"""The Agent-Refined Stage — iterative refinement to satisfaction as the ACP step shape (#145).

RH (2026-07-05): *never* choke on overflows; context size must never be the hard limit on any
deliverable's length. The pattern that guarantees both:

- The deliverable is a WORKSPACE FILE the agent edits across turns (the client advertises fs
  capabilities, so vibe enables its read/write/edit builtins) — context bounds the working set
  per turn, never the artifact.
- Per-turn prompts stay lean: the brief + payload land as workspace files, not prompt-inline.
- Context overflow is a RECOVERABLE SIGNAL: the session is rebuilt and the work continues from
  workspace state (brief + progress + the artifact itself) — nothing is lost because nothing
  lived only in the conversation.
- HANDOFF is a FILE ACT: the agent writes ``HANDOFF.json`` when satisfied (robust to response
  truncation). Agent satisfaction is the PRECONDITION to handoff; the deterministic gates are
  the POSTCONDITION — a failing gate writes ``FEEDBACK.md``, clears the handoff, and the loop
  continues (the agent-mediated feedback doctrine).

``refine_to_satisfaction`` is the unit; Metaflow steps wrap it exactly like ``run_queue``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from aegir.refine.acp import BaseACPClient

# ACP/backend errors that mean "this SESSION is exhausted, the WORK is fine" — rebuild the
# session and continue from workspace state. Everything else propagates as a real failure.
_RECOVERABLE = ("context too long", "context_length", "maximum context", "context window")


@dataclass
class StageSpec:
    """One agent-refined deliverable."""
    workspace: Path                    # the stage's fs_root; brief/payload/artifact live here
    brief: str                         # written to brief.md — the deliverable contract
    deliverable: str = "deliverable.md"
    max_turns: int = 12                # bounded turns; the artifact size is NOT bounded
    max_gate_rounds: int = 3           # satisfaction→gate cycles before giving up loud
    turn_timeout: float = 900.0
    gates: "list[Callable[[Path], tuple[bool, str]]]" = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)


@dataclass
class StageResult:
    ok: bool
    turns: int = 0
    sessions: int = 0                  # >1 ⇒ overflow recovery happened and was survived
    gate_rounds: int = 0
    handoff: dict = field(default_factory=dict)
    gate_log: list = field(default_factory=list)
    error: str = ""

    @property
    def artifact_chars(self) -> int:
        return self._chars

    _chars: int = 0


_TURN_PROTOCOL = """
WORKING PROTOCOL (multi-turn; your context is small, the deliverable need not be):
- The deliverable is the FILE `{deliverable}` in this workspace. Build and revise it with your
  file tools — write/extend it INCREMENTALLY (a section per write); never paste the whole
  artifact into chat.
- `brief.md` holds the contract; `payload/` holds material to draw on (read pieces as needed).
- Keep `progress.md` current: 3-6 lines — what is done, what remains, what you'd do next.
  A future turn (possibly with fresh context) resumes from it.
- When — and only when — the deliverable fully satisfies brief.md and you would stake your
  name on it, write `HANDOFF.json`: {{"satisfied": true, "self_assessment": "<2-3 sentences>"}}.
- If `FEEDBACK.md` exists, it is a gate verdict on your last handoff: address it, revise the
  deliverable, then hand off again.
Do NOT write HANDOFF.json until the work is genuinely complete.
"""


def _continuation_prompt(spec: StageSpec) -> str:
    ws = spec.workspace
    progress = (ws / "progress.md").read_text() if (ws / "progress.md").exists() else "(none yet)"
    d = ws / spec.deliverable
    size = d.stat().st_size if d.exists() else 0
    feedback = ""
    if (ws / "FEEDBACK.md").exists():
        feedback = "\n\nGATE FEEDBACK on your last handoff (address before handing off again):\n" \
                   + (ws / "FEEDBACK.md").read_text()
    return (
        "Continue the deliverable in this workspace.\n"
        f"Current state: `{spec.deliverable}` is {size} chars. Your progress notes:\n{progress}"
        + feedback
        + _TURN_PROTOCOL.format(deliverable=spec.deliverable)
    )


def _initial_prompt(spec: StageSpec) -> str:
    return ("Produce the deliverable described in `brief.md` in this workspace.\n"
            + _TURN_PROTOCOL.format(deliverable=spec.deliverable))


async def refine_to_satisfaction(spec: StageSpec, *, agent_spec_factory) -> StageResult:
    """Drive one deliverable to agent-satisfied, gate-passed completion.

    ``agent_spec_factory() -> (AgentSpec, cfg_toml|None)`` builds a FRESH agent per session —
    called again after every overflow recovery, so a session is disposable by construction.
    """
    ws = spec.workspace
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "brief.md").write_text(spec.brief)
    handoff_p, feedback_p = ws / "HANDOFF.json", ws / "FEEDBACK.md"
    handoff_p.unlink(missing_ok=True)

    res = StageResult(ok=False)
    client: "BaseACPClient | None" = None

    async def _new_session() -> BaseACPClient:
        nonlocal client
        if client is not None:
            await client.close()
        agent, cfg_toml = agent_spec_factory()
        if cfg_toml:
            Path(agent.env["VIBE_HOME"], "config.toml").write_text(cfg_toml)
        c = BaseACPClient(agent, fs_root=ws, mcp_servers=spec.mcp_servers)
        await c.connect()
        res.sessions += 1
        client = c
        return c

    try:
        c = await _new_session()
        prompt = _initial_prompt(spec)
        while res.turns < spec.max_turns:
            res.turns += 1
            try:
                await c.prompt(prompt, timeout=spec.turn_timeout)
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if any(k in msg for k in _RECOVERABLE) or "timeout" in type(e).__name__.lower():
                    # OVERFLOW/EXHAUSTION → the session is spent, the WORK persists in the
                    # workspace. Rebuild and continue — the never-choke guarantee.
                    c = await _new_session()
                    prompt = _continuation_prompt(spec)
                    continue
                raise
            if handoff_p.exists():
                res.gate_rounds += 1
                try:
                    res.handoff = json.loads(handoff_p.read_text())
                except Exception:  # noqa: BLE001
                    res.handoff = {"satisfied": True, "raw": handoff_p.read_text()[:400]}
                verdicts = [(g(ws)) for g in spec.gates]
                res.gate_log.append([v for v in verdicts])
                failed = [reason for ok, reason in verdicts if not ok]
                if not failed:
                    res.ok = True
                    break
                if res.gate_rounds >= spec.max_gate_rounds:
                    res.error = f"gates still failing after {res.gate_rounds} handoffs: {failed[:2]}"
                    break
                feedback_p.write_text("\n\n".join(failed))
                handoff_p.unlink(missing_ok=True)
            prompt = _continuation_prompt(spec)
        else:
            res.error = f"no handoff within {spec.max_turns} turns"
    except Exception as e:  # noqa: BLE001
        res.error = str(e)[:400]
    finally:
        if client is not None:
            await client.close()
    d = ws / spec.deliverable
    res._chars = d.stat().st_size if d.exists() else 0
    return res

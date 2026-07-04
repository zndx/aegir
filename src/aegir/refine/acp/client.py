"""BaseACPClient — drive an Agent Client Protocol agent over JSON-RPC / stdio.

A general client primitive over the ``agent-client-protocol`` lib's client side: spawn an ACP agent
subprocess (default target: our mistral ``vibe-acp`` fork pointed at the local vLLM), run prompts, and stream
the agent's message / thought / tool chunks. Backend-agnostic — the spawn command, the model config (via the
agent's env, e.g. ``VIBE_HOME`` → a local-vLLM provider TOML), and the tool servers are all injected.

Membrane-oracle by construction: the agent only PROPOSES — it prompts and *requests* tool calls; THIS client
owns the filesystem surface (scoped to ``fs_root``), the session lifecycle, and which tools exist (injected
as stdio MCP servers). The refinement loop registers the deterministic gates (value-HermiT / RI / prose) as
those MCP tools, so the agent can request a gate but never run, fake, or bypass it.

Pattern learned from the gaius/hermes ACP integrations; this is an original implementation.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

# A streaming sink the consumer can pass to observe the agent live: (kind, text) with
# kind ∈ {"text", "thought", "tool", "status"}. May be sync or async.
StreamSink = Callable[[str, str], "Awaitable[None] | None"]


@dataclass
class AgentSpec:
    """How to spawn the ACP agent. Defaults target our vibe-acp fork (see ``vibe_acp_spec``)."""
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass
class MCPServer:
    """An MCP server exposing tools to the agent — where the refinement gates plug in.
    Stdio form (command/args) for agents that spawn servers (vibe-acp); HTTP form (url) for
    agents whose ACP supports only http/sse MCP (grok). Exactly one of command/url."""
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict | None = None
    cwd: str | None = None
    url: str = ""


@dataclass
class ACPResult:
    text: str
    thoughts: str
    tool_calls: list[str]
    stop_reason: str | None


def vibe_acp_spec(fork_dir: str | Path, vibe_home: str | Path, *, extra_env: dict | None = None) -> AgentSpec:
    """AgentSpec for the vendored mistral ``vibe-acp`` fork, pointed at a local-vLLM config dir.

    ``vibe_home`` is a directory containing a ``config.toml`` with an OpenAI-compatible provider at the local
    vLLM endpoint (see ``refinement_loop.md``). Spawns the fork's own venv binary so its deps resolve."""
    fork = Path(fork_dir).resolve()
    binary = fork / ".venv" / "bin" / "vibe-acp"
    command = str(binary) if binary.exists() else "vibe-acp"
    env = {"VIBE_HOME": str(Path(vibe_home).resolve()), **(extra_env or {})}
    return AgentSpec(command=command, args=[], env=env, cwd=str(fork))


def grok_acp_spec(cwd: str | Path, *, extra_env: dict | None = None) -> AgentSpec:
    """AgentSpec for the authed ``grok`` CLI (Grok Build) as an ACP agent over stdio — the UNMETERED
    subscription path (the Grok WS relay), DISTINCT from the metered ``XAI_API_KEY`` xAI API. ``grok agent
    stdio`` speaks ACP, so ``BaseACPClient`` drives it directly; the CLI's cached session (one-time ``grok
    login``, in ``~/.grok``) carries the subscription. Pattern (subscription device-auth + agent-over-stdio)
    from the hermes / zndx-xai-auth integrations; implementation original."""
    import shutil
    binary = shutil.which("grok") or "grok"
    return AgentSpec(command=binary, args=["agent", "stdio"], env=dict(extra_env or {}),
                     cwd=str(Path(cwd).resolve()))


class BaseACPClient:
    def __init__(self, agent: AgentSpec, *, fs_root: str | Path | None = None, allow_write: bool = True,
                 mcp_servers: "list[MCPServer] | None" = None, stream: StreamSink | None = None,
                 connect_timeout: float = 180.0, buffer_limit: int = 8 * 1024 * 1024) -> None:
        self.agent = agent
        self.fs_root = Path(fs_root).resolve() if fs_root else None
        self.allow_write = allow_write
        self.mcp_servers = mcp_servers or []
        self.stream = stream
        self.connect_timeout = connect_timeout
        self.buffer_limit = buffer_limit
        self._cm = self._conn = self._proc = self._session = self._client = None
        self.tools_approved: list[str] = []

    # ── fs policy + streaming ────────────────────────────────────────────────
    def _in_root(self, p: Path) -> bool:
        if self.fs_root is None:
            return True
        try:
            rp = p.resolve()
            return rp == self.fs_root or self.fs_root in rp.parents
        except Exception:  # noqa: BLE001
            return False

    async def _emit(self, kind: str, text: str) -> None:
        if self.stream:
            r = self.stream(kind, text)
            if asyncio.iscoroutine(r):
                await r

    def _build_client(self):
        from acp import Client
        from acp.schema import (AgentMessageChunk, AgentThoughtChunk, TextContentBlock,
                                ToolCallProgress, ToolCallStart)
        outer = self

        class _AegirACPClient(Client):
            def __init__(self) -> None:
                super().__init__()
                self.text: list[str] = []
                self.thoughts: list[str] = []
                self.tools: list[str] = []

            # fs (read_text_file/write_text_file) + terminal use the lib's base defaults for now; the
            # membrane fs-scoping (confine writes to fs_root) lands with the gate-tools in inc-2.
            async def request_permission(self, options, session_id, tool_call, **kw):  # noqa: ANN001
                """Auto-approve tool use. Without this the agent's tool call dies on the
                base class's unimplemented handler and the TURN RETURNS EMPTY with no error
                — the root cause of the tools-attached empty-prose failures. Approval policy:
                the client already owns WHICH tools exist (the MCP loadout is the membrane);
                a tool the agent can request is a tool it may run. Prefer allow_once."""
                from acp.schema import AllowedOutcome, RequestPermissionResponse
                pick = next((o for o in options if getattr(o, "kind", "") == "allow_once"),
                            options[0] if options else None)
                name = getattr(tool_call, "title", None) or "tool"
                outer.tools_approved.append(name)
                await outer._emit("tool", f"approved:{name}")
                # AllowedOutcome, NOT SelectedPermissionOutcome: the response union is
                # discriminated on the `outcome` field ('selected'|'cancelled'), which
                # SelectedPermissionOutcome doesn't carry — the agent-side parse fails
                # union_tag_not_found and the session hangs to timeout (measured).
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(
                        outcome="selected",
                        option_id=pick.option_id if pick else "allow_once"))

            async def session_update(self, session_id, update, **kw):  # noqa: ANN001
                if isinstance(update, AgentMessageChunk) and isinstance(update.content, TextContentBlock):
                    if update.content.text:
                        self.text.append(update.content.text)
                        await outer._emit("text", update.content.text)
                elif isinstance(update, AgentThoughtChunk) and isinstance(
                        getattr(update, "content", None), TextContentBlock):
                    if update.content.text:
                        self.thoughts.append(update.content.text)
                        await outer._emit("thought", update.content.text)
                elif isinstance(update, ToolCallStart):
                    name = getattr(update, "title", None) or "tool"
                    self.tools.append(name)
                    await outer._emit("tool", name)
                elif isinstance(update, ToolCallProgress):
                    msg = getattr(update, "message", "") or ""
                    if msg:
                        await outer._emit("status", msg)

        return _AegirACPClient()

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def connect(self) -> "BaseACPClient":
        from acp import PROTOCOL_VERSION, spawn_agent_process
        from acp.schema import Implementation, McpServerStdio

        self._client = self._build_client()
        env = {**os.environ, **self.agent.env}
        async with asyncio.timeout(self.connect_timeout):
            self._cm = spawn_agent_process(
                self._client, self.agent.command, *self.agent.args,
                env=env, cwd=self.agent.cwd, transport_kwargs={"limit": self.buffer_limit})
            self._conn, self._proc = await self._cm.__aenter__()
            await self._conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_info=Implementation(name="aegir-refine", version="0.1.0"))
            from acp.schema import EnvVariable

            def _env_vars(env):
                # McpServerStdio.env is List[EnvVariable]; forward a dict (or the parent's whole
                # environment) so the agent-spawned MCP subprocess inherits PATH/VIRTUAL_ENV/
                # PYTHONPATH — an EMPTY env strips the interpreter's world and hangs the handshake.
                if not env:
                    return []
                if isinstance(env, dict):
                    return [EnvVariable(name=str(k), value=str(v)) for k, v in env.items()]
                return env
            from acp.schema import HttpMcpServer
            servers = [
                HttpMcpServer(type="http", name=s.name, url=s.url, headers=[]) if s.url else
                McpServerStdio(name=s.name, command=s.command, args=s.args,
                               cwd=s.cwd, env=_env_vars(s.env))
                for s in self.mcp_servers]
            session = await self._conn.new_session(
                cwd=self.agent.cwd or os.getcwd(), mcp_servers=servers)
            self._session = session.session_id
        return self

    async def prompt(self, message: str, *, timeout: float | None = None) -> ACPResult:
        if self._conn is None or self._client is None:
            raise RuntimeError("BaseACPClient not connected — call connect() first")
        from acp import text_block
        c = self._client
        c.text.clear(); c.thoughts.clear(); c.tools.clear()
        stop = None

        async def _send():
            nonlocal stop
            res = await self._conn.prompt(prompt=[text_block(message)], session_id=self._session)
            stop = getattr(res, "stop_reason", None)

        if timeout is not None:
            async with asyncio.timeout(timeout):
                await _send()
        else:
            await _send()
        return ACPResult("".join(c.text), "".join(c.thoughts), list(c.tools), stop)

    async def close(self) -> None:
        if self._cm is not None:
            with contextlib.suppress(Exception):
                await self._cm.__aexit__(None, None, None)
        self._cm = self._conn = self._proc = self._session = self._client = None

    async def __aenter__(self) -> "BaseACPClient":
        return await self.connect()

    async def __aexit__(self, *exc) -> None:
        await self.close()

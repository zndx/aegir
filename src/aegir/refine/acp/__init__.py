"""Agent Client Protocol (ACP) client for the refinement loop.

``BaseACPClient`` drives an ACP agent (e.g. our mistral ``vibe-acp`` fork pointed at the local vLLM) over
JSON-RPC/stdio. Membrane-oracle by construction: the agent only PROPOSES (prompts + tool requests); this
client controls the filesystem surface, the session, and tool execution (via injected MCP servers), so the
refinement loop gates everything the agent does. Pattern learned from the gaius/hermes ACP integrations;
implementation original.
"""
from aegir.refine.acp.client import ACPResult, AgentSpec, BaseACPClient, MCPServer, vibe_acp_spec

__all__ = ["BaseACPClient", "AgentSpec", "ACPResult", "MCPServer", "vibe_acp_spec"]

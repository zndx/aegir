"""ACPMintEffector — the meta-harness `mint` effector: a real reasoning agent.

An ACP *client* that spawns `grok agent stdio` over stdio (the `acp` lib) and, each
FSM iteration, asks it to draft/enrich/refine ONE OWL-Manchester construct for a gap
topic — conditioned on the contract + the topic's source text + its salient terms +
the current gate feedback. The agent reasons; the spine gates. Reasoning is never
simulated (RH: intelligent-by-default). Modeled on GaiusACPClient
(zndx/gaius/src/gaius/acp/client.py); uses the in-venv `acp` lib directly (no
cross-repo dep). The agent returns a fenced ```json construct in its response — we
parse it (no agent file I/O: reads/writes/permissions are denied).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402
import generate_ontology as go  # noqa: E402

from acp import Client, PROTOCOL_VERSION, spawn_agent_process, text_block  # noqa: E402
from acp.schema import Implementation, AgentMessageChunk, AgentThoughtChunk, TextContentBlock  # noqa: E402

log = logging.getLogger("acp_mint")


class _MintClient(Client):
    """Buffers the agent's message text. `acp.Client` is a Protocol whose other
    methods (read/write file, request_permission, terminals) default to no-ops
    returning None — a soft-deny. inc-1 needs none of them: the agent reasons and
    returns a fenced JSON construct in its message; no environment access."""

    def __init__(self) -> None:
        self.buf: list[str] = []
        self.thoughts: list[str] = []

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        if isinstance(update, AgentMessageChunk) and isinstance(update.content, TextContentBlock):
            self.buf.append(update.content.text)
        elif isinstance(update, AgentThoughtChunk) and isinstance(getattr(update, "content", None), TextContentBlock):
            self.thoughts.append(update.content.text)

    def take(self) -> str:
        r = "".join(self.buf)
        self.buf.clear()
        self.thoughts.clear()
        return r


def build_mint_prompt(topic: dict, objective: str, feedback: dict, prev: dict | None,
                      exemplars: list[CatalogTemplate]) -> str:
    src = (topic.get("topic_repr_text") or "")[:2500]
    top_terms = topic.get("_top_terms") or []
    ex = "\n".join(
        f"- {t.template_id}: {t.manchester_template}  (anchor {t.bfo_anchor_path[-1:] })"
        for t in exemplars[:2])
    contract = (
        "CONTRACT (your construct must pass ALL — this is a conjunctive membrane):\n"
        "  • DeepOnto verbalizes it AND it asserts a COMPLEX class (a restriction/intersection — "
        "not a bare `X SubClassOf: anchor`).\n"
        "  • LOGICALLY COHERENT (HermiT, sound+complete): no class unsatisfiable — do NOT anchor across "
        "DISJOINT BFO categories (e.g. a Process is an Occurrent and CANNOT also be a Continuant/Artifact/ICE).\n"
        "  • It lowers to a valid relational table (≥3 typed columns; Trino+Spark-parseable DDL).\n"
        "  • NOVEL (not a near-duplicate of the seed catalog).\n"
        "  • R1 (the binding one): its DOMAIN TERMS must match THIS topic's salient terms. "
        "R1 scores your minted class/property/slot names against the topic vocabulary below. "
        "Generic names (Article, Document, Person, hasName) score ~0 and FAIL. Mint "
        "TOPIC-SPECIFIC concepts (e.g. for herbal medicine: MedicinalPlant, hasActiveCompound, "
        "AntiInflammatoryEffect).\n"
        f"  TOPIC SALIENT TERMS (match these): {', '.join(sorted(top_terms)[:25])}\n"
    )
    if objective in ("draft_initial",) or not prev:
        task = ("Author ONE new Manchester axiom template capturing the SPECIFIC domain concepts of "
                "the SOURCE TEXT below, grounded in BFO/CCO, with topic-specific class/property names.")
    else:
        pv = json.dumps({k: v for k, v in (prev.get("template_dict") or {}).items()
                         if k in ("template_id", "manchester_template", "slot_types", "bfo_anchor_path")})
        fails = ", ".join(f"{k}={feedback.get(k)}" for k in
                          ("deeponto_ok", "deeponto_complex", "consistent", "polyglot_ok", "novelty_ok",
                           "schema_ok", "r1_on", "r1_ci_low") if k in feedback)
        obj_hint = {
            "enrich_domain_terms": "Your terms are too generic — R1 is low. REWRITE with topic-specific "
                                   "minted class/property names drawn from the salient terms.",
            "fix_verbalizability": "DeepOnto could not verbalize it — fix the Manchester syntax/IRIs.",
            "fix_nontriviality": "It is a bare subsumption — add a restriction (e.g. `{p:ObjectProperty} some {Y:Class}`).",
            "fix_consistency": "HermiT found it INCOHERENT (a named class is unsatisfiable — usually anchored "
                               "across DISJOINT BFO categories, e.g. Process ⊓ Continuant). Re-anchor to ONE category.",
            "fix_ddl": "Its DDL does not parse — simplify to a clean typed table shape.",
            "diversify": "It duplicates a seed — make it materially different.",
            "de_can": "Too few/uniform columns — add typed DataProperty slots (values/dates/counts).",
            "refine": "Refine it to fully satisfy the contract.",
        }.get(objective, "Refine it to satisfy the contract.")
        task = (f"Your PREVIOUS construct was:\n{pv}\nGate feedback: {fails}\n"
                f"OBJECTIVE [{objective}]: {obj_hint}\nReturn an improved single construct.")
    return (
        "You are an ontology engineer extending a BFO 2020 / CCO grounded ontology.\n\n"
        f"{go.SLOT_DSL}\n\n{contract}\n"
        f"Family exemplars (imitate shape + grounding):\n{ex}\n\n"
        f"{task}\n\n"
        f'SOURCE TEXT (real document from the target topic):\n"""{src}"""\n\n'
        "Output ONLY a fenced ```json block with a JSON list containing ONE object: "
        '{"template_id": str (lowercase_snake, descriptive of the DOMAIN), "manchester_template": str, '
        '"slot_types": {slot: "Class|ObjectProperty|DataProperty|Individual"}, "bfo_anchor_path": [iri]}. '
        f"bfo_anchor_path must end at one of {sorted(go.ANCHORS)}."
    )


class ACPMintEffector:
    def __init__(self, agent_cmd: tuple[str, list[str]] = ("grok", ["agent", "stdio"]),
                 prompt_timeout: float = 300.0):
        self.cmd, self.args = agent_cmd
        self.prompt_timeout = prompt_timeout
        self.exemplars: dict[str, list[CatalogTemplate]] = {}
        import glob
        for f in sorted(glob.glob(str(REPO / "src/aegir/ontology/catalog/0[1-7]_*.json"))):
            if "candidate" in f or "combined" in f:
                continue
            self.exemplars[Path(f).stem] = load_catalog(f).templates
        self.loop = asyncio.new_event_loop()
        self.client = _MintClient()
        self._cm = None
        self.conn = None
        self.session: str | None = None
        self.loop.run_until_complete(self._connect())

    async def _connect(self) -> None:
        import shutil
        cmd = shutil.which(self.cmd) or self.cmd
        if not os.environ.get("XAI_API_KEY") and not (Path.home() / ".grok/auth.json").exists():
            raise RuntimeError("grok has no creds: set XAI_API_KEY or `grok login`.")
        # 16MB stdio buffer (Gaius's default) — agent session/update lines can exceed
        # asyncio's 64KB readline limit → LimitOverrunError / receive-loop death.
        self._cm = spawn_agent_process(self.client, cmd, *self.args,
                                       env=dict(os.environ), cwd=str(REPO),
                                       transport_kwargs={"limit": 16 * 1024 * 1024})
        self.conn, self.proc = await self._cm.__aenter__()
        await self.conn.initialize(protocol_version=PROTOCOL_VERSION,
                                   client_info=Implementation(name="aegir-mediate", version="0.1.0"))
        s = await self.conn.new_session(cwd=str(REPO), mcp_servers=[])
        self.session = s.session_id
        log.info("ACP connected: %s %s  session=%s", cmd, self.args, self.session)

    def mint(self, topic: dict, objective: str, feedback: dict, prev: dict | None) -> dict:
        fam = topic.get("top_family") or "07_long_tail"
        prompt = build_mint_prompt(topic, objective, feedback, prev, self.exemplars.get(fam, []))
        self.client.take()  # clear buffer
        try:
            self.loop.run_until_complete(asyncio.wait_for(
                self.conn.prompt(prompt=[text_block(prompt)], session_id=self.session),
                timeout=self.prompt_timeout))
        except Exception as e:
            log.warning("mint prompt failed (%s): %s", objective, type(e).__name__)
            return prev or {"template": None, "template_id": None, "error": str(e)[:120]}
        resp = self.client.take()
        templates = go.parse_templates(resp)
        if not templates:
            log.warning("mint produced no parseable construct (objective=%s, resp=%dchars)", objective, len(resp))
            return prev or {"template": None, "template_id": None, "raw_len": len(resp)}
        t = templates[0]
        t.provenance = {"generated_from_topic": str(topic.get("topic_id")), "family": fam,
                        "model": "grok-acp", "objective": objective}
        return {"template": t, "template_id": t.template_id,
                "template_dict": asdict(t), "raw_len": len(resp)}

    def close(self) -> None:
        try:
            if self._cm is not None:
                self.loop.run_until_complete(self._cm.__aexit__(None, None, None))
        finally:
            self.loop.close()

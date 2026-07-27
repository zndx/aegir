"""Greenfield derive harness — objective-oriented, EMD-driven entity extraction (#141).

The agentic mini-harness that replaces the one-shot template deriver (RH 2026-07-04). It
prompts the engine to read a domain-adapted FinePDFs passage and extract an ENTITY-CENTRIC
ontology fragment — real named classes with typed DataProperties, cardinality-bounded object
relations, and enumerations — the exact structure that lowers (via kvasir) into wide,
junction/lookup-bearing DDL, driving the Shape EMD toward SchemaPile parity.

The prompt targets the three measured DDL deficits explicitly (attr_zero 0.83 → DataProperties,
0 junctions → min/max>1 relations, 0 lookups → enumerations). Structured output is enforced by
the engine's json_schema (:data:`aegir.ontology.entities.ENTITY_SCHEMA`), so the agent returns
validated entity records, not free text to regex-parse.
"""
from __future__ import annotations

import json

from aegir.ontology.entities import Entity, ENTITY_SCHEMA, from_json

_SYSTEM = """\
You are a domain data architect building a real relational schema, expressed as a BFO 2020 /
CCO-anchored OWL ontology, from a source passage. Model the ENTITIES the passage actually
describes as a working database would — richly, the way a real arXiv preprint's data section
or a production LIMS schema reads. NOT abstract taxonomy.

CRITICAL — model the SUBJECT MATTER, never the document. If the passage is a spatial-planning
paper, model land parcels, zoning districts, transit corridors, temperature readings, policy
instruments — the things the RESEARCH studies and records. Do NOT model the publication itself:
NO AcademicArticle / Journal / Author / Researcher / Citation / DOI / Institution classes. The
bibliographic wrapper is noise; the domain phenomena the text is ABOUT are the schema.

For each entity (a real named class):
- give a CamelCase `name` and a plain-English `label`;
- anchor it to a `genus` — a real BFO/CCO IRI. Ask what KIND of thing it is and pick from that
  branch. Getting the branch wrong makes the class logically INCONSISTENT, not merely imprecise:
  a role filed as a thing, or an act filed as an artifact, contradicts BFO's continuant/occurrent
  split and the class becomes unsatisfiable.
    · a PERSON or people — cco:ont00001262 (Person), cco:ont00000914 (Group of Persons)
    · an ORGANIZATION — cco:ont00001180 (Organization), cco:ont00000443 (Commercial Organization)
    · a ROLE something BEARS — cco:ont00000175 (Organization Member Role), cco:ont00000187
      (Authority Role), cco:ont00000984 (Occupation Role), bfo:0000023 (Role, when none fits).
      Employee, Member, Owner, Donor, Authority, Participant name ROLES — never things.
    · an ACT or PROCESS unfolding in time — bfo:0000015 (Process). Anything named as an activity,
      request, assessment, enrolment, transfer, or approval belongs here.
    · INFORMATION — a record, dataset, plan, curriculum, standard, specification, measurement
      result, requirement: cco:ont00000958 (Information Content Entity), cco:ont00002039
      (Document Content Entity), cco:ont00000853 (Descriptive Information). A record is
      INFORMATION; it is not a physical object.
    · a MADE PHYSICAL THING — cco:ont00000995 (Material Artifact), bfo:0000040 (Material Entity).
      Only for things with mass that you could point at. This is NOT the fallback.
    · a PLACE — cco:ont00000472 (Geospatial Region)
    · a CAPABILITY or disposition — cco:ont00000568 (Organization Capability)
  Prefer the most specific class you are confident in. If two branches seem possible, decide what
  the entity IS — a person, a role, an act, information, a physical object, a place — and take
  that branch. Never guess at Material Artifact to avoid choosing;
- write a one-sentence `definition`;
- give it MANY `attributes` (typed DataProperties) — 4 to 10 per entity, the real measured and
  recorded fields: identifiers, dates, quantities, names, statuses, codes. Use xsd types
  string/integer/decimal/dateTime/date/boolean. Where a field is a closed set of states, give
  its `enum` values (e.g. status: pending/running/complete/failed) — these become lookup tables;
- give it `relations` to OTHER entities you also define — FKs and many-to-many links. Each has
  a `prop` (verb, e.g. storedIn, measuredBy, derivedFrom), a `target` entity name, and a `card`:
  "exactly 1" or "some" for a to-one FK, "max 1" for an optional FK, "min 2" or "min 3" for a
  genuine many-to-many (these become junction tables).

Define 4-8 interrelated entities that share foreign keys, so the schema is a connected graph,
not islands. Prefer real domain vocabulary over generic names. Return ONLY the JSON object."""


def deriver_version() -> str:
    """The derive-stage staleness key: entities cached on disk are REUSED only when stamped
    with the current version (system prompt + output schema). A prompt change re-derives —
    idempotence keyed on (passage_hash, deriver_version)."""
    import hashlib
    from aegir.ontology.entities import ENTITY_SCHEMA
    return hashlib.sha256((_SYSTEM + json.dumps(ENTITY_SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]


def propose(passage: str, *, capability: str = "instruct", temperature: float = 0.4,
            max_tokens: int = 6000, context: str = "") -> "tuple[list[Entity], dict]":
    """One engine call → entity records from a passage. Returns ``(entities, meta)`` where
    ``meta`` carries the raw output + reasoning for tracing. json_schema-enforced."""
    from aegir.engine.client import complete_detailed

    prompt = passage.strip()
    if context:
        prompt = context.strip() + "\n\n---\n\n" + prompt
    out = complete_detailed(
        prompt, capability=capability, system_prompt=_SYSTEM,
        temperature=temperature, max_tokens=max_tokens,
        json_schema=json.dumps(ENTITY_SCHEMA),
    )
    text = out.get("text", "") if isinstance(out, dict) else str(out)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # tolerate a fenced block if the engine wrapped it — and NEVER let one malformed
        # output kill a multi-hundred-passage stage (FMEA: max_tokens truncation mid-JSON
        # took down the 431-passage run at passage 209). Unparseable → zero entities →
        # the metrology loop records verdict=malformed for THIS passage and continues.
        import re
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {"entities": []}
        except json.JSONDecodeError:
            obj = {"entities": []}
    entities = from_json(obj)
    # Carry the FULL exchange provenance, not a summary of it. `complete_detailed` returns model,
    # token counts and latency; this meta used to drop all of them alongside the reasoning, so the
    # boundary that lost the trace also lost WHICH MODEL produced the entities. provider/model are
    # first-class provenance (the doctrine refine/lineage.py already states): each data element should
    # record the agent and model that produced it. The caller writes the Iceberg exchange record.
    meta = {"n_entities": len(entities),
            "reasoning": out.get("reasoning_content", "") if isinstance(out, dict) else "",
            "raw_len": len(text),
            "prompt": prompt,
            "system_prompt": _SYSTEM,
            "response_text": text,
            "capability": capability,
            "model": (out.get("model") or "") if isinstance(out, dict) else "",
            "prompt_tokens": (out.get("prompt_tokens") or 0) if isinstance(out, dict) else 0,
            "completion_tokens": (out.get("completion_tokens") or 0) if isinstance(out, dict) else 0,
            "latency_ms": (out.get("latency_ms") or 0) if isinstance(out, dict) else 0,
            "finish_reason": (out.get("finish_reason") or "") if isinstance(out, dict) else ""}
    return entities, meta

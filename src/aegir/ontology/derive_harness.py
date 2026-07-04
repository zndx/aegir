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

For each entity (a real named class):
- give a CamelCase `name` and a plain-English `label`;
- anchor it to a `genus` — a real BFO/CCO IRI: cco:Artifact (a made thing / record / sample),
  bfo:0000015 (a process / activity / measurement), cco:InformationContentEntity (a document
  / dataset / designator), bfo:0000023 (a role);
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


def propose(passage: str, *, capability: str = "instruct", temperature: float = 0.4,
            max_tokens: int = 4000, context: str = "") -> "tuple[list[Entity], dict]":
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
        # tolerate a fenced block if the engine wrapped it
        import re
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {"entities": []}
    entities = from_json(obj)
    meta = {"n_entities": len(entities),
            "reasoning": out.get("reasoning_content", "") if isinstance(out, dict) else "",
            "raw_len": len(text)}
    return entities, meta

#!/usr/bin/env python
"""Emit a TTL block declaring all sdg:* properties newly introduced
in Batch 7. Properties are enumerated from the candidate JSON; for
each one we synthesise a plausible label + skos:definition based on
its naming pattern.

This is mechanical scaffolding — it satisfies the charter's
"every term used in templates is declared" requirement so the
catalog stays internally closed. Definitions can be hand-edited
later if a property's intent surfaces a sharper paraphrase.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CANDIDATE = Path("src/aegir/ontology/catalog/07_long_tail.json")
EXISTING_TTL = Path("src/aegir/ontology/sdg-vocab.ttl")
OUT = Path("/tmp/b7_ttl_block.txt")


def split_camel(name: str) -> list[str]:
    parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", name)
    return [p.lower() for p in parts]


def humanize(name: str) -> str:
    return " ".join(split_camel(name))


# Map a property's leading verb / shape to a definition template.
DEFINITION_TEMPLATES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^has[A-Z]"),         "Relates an entity to its associated {} value or designator."),
    (re.compile(r"^is[A-Z]"),          "Relates an entity to a structural relationship in which it serves as the {}."),
    (re.compile(r"^at[A-Z]"),          "Relates an entity to the contextual {} at which it is located, observed, or evaluated."),
    (re.compile(r"^by[A-Z]"),          "Relates an entity to the {} responsible for or attached to it."),
    (re.compile(r"^for[A-Z]"),         "Relates an entity to the {} it pertains to or is intended for."),
    (re.compile(r"^within[A-Z]"),      "Relates an entity to the {} within which it is contained."),
    (re.compile(r"^matches[A-Z]"),     "Relates an entity to the {} it is identified with under entity-matching."),
    (re.compile(r"^aligned[A-Z]"),     "Relates an entity to the {} alignment under cross-vocabulary mapping."),
    (re.compile(r"^attaches[A-Z]"),    "Relates an entity to the {} to which it attaches at runtime."),
    (re.compile(r"^attests[A-Z]"),     "Relates a directive or attestation to the {} it certifies."),
    (re.compile(r"^annotates[A-Z]"),   "Relates an annotation to the {} it labels."),
    (re.compile(r"^anomalyIn[A-Z]"),   "Relates an anomaly description to the {} in which it manifests."),
    (re.compile(r"^baseline[A-Z]"),    "Relates a baseline description to the {} it serves as a reference for."),
    (re.compile(r"^belief[A-Z]"),      "Relates a Dempster-Shafer belief value to the {} it scores."),
    (re.compile(r"^breaking[A-Z]"),    "Relates a breaking change to the {} it impacts."),
    (re.compile(r"^called[A-Z]"),      "Relates an artifact to the {} that calls or invokes it."),
    (re.compile(r"^coarsens[A-Z]"),    "Relates a frame coarsening to the {} it coarsens to."),
    (re.compile(r"^combines[A-Z]"),    "Relates a combined description to the {} via its combination operator."),
    (re.compile(r"^conjunctively[A-Z]"), "Relates a conjunctive combination to the {}."),
    (re.compile(r"^disjunctively[A-Z]"), "Relates a disjunctive combination to the {}."),
    (re.compile(r"^describes[A-Z]"),   "Relates a description to the {} it characterizes."),
    (re.compile(r"^dropped[A-Z]"),     "Relates a column to the {} at which it was dropped."),
    (re.compile(r"^emits[A-Z]"),       "Relates a process to the {} it produces as observable output."),
    (re.compile(r"^entityType[A-Z]"),  "Relates an entity-type tag to the {} it is grouped under."),
    (re.compile(r"^evidence[A-Z]"),    "Relates a piece of evidence to the {} relevant to its evidential strength."),
    (re.compile(r"^foreignKey[A-Z]"),  "Relates a foreign-key column to the {} it references."),
    (re.compile(r"^forward[A-Z]"),     "Relates an artifact version to the {} it remains forward-compatible with."),
    (re.compile(r"^backward[A-Z]"),    "Relates an artifact version to the {} it remains backward-compatible with."),
    (re.compile(r"^gdpr[A-Z]"),        "Relates a directive to the {} from the GDPR text under which it is grounded."),
    (re.compile(r"^grants[A-Z]"),      "Relates a directive to the {} it confers."),
    (re.compile(r"^hipaa[A-Z]"),       "Relates a directive to the {} from the HIPAA framework it implements."),
    (re.compile(r"^iso\d"),            "Relates a directive to the {} from the corresponding ISO/IEC standard text."),
    (re.compile(r"^migrates[A-Z]"),    "Relates a schema-migration process to the {} it produces."),
    (re.compile(r"^nist\d"),           "Relates a directive to the {} from NIST 800-53."),
    (re.compile(r"^nonBreaking[A-Z]"), "Relates a non-breaking change to the {} it modifies."),
    (re.compile(r"^pci[A-Z]"),         "Relates a directive to the {} from PCI DSS."),
    (re.compile(r"^plausibility[A-Z]"), "Relates a Dempster-Shafer plausibility value to the {} it scores."),
    (re.compile(r"^refines[A-Z]"),     "Relates a frame refinement to the {} it refines."),
    (re.compile(r"^renamed[A-Z]"),     "Relates a column-version to the {} it was renamed from."),
    (re.compile(r"^reviews[A-Z]"),     "Relates a review-content entity to the {} it reviews."),
    (re.compile(r"^rolls[A-Z]"),       "Relates a rollback record to the {} it rolls back to."),
    (re.compile(r"^schema[A-Z]"),      "Relates a schema artifact to the {} it stands in lineage with."),
    (re.compile(r"^signs[A-Z]"),       "Relates a signing agent to the {} it signs off on."),
    (re.compile(r"^snapshot[A-Z]"),    "Relates a dataset snapshot to the {} it captures."),
    (re.compile(r"^soc[A-Z]"),         "Relates a directive to the {} from the SOC2 trust-services criteria."),
    (re.compile(r"^sourced[A-Z]"),     "Relates a derived record to the {} it was sourced from."),
    (re.compile(r"^triggered[A-Z]"),   "Relates an alert event to the {} that triggered it."),
    (re.compile(r"^type[A-Z]"),        "Relates a column-version to the {} from which its type changed."),
    (re.compile(r"^under[A-Z]"),       "Relates an entity to the {} under which it falls."),
    (re.compile(r"^with[A-Z]"),        "Relates an entity to the {} it carries as context."),
    (re.compile(r"^yager[A-Z]"),       "Relates a Yager combination to the {} via the Yager rule of combination."),
]


def synth_definition(prop_local: str) -> str:
    parts = split_camel(prop_local)
    # phrase = words after the first verb segment
    phrase = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
    for pat, fmt in DEFINITION_TEMPLATES:
        if pat.search(prop_local):
            return fmt.format(phrase)
    # fall-back: use the whole humanized form
    return f"Relates an entity to the associated {humanize(prop_local)}."


def main() -> None:
    data = json.loads(CANDIDATE.read_text())
    new_props: set[str] = set()
    for t in data["templates"]:
        for m in re.finditer(r"sdg:[a-zA-Z][\w]*", t["manchester_template"]):
            new_props.add(m.group(0))

    existing = set()
    for line in EXISTING_TTL.read_text().splitlines():
        m = re.match(r"^(sdg:[a-zA-Z][\w]*)\s+a", line)
        if m:
            existing.add(m.group(1))

    new = sorted(new_props - existing)
    lines = [
        "## ======================================================================",
        "## Batch 7 — long tail / benchmark gap-filling.",
        "##",
        "## Properties enumerated below are the procedurally-generated",
        "## tail covering SOTAB schema.org alignment, CTA/CPA annotation,",
        "## compliance frame depth (NIST/ISO/SOC2/GDPR/HIPAA/PCI), DST",
        "## combination operators, schema evolution, telemetry, eBPF",
        "## helpers, and cross-branch density. Definitions are synthesised",
        "## from name patterns and are intentionally generic — sharpen by",
        "## hand on a per-property basis if a particular term carries",
        "## non-obvious semantics that the synthesised gloss misses.",
        "## ======================================================================",
        "",
    ]
    for p in new:
        local = p.split(":", 1)[1]
        defn = synth_definition(local)
        # escape double quotes for TTL string-literal safety
        defn_escaped = defn.replace('"', '\\"')
        lines.append(f"{p} a owl:ObjectProperty ;")
        lines.append(f"    rdfs:label \"{humanize(local)}\" ;")
        lines.append(f"    skos:definition \"{defn_escaped}\" .")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"emitted {len(new)} new properties to {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

"""Catalog schema for the procedural ontology pipeline.

The catalog *C* is the central data artifact of the ontology / RLVR
research thread (see ``docs/current/src/ontology/concept_brief.md``).
Each row is an OWL Manchester-syntax template with typed slots,
plus DeepOnto-derived metadata cached at catalog-construction time.
At runtime the verifier and the policy consume *C* by lookup; no
JVM, no DeepOnto, no Java dependency is involved after construction.

This module defines the schema only. The catalog itself is authored
in ``catalog/*.json`` and populated by an offline DeepOnto pass when
phase P1a runs. Example rows used to validate the schema and to
exercise the slot DSL live at ``catalog/examples.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CatalogTemplate:
    """A typed-slot OWL Manchester-syntax template.

    Attributes:
        template_id: Unique handle for the template.
        manchester_template: OWL Manchester syntax with typed slots,
            e.g. ``"Class: {X:Class} SubClassOf: {p:ObjectProperty} some {Y:Class}"``.
            See ``SLOT_DSL.md`` for slot grammar.
        slot_types: Map from slot name to OWL type
            (e.g. ``{"X": "Class", "p": "ObjectProperty", "Y": "Class"}``).
        is_complex: Whether DeepOnto's ``get_asserted_complex_classes()``
            classifies this template as a complex axiom. Determined
            offline at P1a; set to ``False`` until then for placeholder
            rows.
        verbal_template: DeepOnto ``OntologyVerbaliser`` output with
            slot variables retained for runtime substitution. Empty
            string for placeholder rows.
        mean_verbal_length: Character length of the verbalization
            after slot substitution with representative fillers.
            Used by the verifier as the ``R_C`` semantic-richness
            proxy. ``0.0`` for placeholder rows.
        bfo_anchor_path: Ordered list of IRIs from this template's
            target class up to a BFO 2020 upper class.
        broader: Parent term ids (``rdfs:subClassOf`` / ``skos:broader`` semantics) —
            the subsumption hierarchy over terms, promoted from ``mediate_hierarchy``'s
            autonomous, HermiT-verified mediation (the SoT record of the hierarchy).
        provenance: Author / date / gate-version metadata for review.
    """

    template_id: str
    manchester_template: str
    slot_types: dict[str, str]
    is_complex: bool = False
    verbal_template: str = ""
    mean_verbal_length: float = 0.0
    bfo_anchor_path: list[str] = field(default_factory=list)
    broader: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass
class Catalog:
    """The procedural catalog — a flat table of typed-slot templates.

    Provides lookup helpers consumed by the runtime verifier
    (``aegir-verify``) and by the RL policy when composing
    ontologies. The catalog is read-only at runtime; mutations
    happen offline at P1a and are committed as new ``catalog/*.json``
    versions.

    Attributes:
        version: Semver string for the catalog. Major version bumps
            indicate non-backward-compatible schema changes.
        templates: All templates in the catalog.
        null_stats: Pre-computed null-distribution statistics
            (``tau_b``, ``r_d_normalization``) cached at
            catalog-construction time so the runtime verifier doesn't
            recompute them. Empty for placeholder catalogs.
    """

    version: str
    templates: list[CatalogTemplate]
    null_stats: dict[str, float] = field(default_factory=dict)

    def by_id(self, template_id: str) -> CatalogTemplate:
        """Look up a template by id. Raises ``KeyError`` if absent."""
        for t in self.templates:
            if t.template_id == template_id:
                return t
        raise KeyError(f"template_id {template_id!r} not in catalog")

    def complex_only(self) -> list[CatalogTemplate]:
        """Return only the templates flagged ``is_complex=True``."""
        return [t for t in self.templates if t.is_complex]

    def by_axiom_kind(self, kind: str) -> list[CatalogTemplate]:
        """Return templates whose Manchester template contains the
        given axiom keyword (e.g. ``"some"``, ``"only"``,
        ``"EquivalentTo"``, ``"min"``, ``"max"``, ``"exactly"``).

        Whitespace-bounded substring match; not a parser.
        """
        needle = f" {kind} "
        return [
            t for t in self.templates
            if needle in f" {t.manchester_template} "
        ]


def load_catalog(path: str | Path) -> Catalog:
    """Load a catalog from JSON.

    The JSON must have top-level keys ``version``, ``templates``,
    optionally ``null_stats``. Each row in ``templates`` is decoded
    into a :class:`CatalogTemplate`.
    """
    data = json.loads(Path(path).read_text())
    templates = [CatalogTemplate(**row) for row in data["templates"]]
    return Catalog(
        version=data["version"],
        templates=templates,
        null_stats=data.get("null_stats", {}),
    )


def save_catalog(catalog: Catalog, path: str | Path) -> None:
    """Write a catalog to JSON, round-trippable with :func:`load_catalog`."""
    payload = {
        "version": catalog.version,
        "templates": [asdict(t) for t in catalog.templates],
        "null_stats": catalog.null_stats,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")

"""vocab_lowering — the scheme→lookup lowering's aegir half (#44 increment (c)).

AUTHORITY ORDER (RH 2026-07-25): the entailment direction OWL ⊨ SKOS ⊨ SHACL IS
the authority direction — **the realized OMN is the source of logical truth**
(differentia, subsumption, restrictions, HermiT-checkable consistency); SKOS is
the controlled-vocabulary CONTENT model only (labels, membership, scheme
structure — the Primer's scope: thesauri/taxonomies/controlled vocabularies,
with no logical semantics). Collections annotated ``sdg:lowersToProperty``
supply CONTENT (Collection = table, member = row); this generator carries that
content INTO the realized OMN — the one artifact HermiT certifies and kvasir
consumes — as ``DataProperty`` frames whose ``skos:definition`` carries the
canonical parenthesized value list, activating kvasir's designed-dormant
definition-enum path → ``@Enum`` → lookup table + FK wherever an elected class
uses the property.

One critical path: edit the Collection, the realized OMN regenerates the enum,
kvasir regenerates the lookup — and any LOGICAL claims about the values (status
distinctness, transition constraints) author in OMN, never in SKOS. The join is
the EXPLICIT annotation, never name derivation. Values are the members'
prefLabels, sorted for determinism (lookup rows are a set; lifecycle ordering
is presentation).
"""
from __future__ import annotations


def vocab_enums_omn() -> str:
    """OMN DataProperty frames for every lowersToProperty-annotated Collection."""
    try:
        import rdflib
    except Exception:  # noqa: BLE001 — no rdflib, no vocab enums (never a crash)
        return ""
    from aegir.ontology.domain_index import integration_overlay_files

    SK = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    SDG = rdflib.Namespace("https://signals.zndx.org/sdg#")
    g = rdflib.Graph()
    for f in integration_overlay_files():
        try:
            g.parse(str(f))
        except Exception as e:  # noqa: BLE001 — an unreadable overlay is VISIBLE
            print(f"vocab_lowering: overlay unreadable: {f.name}: {e}")
    frames: "list[str]" = []
    for coll in sorted(set(g.subjects(SDG.lowersToProperty, None)), key=str):
        prop = str(next(g.objects(coll, SDG.lowersToProperty), "")).strip()
        if not prop.startswith("sdg:"):
            print(f"vocab_lowering: {coll} lowersToProperty must name an sdg: property "
                  f"(got {prop!r}) — skipped VISIBLY")
            continue
        labels = sorted(str(next(g.objects(m, SK.prefLabel), ""))
                        for m in g.objects(coll, SK.member))
        labels = [v for v in labels if v]
        if len(labels) < 2:
            print(f"vocab_lowering: {coll} has <2 labeled members — skipped VISIBLY")
            continue
        cname = str(next(g.objects(coll, SK.prefLabel), "")) or str(coll).rsplit("#", 1)[-1]
        frames.append(
            f"DataProperty: {prop}\n"
            f'    Annotations: rdfs:label "{prop.split(":", 1)[-1]}",\n'
            f'        skos:definition "{cname} value; one of ({", ".join(labels)}). '
            f"Generated from the {str(coll).rsplit('#', 1)[-1]} skos:Collection — the SKOS "
            f'source of this lookup (scheme-to-lookup lowering, one source of truth)."\n')
    return ("\n" + "\n".join(frames)) if frames else ""

def vocab_provenance_sidecar(out_path=None) -> dict:
    """Per-value provenance for the lut columns (#44⇄#45 convergence): keyed by the
    FULL property IRI → {value → {prov, iri}}, drawn from the members' uniform
    ``sdg:valueProvenance`` channel. kvasir ``ddl --vocab-provenance`` widens the
    reference tables so every seeded row carries its authority IN THE DATABASE
    (census | gittables lineage | AUTHORED flag), continuously validatable
    against the corpus indexes."""
    import json
    from pathlib import Path as _P
    try:
        import rdflib
    except Exception:  # noqa: BLE001
        return {}
    from aegir.ontology.domain_index import integration_overlay_files

    SK = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    SDG = rdflib.Namespace("https://signals.zndx.org/sdg#")
    g = rdflib.Graph()
    for f in integration_overlay_files():
        try:
            g.parse(str(f))
        except Exception:  # noqa: BLE001
            pass
    out: dict = {}
    for coll in g.subjects(SDG.lowersToProperty, None):
        prop = str(next(g.objects(coll, SDG.lowersToProperty), "")).strip()
        if not prop.startswith("sdg:"):
            continue
        prop_iri = "https://signals.zndx.org/sdg#" + prop.split(":", 1)[-1]
        vals = {}
        for m in g.objects(coll, SK.member):
            label = str(next(g.objects(m, SK.prefLabel), ""))
            prov = str(next(g.objects(m, SDG.valueProvenance), ""))
            if label:
                vals[label] = {"prov": prov, "iri": str(m)}
        if vals:
            out[prop_iri] = vals
    if out_path:
        _P(out_path).write_text(json.dumps(out, indent=1))
    return out

def element_projections_sidecar(out_path=None) -> dict:
    """The both-shapes projection map (RH 2026-07-25): normalized domain tables
    stay authoritative; the element face is a MATERIALIZED VIEW derived from
    them (RI by construction). Keys = full class IRIs, values = census-confirmed
    SysMLv2 metaclasses (lut codes — the correspondence to metaclass_lut holds
    by construction since both derive from the census)."""
    import json
    from pathlib import Path as _P
    from aegir.ontology.manufacturing_module import ELEMENT_PROJECTIONS
    out = {f"https://signals.zndx.org/sdg#{k}": v for k, v in ELEMENT_PROJECTIONS.items()}
    if out_path:
        _P(out_path).write_text(json.dumps(out, indent=1))
    return out

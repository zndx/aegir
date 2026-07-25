"""vocab_lowering — the scheme→lookup lowering's aegir half (#44 increment (c)).

Collections annotated ``sdg:lowersToProperty`` in the integration overlays are
the SKOS SOURCES of relational reference tables (Collection = table, member =
row). This generator serializes each into an OMN ``DataProperty`` frame whose
``skos:definition`` carries the canonical parenthesized value list — activating
kvasir's designed-dormant definition-enum path (manchester/lower.rs: "the
deriver carrying skos:definition into the realized omn activates it") →
``@Enum`` → lookup table + FK wherever an elected class uses the property.

ONE SOURCE OF TRUTH, one critical path: edit the Collection in the TTL, the
realized OMN regenerates the enum, kvasir regenerates the lookup. The join is
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

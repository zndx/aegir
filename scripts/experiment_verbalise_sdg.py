#!/usr/bin/env python
"""Experimental: load sdg-vocab.ttl and run DeepOnto's OntologyVerbaliser.

One-off probe. Goal: see what the verbaliser actually produces on our
real ontology, by eye. No metrics, no caching, no catalog updates —
just print verbalizations.

Invoke via::

    LD_PRELOAD=/nix/store/pa6n8nrmgq8jswk2pkrl5qprcls1r0ch-expat-2.7.5/lib/libexpat.so.1 \\
        uv run --no-sync python scripts/experiment_verbalise_sdg.py

(LD_PRELOAD forces the libexpat that pyexpat.so was compiled against;
without it deeponto's import chain hits an undefined-symbol error.)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.deeponto_harness import ensure_jvm

ensure_jvm()

from deeponto.onto import Ontology, OntologyVerbaliser  # noqa: E402


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ontology",
        default=str(REPO / "build" / "experiments" / "audit_risk_mini.ttl"),
        help="Path to TTL/OWL file to load (default: hand-crafted "
             "audit_risk_mini.ttl). Override with sdg-vocab.ttl etc.",
    )
    args = p.parse_args()
    ttl_path = Path(args.ontology)
    print(f"# loading: {ttl_path}")
    onto = Ontology(str(ttl_path))
    verbaliser = OntologyVerbaliser(onto)

    print(f"# owl_classes:           {len(onto.owl_classes)}")
    print(f"# owl_object_properties: {len(onto.owl_object_properties)}")
    print(f"# owl_data_properties:   {len(onto.owl_data_properties)}")
    print()

    # --- the non-triviality check ---
    # A valid, non-trivial ontology has classes defined with complex class
    # expressions (intersection, union, existential/universal restriction,
    # cardinality) — not just simple named-class subsumption. The DeepOnto
    # criterion: get_asserted_complex_classes() returns a non-empty set.
    print("=" * 78)
    print("NON-TRIVIALITY CHECK: get_asserted_complex_classes()")
    print("=" * 78)
    complex_concepts = list(onto.get_asserted_complex_classes())
    print(f"# count: {len(complex_concepts)}")
    print()
    for i, cls in enumerate(complex_concepts[:30]):
        try:
            v = verbaliser.verbalise_class_expression(cls)
            print(f"  [{i:3d}] {v.verbal}")
        except Exception as e:
            print(f"  [{i:3d}] {cls}  (verbalise error: {type(e).__name__}: {e})")
    if not complex_concepts:
        print("  (empty — ontology is structurally trivial by this measure)")
    print()

    # --- subsumption axioms (SubClassOf) ---
    print("=" * 78)
    print("SUBSUMPTION AXIOMS (SubClassOf)")
    print("=" * 78)
    sub_axioms = onto.get_subsumption_axioms("Classes")
    print(f"# count: {len(sub_axioms)}")
    print()
    for i, ax in enumerate(sub_axioms[:30]):
        try:
            result = verbaliser.verbalise_class_subsumption_axiom(ax)
            # result is a tuple (sub_verbal, super_verbal); print as
            # "<sub> ⊑ <super>"
            if isinstance(result, tuple) and len(result) == 2:
                sub_v, super_v = result
                print(f"  [{i:3d}] {sub_v.verbal}  ⊑  {super_v.verbal}")
            else:
                print(f"  [{i:3d}] (raw result: {result!r})")
        except Exception as e:
            print(f"  [{i:3d}] ERROR: {type(e).__name__}: {e}")
    print()

    # --- equivalence axioms (EquivalentTo) ---
    print("=" * 78)
    print("EQUIVALENCE AXIOMS (EquivalentTo)")
    print("=" * 78)
    eq_axioms = onto.get_equivalence_axioms("Classes")
    print(f"# count: {len(eq_axioms)}")
    print()
    for i, ax in enumerate(eq_axioms[:20]):
        try:
            result = verbaliser.verbalise_class_equivalence_axiom(ax)
            if isinstance(result, tuple) and len(result) == 2:
                lhs, rhs = result
                print(f"  [{i:3d}] {lhs.verbal}  ≡  {rhs.verbal}")
            else:
                print(f"  [{i:3d}] (raw result: {result!r})")
        except Exception as e:
            print(f"  [{i:3d}] ERROR: {type(e).__name__}: {e}")
    print()

    # --- object-property domain axioms ---
    # sdg-vocab.ttl is a property vocab (314 ObjectProperties), so the
    # natural verbalisation targets here are domain/range axioms.
    print("=" * 78)
    print("OBJECT-PROPERTY DOMAIN AXIOMS")
    print("=" * 78)
    # The OWL-API method is per-property (takes an OWLObjectPropertyExpression),
    # so we walk our 314 properties and collect their domain axioms.
    domain_axioms = []
    for _op_iri, op in list(onto.owl_object_properties.items()):
        for ax in onto.owl_onto.getObjectPropertyDomainAxioms(op):
            domain_axioms.append(ax)
            if len(domain_axioms) >= 30:
                break
        if len(domain_axioms) >= 30:
            break
    print(f"# count (capped at 30): {len(domain_axioms)}")
    print()
    for i, ax in enumerate(domain_axioms):
        try:
            result = verbaliser.verbalise_object_property_domain_axiom(ax)
            if isinstance(result, tuple) and len(result) == 2:
                prop_v, dom_v = result
                print(f"  [{i:3d}] domain of «{prop_v.verbal}»  =  {dom_v.verbal}")
            else:
                print(f"  [{i:3d}] (raw: {result!r})")
        except Exception as e:
            print(f"  [{i:3d}] ERROR: {type(e).__name__}: {e}")
    print()

    # --- object-property range axioms ---
    print("=" * 78)
    print("OBJECT-PROPERTY RANGE AXIOMS")
    print("=" * 78)
    range_axioms = []
    for _op_iri, op in list(onto.owl_object_properties.items()):
        for ax in onto.owl_onto.getObjectPropertyRangeAxioms(op):
            range_axioms.append(ax)
            if len(range_axioms) >= 30:
                break
        if len(range_axioms) >= 30:
            break
    print(f"# count (capped at 30): {len(range_axioms)}")
    print()
    for i, ax in enumerate(range_axioms):
        try:
            result = verbaliser.verbalise_object_property_range_axiom(ax)
            if isinstance(result, tuple) and len(result) == 2:
                prop_v, rng_v = result
                print(f"  [{i:3d}] range of «{prop_v.verbal}»  =  {rng_v.verbal}")
            else:
                print(f"  [{i:3d}] (raw: {result!r})")
        except Exception as e:
            print(f"  [{i:3d}] ERROR: {type(e).__name__}: {e}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

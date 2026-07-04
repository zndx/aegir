#!/usr/bin/env python
"""Convert any OWL/TTL ontology → OWL Manchester syntax (.omn) via OWLAPI.

The user-side pre-step of the kvasir-ddl Manchester-only ingestion ruling (map §7):
kvasir ingests Manchester; format conversion is the ecosystem's standard step (ROBOT /
Protégé / DeepOnto). This is the DeepOnto path — it reuses the same OWLAPI the pipeline
already runs, so the capability gate can convert a foreign ontology (CCO, IOF, an OBO
member) with no new dependency.

Run with the JVM lib bootstrap per CLAUDE.md:
  LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/convert_to_manchester.py <in> <out.omn>
"""
from __future__ import annotations

import sys
from pathlib import Path


def convert(src: str, dst: str) -> int:
    # ensure_jvm sets JVM_MEMORY before any deeponto import — dodges the click.prompt
    # "maximum memory" hang documented in CLAUDE.md / deeponto_harness.
    from aegir.ontology.deeponto_harness import ensure_jvm

    ensure_jvm()
    import jpype

    # OWLAPI directly (not DeepOnto's Ontology wrapper, which requires an ontology IRI
    # a module-extraction TTL may lack).
    OWLManager = jpype.JClass("org.semanticweb.owlapi.apibinding.OWLManager")
    IRI = jpype.JClass("org.semanticweb.owlapi.model.IRI")
    File = jpype.JClass("java.io.File")
    manch = jpype.JClass(
        "org.semanticweb.owlapi.formats.ManchesterSyntaxDocumentFormat")()

    mgr = OWLManager.createOWLOntologyManager()
    onto = mgr.loadOntologyFromOntologyDocument(File(src))
    mgr.saveOntology(onto, manch, IRI.create(File(dst).toURI()))
    n_cls = int(onto.getClassesInSignature().size())
    n_op = int(onto.getObjectPropertiesInSignature().size())
    n_ax = int(onto.getAxiomCount())
    print(f"{src} → {dst}: {n_cls} classes, {n_op} object properties, {n_ax} axioms")
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: convert_to_manchester.py <in.owl|in.ttl> <out.omn>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    if not Path(src).exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    rc = convert(src, dst)
    from aegir.utils.clean_exit import clean_exit  # os._exit — dodge the jpype teardown crash
    clean_exit(rc)
    return rc  # unreachable (clean_exit calls os._exit) — satisfies the type checker


if __name__ == "__main__":
    main()

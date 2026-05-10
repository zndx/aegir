# Provenance — reviewer's guide

This document is for code reviewers of any PR that touches the
ontology — `sdg-vocab.ttl`, `catalog/*.json`, or any
`aegir.ontology.*` Python module. It distills the editorial
discipline from
[`docs/current/ontology/charter.md#provenance-discipline`](../../docs/current/ontology/charter.md)
into a checklist that a reviewer can run through in five minutes.

The discipline is editorial — there is no script that decides
"novel-vs-derived." That judgment lives in code review, the same
way every other authorship judgment in the project does.

## Checklist for ontology PRs

For each new or modified term:

- [ ] **Public-namespace IRIs** (`schema:`, `dbo:`, `bfo:`, `cco:`,
      `skos:`, `prov:`, etc.) — accept on the namespace's authority.
      No further review needed beyond verifying the IRI resolves.
- [ ] **`sdg:`-namespace terms** — review the four following
      points before accepting:

  1. **`rdfs:subClassOf` chain** anchors at a public-namespace
     ancestor (BFO 2020 upper class via CCO when appropriate).
  2. **`rdfs:label`** is a clear, project-authored phrase; not
     copy-pasted from any external dataset.
  3. **`skos:definition`** describes what the term denotes and
     why it exists in the project's own conceptual frame.
  4. **The term is not a thin proxy** for a public-namespace
     class — if `sdg:Foo` is just `subClassOf schema:Bar` with
     no further constraints or definition, declare `schema:Bar`
     directly and skip the bespoke proxy.

- [ ] **Catalog template** edits — ensure each template still
      satisfies the slot-DSL grammar
      ([`SLOT_DSL.md`](./SLOT_DSL.md)) and that the slot type
      declarations match the IRIs used in the Manchester string.

## Authorship judgment

The review question is:

> Does this term read as the project's own engineering and
> conceptual work, or as material lifted from elsewhere?

If the latter, raise it in review the same way you'd raise any
copy-paste concern. Specific signs to watch for:

- Labels or definitions that read as adapted rather than authored.
- Cluster patterns (e.g., introducing several terms whose only
  shared property is appearing together in some external source).
- Acronyms or column-naming conventions characteristic of a
  particular dataset rather than ontology terminology.
- Unusual specificity that can't be defended on project-domain
  grounds.

When in doubt, ask the contributor where the term came from.
The Charter's principle is positive admission — terms must
demonstrate they belong, rather than the gate maintaining a
record of what to exclude. A clean answer to "where did this
come from?" is part of demonstrating belonging.

## What this guide is not

- It is **not** a script. The mechanical CI checks
  ([Charter §Mechanical checks](../../docs/current/ontology/charter.md#mechanical-checks))
  cover structural integrity (TTL parses, label + definition
  present, BFO ancestry for `sdg:` terms, label-map JSON
  consistency, SPARQL totality). They do not assess provenance.
- It is **not** a deny-list. The reviewer's instinct is the
  filter, calibrated by familiarity with the project domain and
  with what the project's bespoke vocabulary is supposed to mean.
- It is **not** a license review. Public-namespace IRIs come from
  ontologies with their own licenses; consult those when packaging
  for distribution, not at term-by-term review time.

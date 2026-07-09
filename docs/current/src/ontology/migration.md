# Migration

Plan for relocating the bespoke vocabulary and the synthetic data
generators into Ægir without breaking the consumer projects that use
them today. Designed to be reversible up to the cutover, and to fail
loudly rather than drift silently.

> **Status (2026-06-29) — historical; partially executed, then
> superseded.** This is the *original* migration plan as written on
> 2026-05-09. Phase 0 and the authoring step of Phase 1 happened: the
> renamed canonical vocabulary now lives in-tree at
> `src/aegir/ontology/sdg-vocab.ttl` and the mechanical TTL checks run
> in CI (`just check-ontology-schema`). The rest of the plan did **not**
> execute as written and is retained here only as a record of intent:
> - The named cutover artifact, `vocab_label_map.json` v1.0.0, was
>   never produced, and there is no `aegir-vocab.ttl` in the tree. The
>   project's vocabulary work moved past a benchmark-label-map
>   deliverable to a **content-derived, HermiT-validated realized
>   ontology** — `corpora/ontology/sdg-ontology.{omn,owl}` with a
>   consistency certificate (`HERMIT_CERTIFICATE.md`) — authored as
>   catalog templates and gated by the OQuaRE publish gate. The current
>   reference for that artifact and its gates is the
>   [Authors Guide](./authors_guide.md).
> - The `_LABEL_DIMS["sotab"]` reconciliation in Phase 2 is **done**:
>   `src/aegir/data/table_dataset.py` carries `"sotab": 82`.
> - The synth-migration phases (4–6) and the SOTAB-CTA empirical gate
>   (Phase 3) were not run; current evidence (`EVIDENCE.md`) and project
>   framing treat SOTAB-CTA as likely the wrong eval and gate the
>   ontology Data Product on OQuaRE instead.
>
> The phased plan below is preserved unchanged for provenance. Read it
> as history, not as the plan of record. **Structural recommendation
> (flagged, not performed): archive this page, or fold it into a short
> "vocabulary provenance" note, once a maintainer confirms nothing
> downstream still links to the phased cutover.**
>
> **Addendum (2026-07-09).** The hand-authored seed family catalogs
> themselves have since been retired (`4d200e4`): the live catalog is
> `src/aegir/ontology/catalog/catalog.json`, fully derived and grown by
> the derive→promote loop. The present-state landing is
> [Ontology](../ontology.md); the canonical gate reference is the
> [Authors Guide](./authors_guide.md).

## Starting state (2026-05-09)

- **`atelier-vocab.ttl`** lives in a sibling project, ~1052 lines.
  Operational — used by the sibling's classification pipeline today.
- **Synthetic generators** (`synth_generators.py` + `synth_registry.py`
  + `synth.py`) live in the same sibling, 316+ hand-coded generators
  plus a three-tier priority dispatcher. Used by the sibling's BDD and
  pytest test infrastructure.
- **Ægir** has no `src/aegir/ontology/` directory and no `.ttl` files
  anywhere in the tree. Greenfield destination.
- The sibling project has published an ownership-migration
  note confirming it will become a consumer of trained Ægir
  artifacts and surrender ownership of the vocabulary. That note is
  treated here as input, not specification — Ægir designs the move on
  its own terms.

## Provenance discipline

Ægir's vocabulary admits terms only on positive public-source
grounding — see
[Charter §Provenance discipline](./charter.md#provenance-discipline)
for the principle and admission rules. The migration boundary below
is where those rules first apply.

## Cutover criterion (named, not implicit)

> **Cutover is when Ægir publishes `vocab_label_map.json` v1.0.0 and the
> sibling project commits a release that consumes it.** From that
> commit onward, the sibling's local TTL snapshot is frozen and any
> vocabulary change must land in Ægir.

Until that point, both copies are operational; the sibling continues
its own development against its own TTL. After that point, the
sibling's TTL is documentation of its starting state, not a live
artifact.

This avoids the failure mode where two TTLs evolve in parallel for
weeks under the assumption the migration is "in progress" — the kind
of two-source-of-truth state that doesn't fail loudly and is expensive
to reconcile after the fact.

## Phased plan

### Phase 0 — scaffolding (this commit)

- Create `src/aegir/ontology/` and `src/aegir/synth/` as empty Python
  packages with `__init__.py`.
- Add the directory layout described in [Charter](./charter.md) as
  empty placeholders or stubs that import-test cleanly.
- Wire CI to no-op gracefully on the empty tree (totality query returns
  trivially when there are zero labels).
- This document and the charter, in the published mdbook.

### Phase 1 — author the initial vocabulary

`src/aegir/ontology/sdg-vocab.ttl` is authored fresh. The starting
content is what Ægir actually needs at M2: BFO 2020 + CCO upper
structure, the Schema.org and DBpedia terms that map to the SOTAB v2
and GitTables benchmark label sets we train against, and whatever
bespoke `sdg:`-namespace entities the project decides to introduce
on its own terms.

External TTL working sets (the sibling project's vocabulary, OBO
Foundry exports, etc.) are reference material, not source-of-truth
inputs. The author may consult them while writing
`sdg-vocab.ttl`, the same way they'd consult any reference document.
A small mechanical helper at `scripts/extract_public_terms.py` can
pull just the public-namespace IRIs out of a reference TTL into a
plain text scratchpad — a convenience for sifting through large
external files — but it does not produce committable output. The
TTL that lands in `sdg-vocab.ttl` is the author's edit.

Concrete process:

1. Draft `sdg-vocab.ttl` by hand, aiming for benchmark-label
   coverage that the M2 dataset loader can resolve. Every term:
   - has an `rdfs:label`,
   - has a `skos:definition`,
   - and (for `sdg:`-namespace terms) at least one
     `rdfs:subClassOf` edge to a public-namespace ancestor.
2. Author the SPARQL queries (`totality.rq`, `ancestry.rq`,
   `coverage_by_namespace.rq`) against Ægir's own namespace
   conventions.
3. Author `scripts/build_vocab_label_map.py`: TTL in, versioned
   `vocab_label_map.json` out. The initial map covers whatever
   benchmark labels the authored vocabulary supports — likely a
   partial picture of SOTAB v2 Schema.org and a partial picture of
   GitTables DBpedia. Coverage extends iteratively as the project
   demonstrates need.
4. Author `src/aegir/ontology/label_map.py` with `load()`,
   `iri_for()`, `bfo_ancestry()`, `labels_for_namespace()` helpers.
5. Wire the [mechanical checks](./charter.md#provenance-discipline) into
   CI: TTL parse + label/definition presence + BFO ancestry presence
   for `sdg:` terms + label-map JSON consistency + SPARQL totality.
6. External working sets are unaffected.

Ongoing discipline lives in PR review, guided by a short
`src/aegir/ontology/PROVENANCE.md` reviewer's guide. The mechanical
TTL checks described in
[Charter §Provenance discipline](./charter.md#provenance-discipline) cover
structural integrity. Whether a candidate term reads as the project's
own engineering work or as material lifted from elsewhere is an
authorship judgment, made the way every other code-review judgment is
made.

### Phase 2 — `_LABEL_DIMS` reconciliation + `label_to_iri` resolver

- Fix the stale `_LABEL_DIMS["sotab"] = 91` to `82` in
  `src/aegir/data/table_dataset.py` (this is mechanical; the SOTAB v2
  CSV union is verified at 82 distinct labels).
- Add a `label_to_iri(benchmark, label)` resolver to the table dataset
  that consumes `vocab_label_map.json` at load time.
- This is the smallest end-to-end vertical slice that proves the
  pipeline: a training run can now ask "what's the BFO ancestry of the
  label this column got?" and get an answer from the in-tree map.

### Phase 3 — empirical-gate validation

The v2→SOTAB head fine-tune runs from
`outputs/mixed-v2/20260426T232240Z/final.pt`. The
[v2 → SOTAB head fine-tune gate](../training_regime.md#11-the-v2-sotab-head-fine-tune-gate)
specifies the liveness thresholds:

- ≥ 3 distinct embedding clusters at coarse MCL inflation
- ≥ 0.10 macro F1 on SOTAB v2 CTA validation
- predictions distributed across ≥ 10 distinct labels

If these fail, **vocabulary expansion is paused** and the model issue
is debugged first. No vocab edits land in main while the gate is red.

### Phase 4 — synth migration (planned, not committed)

- Decide the consumer-side consumption pattern *before* moving code.
  Three options on the table:
  1. **Vendored snapshot**: sibling ships a frozen sample under its own
     `samples/` for tests; Ægir owns regeneration via
     `scripts/snapshot_synth_corpus.py`. Lowest friction.
  2. **Sibling-repo dependency**: sibling imports from
     `src/aegir/synth/` directly. Requires the sibling to keep Ægir
     installable in its dev environment.
  3. **Thin client**: the leaderboard gateway (or a dedicated synth
     gateway) exposes a generation endpoint. Highest deployment cost,
     also the most flexible end state.

  Default plan: vendored snapshot during transition; thin client only
  if a customer deployment needs out-of-process generation later.

- Once the consumption pattern is decided, copy the generator code
  in. Then port the priority registry. Then port the integration tests.

- Cutover for synth runs in parallel with vocabulary cutover but is not
  required to be simultaneous. Sibling-project tests against
  vendored-snapshot fixtures keep passing through the whole transition.

### Phase 5 — `vocab_label_map.json` v1.0.0

- All benchmark labels in `_LABEL_DIMS` (sotab, sotab-dbp,
  sotab-dbp-re, gt-dbp) covered with at least direct or
  subsumption-reachable BFO ancestry.
- Liveness gate passed.
- Synth migration complete or vendored-snapshot stable.
- Tag the release. The sibling project's consumer-side commit can land
  against the tagged URL.

This is the cutover moment. Sibling's TTL is now documentation.

### Phase 6 — vocabulary expansion (Ægir-defined, not Atelier-tiered)

Once v1.0.0 is out and the cutover is done, vocabulary expansion is
genuinely Ægir's call. The Atelier-side three-tier breakdown
(measurement zoo, subClassOf plumbing, Product/JobPosting/economics) is
a useful reference for *what's missing*, but the order, scoping, and
implementation milestones are decided here against Ægir's empirical
priorities — which datasets are getting trained against, where coverage
gaps are blocking benchmark progress, and where vocabulary expansion
can plausibly help model performance vs. just adding label-space
breadth.

In particular: if a v3 corpus mix wants ontology-conditioned synthetic
slices, the vocabulary work prioritizes whatever the v3 corpus needs.
That decision lives in the v3 design notes, not in a tier doc imported
from a consumer.

## What we do not commit to here

- A timeline. The empirical gate (Phase 3) is a real gate; if the
  v2→SOTAB head fine-tune surfaces architectural issues, the rest of
  the migration waits. We commit to the *order*, not to dates.
- A unification of the Schema.org and DBpedia label sets. Both stay
  separate keys in the JSON map per the charter.
- A specific synth-migration consumption pattern. We commit to deciding
  before code moves, not to which option wins.
- Backwards compatibility with the sibling's TTL filename or directory
  structure. The renamed `sdg-vocab.ttl` is the canonical form.

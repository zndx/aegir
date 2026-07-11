# Ontology Authors Guide

This guide is written for an ontology engineer who wants to **extend the `sdg` ontology
independently** — add classes, definitions, roles, properties — and have a reasonable expectation
of *passing every quantitative and qualitative gate*, and ideally *improving on them*. It documents
the full metric suite, the formal quality gate, the disposal membranes, and the pre-registered
objectives, with the exact formulas, bands, and thresholds the tooling enforces. Nothing here is
aspirational: every number is the value the code checks.

The governing principle is **propose / dispose**. You (or an agent) *propose* axioms; a stack of
deterministic membranes *disposes* — admitting only what is well-formed, logically consistent under
the reasoner, and ontologically clean. Rigor is **enforced, not asserted**, and the two strongest
membranes (HermiT and OntoClean) are *un-fakeable*: you cannot talk your way past a contradiction or
an anti-rigidity violation. If your extension passes, it is genuinely rigorous; if it fails, the
gate returns the reason and you refine.

---

## 1. What you are extending

`sdg-ontology` is a **BFO 2020 / CCO-grounded** domain ontology, content-derived from FinePDFs and
realized to a HermiT-validated OWL artifact at `corpora/ontology/sdg-ontology.{omn,owl}` with a
consistency certificate at `corpora/ontology/HERMIT_CERTIFICATE.md`.

Its purpose is to be the **annotation vocabulary for Column Type / Column Property Annotation
(CTA/CPA)** over wide relational tables. That reframes what most of the classes *are*: they are not
leaf terms but **intermediate-depth subsumers** — the property-bearing classes a heterogeneous-but-
coherent column belongs to. A `driver_stops_schedule.stops_addresses` column holds a mix (origin +
destination, residential + business shipping addresses, each bearing an avg-time-on-site); no leaf
type fits — the right annotation is the *least common subsumer that is still property-bearing*,
e.g. `Address ⊓ ∃has-shipping-role ⊓ ∃avg-time-on-site`. **Defining these intermediate classes well
*is* building the annotation vocabulary.** When you extend the ontology, you are extending that
vocabulary, and the gates exist to keep every term a *coherent, grounded* annotation target.

### Namespaces

| prefix | IRI base | use |
|---|---|---|
| `bfo:` | `http://purl.obolibrary.org/obo/BFO_` | upper categories (numeric IRIs, e.g. `bfo:0000040`) |
| `cco:` | `https://www.commoncoreontologies.org/` | mid-level genera (numeric IRIs, e.g. `cco:ont00000713` = Vehicle) — **note https** |
| `fhir:` | `http://hl7.org/fhir/` | clinical/record types, bridged to `cco:ont00000958` |
| `iao:` | `http://purl.obolibrary.org/obo/IAO_` | annotation properties (`iao:0000115` = definition) |
| `sdg:` | `https://signals360.example.org/sdg#` | our own classes/properties |
| `skos:`, `rdfs:`, `owl:` | standard | labels, definitions, structure |

### BFO categories you will reach for

`bfo:0000040` material entity · `bfo:0000004` independent continuant · `bfo:0000002` continuant ·
`bfo:0000015` process · `bfo:0000031` generically dependent continuant (ICE) · `bfo:0000019` quality ·
`bfo:0000020` specifically dependent continuant · `bfo:0000023` role · `bfo:0000016` disposition ·
`bfo:0000034` function · `bfo:0000017` realizable entity.
Realizable-machinery properties: `bfo:0000055` realizes · `bfo:0000197` inheres-in · `bfo:0000196`
bearer-of · `bfo:0000054` realized-in.

---

## 2. How you author

Classes are authored as **catalog templates** — a Manchester-syntax skeleton with typed slots — that
the realizer renders, grounds, and validates into the OWL artifact. The live catalog is
**`src/aegir/ontology/catalog/catalog.json`** — fully derived (the hand-authored 01–07 seed families
are retired; the catalog is what the derive→promote loop has accreted). The content-first deriver
stages candidates in `catalog.candidate.json`; `scripts/promote_candidates.py` gates them in.
Discovery is via `schema.catalog_files()`. **Edit `catalog.json` (or better: let the derive→promote
loop accrete it), never `combined.json`** (regenerated).

### The slot DSL

```
{name:Type}                  e.g. {X:Class}, {p:ObjectProperty}, {Y:Class}
{name:Type:Bound}            subtype constraint: {X:Class:bfo:0000002}
```

`Type ∈ {Class, ObjectProperty, DataProperty, Individual}`. A `CatalogTemplate` carries
`manchester_template`, `slot_types`, `verbal_template`/`verbal_templates` (NL surface forms → the
definition annotation + the corpus verbalization frames), `bfo_anchor_path`, `provenance`
(`pattern` / `tier` / `grounds_ddl` / `domain` / `source_span` — the derivation signals downstream
stages consume), and the **authored retrieval surfaces** (`alt_labels`, `scope_note` — membrane-gated
annotation content, never ontological claims; see §3.5). Templates bind to an axiom pattern from
`src/aegir/ontology/patterns.py` (tiers: `fhir_minimum` / `owl2_core` / `odp` / `sysmlv2`). Three
canonical shapes:

```
Class: {X:Class} SubClassOf: {Y:Class}                              # primitive (a kind, undefined)
Class: {X:Class} SubClassOf: {p:ObjectProperty} some {Y:Class}      # existential restriction
Class: {X:Class} EquivalentTo: {Y:Class} and {p:ObjectProperty} some {Z:Class}   # DEFINED (genus + differentia)
```

### Manchester conventions the membranes enforce

1. **Prefixes are lowercase only** — `cco:` `bfo:` `fhir:` `sdg:`, never `CCO:`/`BFO:`.
2. **Every property is prefixed** — `sdg:hasMeasurement some X`, never bare `hasMeasurement`.
3. **Coined classes/properties use `sdg:`** (camelCase) — do **not** invent `cco:`/`bfo:` names; those
   are numeric IRIs you must look up (see §3). No `#` comments (they break the OMN parser).
4. **The genus must be a *broader* class** — a real BFO/CCO/sdg parent, never the class itself.

### `EquivalentTo` vs `SubClassOf` — the single most important authoring choice

A class with `SubClassOf:` is **primitive** (necessary conditions only). A class with
`EquivalentTo: genus and differentia` is **defined** (necessary *and sufficient*): anything that is
the genus and satisfies the differentia *is* an instance. **Prefer `EquivalentTo` wherever the
differentia are genuinely sufficient** — this is the `definitional_completeness` lever and the IOF
discipline (the IOF defines ~55% of its terms this way). Do not force it: a genuine natural kind
whose essence is not captured by the stated relations should stay primitive. Reserve `EquivalentTo`
for *kinds*; model *roles* with the realizable pattern (§3), not as defined subclasses.

### Grounding: choosing a genus

Every class must chain to a BFO category — directly, or through CCO/FHIR. **Look up the real IRI**
with the grounding-anchor retriever rather than inventing one:

```
uv run --no-sync python scripts/grounding_anchors.py query "shipping address"
#   0.74 [cco]  Mailing Address     cco:ont00000xxx  <!-- coined-ok: placeholder IRI -->
#   0.59 [fhir] Address             fhir:Address
#   0.55 [sdg]  StopLocation        sdg:StopLocation   (reuse our own)
```

The index spans CCO (1431 BFO-aligned genera), FHIR R5 (210 record types, bridged to
`cco:ont00000958`), and **our own grounded classes** (the index accretes — each class
you ground becomes a reusable anchor). Prefer, in order: an existing `sdg:` class (reuse), a CCO/FHIR
genus, then a bare BFO category as a last resort. A generic `bfo:0000040` placeholder where you meant
"Patient" is *grounded but shallow* — find the real genus.

### Roles and the realizable machinery

A **role** is anti-rigid and relational (supplier/operator/origin-address: the bearer could stop
being it and still exist). Model it as a BFO role, never a rigid subclass:

```
Class: {OperatorRole:Class} SubClassOf: bfo:0000023,
   bfo:0000197 some {Operator:Class}, bfo:0000054 some {OperationProcess:Class}
```

The `inheres-in` (`bfo:0000197`) and `realized-in` (`bfo:0000054`) restrictions are what the
`realizable_machinery` metric counts and what BFO discipline requires.

---

## 3. The quantitative metrics

All metrics are computed by `scripts/ontology_metrology.py::compute()` (pure rdflib, JVM-free) over
the realized `.owl`, **with CCO's subClassOf backbone merged** so that `cco:`-grounded chains resolve
to BFO. Run:

```
uv run --no-sync python scripts/ontology_metrology.py corpora/ontology/sdg-ontology.owl   # or --json
```

`n` = number of `sdg:` named classes. Each metric below lists its **formula**, its **IOF/field
target**, and the **authoring lever** that moves it.

### 3.1 IOF-derived rigor dimensions (what field-standard suites miss)

| metric | formula | target | lever |
|---|---|---|---|
| `definitional_completeness` | `‖{c : c owl:equivalentClass …}‖ / n` | **IOF ≈ 0.55** | write `EquivalentTo` (genus+differentia) for definable kinds |
| `bfo_grounded` | `‖{c : subClassOf/≡-genus chain reaches a BFO IRI}‖ / n` | **1.0** | ground every class to a BFO/CCO/FHIR genus |
| `realizable_machinery` | count of restrictions on `realizes/inheres/bearer/realized` props **or** `some role/disposition/function` | **IOF ≥ 14** | model roles/dispositions/functions with the realizable pattern |
| `def_annotation_coverage` | `‖{c : rdfs:comment ∨ iao:0000115 ∨ skos:definition}‖ / n` | **1.0 (IAO req)** | supply a `verbal_template`; the realizer emits `iao:0000115` + `rdfs:comment` |

These are the **discriminators**: an LLM (or a hasty author) recovers taxonomy + existentials
(structure) but not sufficiency, full grounding, or BFO role discipline (rigor). They are where the
`FunctionalAdequacy` gate floor lives.

### 3.2 Field-standard structural metrics (OntoQA / OQuaRE)

| metric | formula | reading |
|---|---|---|
| `rr` relationship richness | `n_∃some / (n_subClassOf + n_∃some)` | non-taxonomic richness; a pure tree → 0 |
| `ir` inheritance richness | `n_subClassOf / n` | subclasses per class |
| `ar` attribute richness | `n_DatatypeProperty / n` | typed data attributes per class |
| `aronto` axiomatic strength | `(n_∃some + n_∀only + n_card) / n` | restrictions per class |
| `dit` depth | longest `subClassOf` chain | taxonomic depth (more developed = deeper) |
| `tm` tangledness *(inverted)* | `‖{c : >1 named parent}‖ / n` | multiple-inheritance load; **lower is better** |

### 3.3 OntoClean taxonomic-correctness proxies (un-gameable)

Reasoner-*invisible* defects that a generic LLM cannot fake (it names meta-properties at ~96% but
cannot operationalize them). Computed via the OntoClean classifier (`src/aegir/ontology/ontoclean.py`).

| metric | formula | target |
|---|---|---|
| `subsumption_cycles` | classes reachable from themselves via `subClassOf` (OOPS! P06) | **0 (hard)** |
| `ontoclean_violations` | `subClassOf` edges where an **anti-rigid (role) parent subsumes a non-anti-rigid (rigid) child** | **0** |
| `sibling_adjudication` | fraction of *adjudicable* sibling pairs carrying a verdict — `disjoint` (a realized `DisjointClasses`, HermiT-enforced) **or** explicit `overlap` (recorded) | **1.0** |
| `sibling_disjointness` | the disjoint *share* of the adjudicable universe (sub-stat, not a target — roles co-inhere; blanket disjointness is a false claim) | report |
| `sibling_pairs_grounding_debt` | sibling pairs sharing only a **bare BFO genus** — not siblinghood but grounding debt (the `bfo_grounded` lever): *ground first, then adjudicate* | ↓ |
| `orphan_rate` | fraction of `sdg:` classes with no parent (OOPS! P04 — islands) | → 0 |
| `taxonomic_cleanliness` | `1 − (subsumption_cycles + ontoclean_violations) / n_subClassOf` | **1.0** |

> **Sibling adjudication (2026-07-09, supersedes the naive disjointness→1.0 objective).** The
> universe = pairs sharing a *specific* genus (CCO/`sdg:`, or a BFO realizable where co-membership
> matters); the closed loop (`scripts/adjudicate_siblings.py`) nominates disjoint pairs over an
> explicit overlap-by-default, gated by membranes that return reasons: shape/membership, **ABox
> refutation** (closure-aware: an individual *inferred* into both classes names itself as the
> witness), and **batch HermiT** over the full theory (with a parse guard — a degraded OWLAPI
> parse is run-fatal, never a vacuous pass). Verdicts persist in `catalog/adjudications.json`;
> the realizer emits the `DisjointClasses` frames (full-IRI form — the Manchester parser
> silently drops prefixed frames against full-IRI declarations) inside the HermiT boundary,
> where kvasir also reads them natively.

### 3.4 Consistency

HermiT over the realized ontology *with CCO imported*. Consumed by the gate from the certificate
(`isConsistent: true`). **Zero unsatisfiable classes** is the real bar — an ontology can be
`isConsistent` yet contain unsatisfiable classes (classes that can have no instances); both must be
clean for a publish.

### 3.5 Topic-layer instruments (the retrieval face)

Every term you author is also a **topic anchor**: the inverted topic layer
(`src/aegir/ontology/topic_layer.py`; phase gate
[here](../roadmap/phase_gate_inverted_topic_layer.md)) embeds each live catalog term as a ColBERT
multivector in qdrant (`sdg_topics`, 29 SKOS domains + all 433 terms), and FinePDFs items
(anchor-proportional passage windows) associate with **one topic or none** by hierarchical-margin
MaxSim. Your class's *retrieval surface* — `term_anchor_text()` composes prefLabel · `alt_labels` ·
verbalization frames · the axiom's referenced vocabulary resolved to labels/definitions ·
`source_span` · `scope_note` — is therefore a measured artifact with its own gates:

| instrument | what it enforces | value |
|---|---|---|
| **τ\*** (`topic_layer.derive_tau`) | the assignment gate on `rel_margin_h` — the (1−α) quantile under the shuffled-window null, pre-registered, **report-not-tune** | **0.1065** (α = 0.05; lens-identity-scoped — new lens ⇒ re-derive) |
| **alignment / margins** (`associate_items`) | one item → one topic via `rel_margin_h` (margin over the nearest **non-ancestor** competitor; a parent/child near-miss is not ambiguity); flat margin recorded alongside; every association content-addressed to the registry's `collection_sha` | assigned ∨ UNASSIGNED — the ambiguous mass is the error signal |
| **M1–M6, M8** (`src/aegir/ontology/annotation_membrane.py`) | authored surfaces (`alt_labels`/`scope_note`/`definition`): M1 shape · M2 token bounds vs the registry's granularity · M3 no smuggled ontological claims (taxonomy lives in the axiom, not prose) · M4 vocabulary grounding · M5 clean-room tripwire · M6 discrimination incl. the **self-retrieval margin gate** (the surface must retrieve its own anchor at rank-1) · **M8 positive voice** | verdict + reason, re-prompt carries the reason |
| **M7 basin gate** (`topic_layer.basin_calibration`) | registry changes gated on the *input* side: a topic flags when its assigned mass explodes between pinned runs (`new ≥ 5 ∧ new ≥ 3× base`), with doc-concentration evidence attached; disposition is manual and provenance-recorded — the gate never silently mutates | PASS = zero unresolved flags |

**M8 is the durable methodology rule:** *definition-by-negation is an embedding anti-pattern.* A
similarity filter has no negation operator — "not used for X" injects X as attractor mass, and
"distinct from ⟨sibling⟩" injects the sibling's vocabulary into this anchor. Anchor surfaces are
**positive-voice only**: discriminate by specific positive vocabulary (participants, artifacts,
settings, instruments), never by exclusion or contrast.

Authored surfaces land on the `CatalogTemplate` with a `skos_authoring` provenance record (date,
model, membranes, registry sha, dispositions) — `catalog.json` stays the single source of truth. On
the output side, **congruence** (`src/aegir/ontology/congruence.py`) classifies generated chapters
against the same ColBERT substrate — the successor to the BERTopic-era R_D
(`topic_alignment.py`, deprecated).

---

## 4. The OQuaRE quality gate

`scripts/ontology_oquare.py` is the **formal publish gate**. OQuaRE (Duque-Ramos et al. 2011) adapts
ISO/IEC 25000 (SQuaRE) to ontologies: each metric is normalized to **[1,5]** against fixed,
IOF-anchored bands, then aggregated into **six characteristics** and one holistic score.

```
uv run --no-sync python scripts/ontology_oquare.py corpora/ontology/sdg-ontology.owl \
    --certificate corpora/ontology/HERMIT_CERTIFICATE.md
```

### 4.1 Normalization bands (fixed a priori — a stable distance-to-IOF)

Piecewise-linear interpolation between breakpoints `(value, score)`, clamped to [1,5]:

| metric | `→1` | `→3` | `→5` |
|---|---|---|---|
| `definitional_completeness` | 0.00 | 0.25 | **0.55** |
| `bfo_grounded` | 0.50 | 0.85 | **1.00** |
| `realizable_machinery` | 0 | 5 | **14** |
| `def_annotation_coverage` | 0.00 | 0.70 | **1.00** |
| `rr` | 0.00 | 0.25 | 0.50 |
| `ir` | 0.00 | 1.00 | 3.00 |
| `ar` | 0.00 | 0.30 | 1.00 |
| `aronto` | 0.00 | 0.60 | 1.50 |
| `dit` | 1 | 3 | 8 |
| `tm` *(inverted)* | 0.50 | 0.15 | 0.00 |
| `consistent` | inconsistent | unknown | consistent |

### 4.2 Characteristics (which metrics feed each)

| characteristic | constituent metric scores |
|---|---|
| **Structural** | aronto, dit, tm, bfo_grounded, rr |
| **FunctionalAdequacy** | definitional_completeness, realizable_machinery, def_annotation_coverage |
| **Reliability** | bfo_grounded, consistent, tm |
| **Operability** | def_annotation_coverage, rr |
| **Maintainability** | tm, dit, ir |
| **Transferability** | bfo_grounded, def_annotation_coverage |

`aggregate` = mean of the six characteristics.

### 4.3 The gate (GREEN requires all three)

| check | floor |
|---|---|
| `oquare_aggregate` | **≥ 3.5** |
| `functional_adequacy` | **≥ 3.0** |
| `hermit_consistent` | **== true** |

**AIM 3.9** — the published OQuaRE class of Brick (3.93) / RealEstateCore (3.91). The
`FunctionalAdequacy ≥ 3.0` floor is deliberate: it forces *definitional rigor and BFO discipline*,
not structural/grounding gains alone. The gate is wired HARD into `aegir.lineup.sync._gate()`:
**`sync --push` of the ontology is refused below GREEN.** You will not publish a regression.

---

## 5. The disposal membranes (what rejects your extension, and why)

Your axioms pass through these in order. Each returns a *reason*, so a failure is a repair
instruction, not a dead end (this is the agent-mediated feedback loop; a human author reads the same
reasons).

1. **Parse membrane** (`evolve_rigor.validate_detailed`) — renders the axiom standalone and parses it
   under OWLAPI. Rejects malformed Manchester: uppercase prefixes, bare properties, undeclared
   entities, `#` comments. Reason: the parser error or "0 classes."
2. **Reasoning-authority membrane** (`build_realized_ontology.consistency_check`) — imports CCO and
   runs HermiT, so your grounding is validated against **CCO's disjointness axioms**. A class grounded
   to a CCO-*disjoint* or BFO-*incompatible* genus (e.g. a `Plant` placed under `cco:ont00000713`, or a
   continuant genus where a process is required) is **unsatisfiable** and rejected. Reason: "genus X is
   incompatible — re-ground to a compatible parent." *This is un-fakeable.*
3. **OntoClean meta-property membrane** (`src/aegir/ontology/ontoclean.py`) — assigns Rigidity /
   Identity / Unity / Dependence and enforces the OntoClean constraint that an **anti-rigid property
   cannot subsume a rigid one** (a role cannot be the parent of a kind). Surfaces as
   `ontoclean_violations`. *Also un-fakeable* — reasoner-invisible yet checkable.
4. **Annotation membranes M1–M8** (`src/aegir/ontology/annotation_membrane.py`) — for authored
   *retrieval surfaces* (`alt_labels`, `scope_note`, definitions), not axioms: HermiT has no stake in
   prose, so these membranes are the whole gate (§3.5). Reasons name the failing rule (and, for M6,
   the sibling to differentiate from — the highest-value re-prompt feedback).

A self-check before you propose:

```
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py --strict-grounding
uv run --no-sync python src/aegir/ontology/ontoclean.py src/aegir/ontology/catalog/catalog.json
just check-ontology-schema      # TTL parses, labels/definitions present, BFO ancestry, SPARQL totality
```

---

## 6. The pre-registered objectives (`EVIDENCE.md`)

Two standing objectives define "good enough to publish" and "rigorous":

- **OQ-Structure** — `bfo_grounded ≥ 0.95` ∧ `def_annotation_coverage ≥ 0.90` ∧ `ar > 0` ∧
  `oquare_aggregate ≥ 3.5`. Gate: the `sync._gate` publish gate.
- **OQ-Rigor** — `definitional_completeness ≥ 0.45` ∧ `realizable_machinery > 0`. Gate: the OQuaRE
  `FunctionalAdequacy ≥ 3.0` floor.

An extension that *holds or raises* both objectives is the bar to clear. The standing rule: **no
`sync --push` of the ontology Data Product until OQuaRE is GREEN.**

---

## 7. Worked example — authoring an intermediate class end-to-end

Goal: a class for the `stops_addresses` column — shipping addresses (origin + destination) bearing an
avg-time-on-site.

**(1) Decide the modeling.** A *kind* (an Address is rigidly an address) → define it with
`EquivalentTo`. The shipping/origin/destination facet is a *role* the address bears, not a rigid
parent — so it enters as a `realizes`-style differentia, keeping the genus an Address.

**(2) Ground the genus.** `grounding_anchors.py query "mailing address"` → reuse `sdg:PostalAddress`
if present, else `cco:ont…` (Mailing Address). Coin `sdg:` only for genuinely new differentia.  <!-- coined-ok: prose -->

**(3) Author.**

```
Class: {ShippingStopAddress:Class} EquivalentTo:
   cco:ont00000xxx  <!-- coined-ok: placeholder IRI -->
   and sdg:bearsShippingRole some {ShippingRole:Class}
   and sdg:hasAverageTimeOnSite some xsd:duration
Annotations: rdfs:label "shipping stop address",
   iao:0000115 "A mailing address that bears a shipping role (origin or destination) on a driver
   stop schedule and has an associated average time on site."
```

**(4) Dispose.** Realize → HermiT (the genus `cco:…Address` is a material/ICE entity; no disjointness
violated → satisfiable). OntoClean (the genus is a rigid kind, not a role → no violation). Parse
(prefixes lowercase, properties prefixed → admitted).

**(5) Measure.** Re-run the metrology + OQuaRE gate. This class *raises* `definitional_completeness`
(an `≡`), holds `bfo_grounded` (real CCO genus), adds `def_annotation_coverage` (the `iao:0000115`),
and — because the role is modeled with a realizable differentia — nudges `realizable_machinery`.

**(6) Iterate.** If HermiT marks it unsatisfiable, the reason names the offending genus; pick a
compatible one and re-dispose. If `ontoclean_violations` rises, you placed a role as a rigid parent —
re-model it as a borne role.

---

## 8. How to *improve upon* the gates

Passing is the floor; the AIM is 3.9 and the IOF frontier beyond it. To raise each lever:

- **Definitional completeness toward 0.55+** — convert primitive kinds to `EquivalentTo` wherever the
  differentia are sufficient; *define the referenced intermediate classes* (the subsumers a column
  needs), not just the heads. This is the highest-leverage dimension and the one shallow extensions
  miss.
- **Realizable machinery toward 14+** — wherever a relational/anti-rigid concept appears, model it as a
  BFO role/disposition/function with `inheres`/`realizes` differentiae rather than a subclass.
- **OntoClean to a clean sheet** — keep `sibling_adjudication` at 1.0 (every adjudicable pair carries
  a verdict; nominate `disjointWith` only between identity-incompatible siblings — the ABox and HermiT
  membranes refute overreach) and keep `ontoclean_violations`/`subsumption_cycles` at 0. These are
  the un-gameable signals; a clean OntoClean profile is the field's blind spot and your differentiator.
- **Retire grounding debt** — `sibling_pairs_grounding_debt` counts pairs whose only shared genus is a
  bare BFO category; the fix is *re-grounding to specific CCO/domain genera* (which also lifts
  `bfo_grounded` and grows the adjudicable universe), never adjudicating meaninglessness.
- **Annotation rigor** — supply genus-differentia definitions (not vacuous label-glosses); the
  `iao:0000115` should be a *real* sufficient definition, mirroring the `EquivalentTo`.
- **Contribute patterns** — recurring genus-differentia or role shapes belong in
  `src/aegir/ontology/axiom_patterns.json` (DOSDP-style: the defining axiom lives in the pattern, you
  fill slots). Reserve `equivalentClass` for kinds; emit roles via the realizable pattern — do not
  conflate them (Neuhaus 2025: roles resist `≡`).

The two reasoners are the discipline you cannot circumvent: **HermiT** rejects any grounding that
contradicts CCO's disjointness, and **OntoClean** rejects any anti-rigid-over-rigid subsumption.
Build with them, not around them, and your extension is rigorous by construction.

---

## 9. Reference — commands & files

```
# metrics + gate
uv run --no-sync python scripts/ontology_metrology.py corpora/ontology/sdg-ontology.owl [--json]
uv run --no-sync python scripts/ontology_oquare.py corpora/ontology/sdg-ontology.owl \
    --certificate corpora/ontology/HERMIT_CERTIFICATE.md [--json]
# membranes / realize (LD_LIBRARY_PATH bootstraps the JVM for HermiT/DeepOnto)
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/build_realized_ontology.py --strict-grounding
uv run --no-sync python src/aegir/ontology/ontoclean.py src/aegir/ontology/catalog/catalog.json
just check-ontology-schema
# grounding + agent-assisted authoring
uv run --no-sync python scripts/grounding_anchors.py query "<concept>"
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/define_intermediate_classes.py --rounds 4
uv run --no-sync python scripts/evolve_rigor.py --batch 12       # convert primitives → ≡ / roles
# derive → promote (the accretion path)
LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/promote_candidates.py   # candidate → catalog.json
```

| file | role |
|---|---|
| `src/aegir/ontology/catalog/catalog.json` | THE live catalog (edit this — or let the derive→promote loop accrete it — never `combined.json`) |
| `src/aegir/ontology/catalog/catalog.candidate.json` | deriver staging, promoted by `scripts/promote_candidates.py` |
| `src/aegir/ontology/SLOT_DSL.md` | the slot grammar |
| `src/aegir/ontology/patterns.py` + `axiom_patterns.json` | the axiom-pattern library (tiers) + DOSDP-style genus-differentia / role patterns |
| `scripts/ontology_metrology.py` | every metric (`compute()`) — the single source of truth |
| `scripts/ontology_oquare.py` | the [1,5] bands, characteristic map, FLOORS, the publish gate |
| `src/aegir/ontology/ontoclean.py` | the OntoClean classifier + meta-property membrane |
| `scripts/build_realized_ontology.py` | render → CCO import → HermiT → `.owl` + certificate |
| `scripts/grounding_anchors.py` | CCO+FHIR+accretive genus retrieval |
| `src/aegir/ontology/topic_layer.py` | the topic registry (`sdg_topics`), item adjudication, τ\* derivation, M7 basin gate |
| `src/aegir/ontology/annotation_membrane.py` | M1–M8 over authored retrieval surfaces |
| `src/aegir/ontology/congruence.py` | output-side concept congruence (the R_D successor) |
| `corpora/ontology/sdg-ontology.{omn,owl}` | the realized artifact (+ `HERMIT_CERTIFICATE.md`) |

**Citations.** OQuaRE: Duque-Ramos et al. 2011. IOF/BFO signature: Smith et al. 2019. OntoClean:
Guarino & Welty. CCO: Common Core Ontologies (CC0). FHIR: HL7 FHIR R5.

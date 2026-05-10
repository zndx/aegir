# Slot DSL

Aegir's catalog templates carry typed slots that the policy fills at
composition time and the verifier type-checks before computing *R_A*.
The grammar is deliberately small.

## Slot syntax

```
{name:Type}
```

- `name` — identifier; alphanumeric plus underscore. Two slots in the
  same template may share a name iff the policy is intended to bind
  them to the same filler.
- `Type` — slot type, drawn from the type lattice below.

Slots appear inside an OWL Manchester-syntax template string:

```
Class: {X:Class} SubClassOf: {p:ObjectProperty} some {Y:Class}
```

## Type lattice

Two layers, both admissible.

**Layer 1 — OWL primitives** (mandatory minimum, always supported):

| Type | Filler | Example fillers |
|---|---|---|
| `Class` | named OWL class IRI | `bfo:Continuant`, `sdg:Sample` |
| `ObjectProperty` | named OWL object property IRI | `bfo:partOf`, `sdg:hasMeasurement` |
| `DataProperty` | named OWL data property IRI | `sdg:hasUnit`, `dcterms:date` |
| `Individual` | named OWL individual IRI | `sdg:reference-instance-1` |

**Layer 2 — subtype constraints** (soft commitment for P1a):

A slot may further constrain its filler to descendants of a given IRI:

```
{X:Class:bfo:Continuant}
```

means "fill `X` with a `Class` whose `rdfs:subClassOf+` chain reaches
`bfo:Continuant`." Constraint validation is a P1a concern; the runtime
verifier does layer-1 type-checking by default and consults the
catalog's BFO-anchor metadata for layer-2 enforcement.

The Layer-2 form is documented here so that template authors at P1a
have a way to express domain-relevant constraints, but P1b's verifier
implementation may launch with Layer-1-only enforcement and add Layer
2 incrementally.

## Substitution semantics

Slot substitution is **simple textual replacement** of the entire
`{name:Type}` (or `{name:Type:Bound}`) expression by the IRI of the
filler. No parser-level transformation; no escaping. Manchester syntax
is line-oriented and IRIs do not contain `{` or `}`, so textual
replacement is unambiguous.

A composition is a sequence of `(template_id, slot_fillers)` pairs.
Rendering iterates over the sequence, substitutes each template's
slots, and concatenates the results separated by blank lines.

## Validation order at composition time

1. **Type check** — every slot's filler matches its declared
   `Type` (and, when supported, its subtype `Bound`).
2. **Reference closure** — every IRI named in any slot filler has
   a declaration somewhere else in the composition or comes from a
   public namespace (`schema:`, `dbo:`, `bfo:`, `cco:`, `skos:`,
   `prov:`).
3. **Render** — textual substitution into each template's
   Manchester string.

Any step 1 or step 2 failure short-circuits to *R_A* = 0; *R_B*,
*R_C*, *R_D* are not evaluated.

## Examples

```
template_id: subclass_basic
manchester_template: Class: {X:Class} SubClassOf: {Y:Class}
slot_types: {"X": "Class", "Y": "Class"}
```

```
template_id: existential_restriction
manchester_template: Class: {X:Class} SubClassOf: {p:ObjectProperty} some {Y:Class}
slot_types: {"X": "Class", "p": "ObjectProperty", "Y": "Class"}
```

```
template_id: equivalent_intersection
manchester_template: Class: {X:Class} EquivalentTo: {Y:Class} and {Z:Class}
slot_types: {"X": "Class", "Y": "Class", "Z": "Class"}
```

## Out of scope here

- Slot defaults / optional slots — every slot is mandatory.
- Anonymous classes (blank-node Manchester expressions). Catalog
  templates are always named-class-headed at the top level; nested
  expressions inside a restriction are part of the template body,
  not slots.
- Annotations beyond `rdfs:label` and `skos:definition`. Those two
  are added by the catalog construction step automatically; they are
  not template-level concerns.

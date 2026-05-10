# Phase 0 — ontology + synth scaffolding landed

*2026-05-09 — second session block of the day, after the docs revisit
+ provenance discipline + concept brief v0.3.*

## What landed

The migration's Phase 0 (per
`docs/current/ontology/migration.md`) has its scaffolding committed.
This is the parallel-safe technical chunk that runs alongside P0
literature review without coupling to lit-review outcomes.

### New files

- `src/aegir/ontology/__init__.py` — package marker, module docstring
  only, matching `src/aegir/gateway/__init__.py` house style.
- `src/aegir/ontology/schema.py` — `CatalogTemplate` and `Catalog`
  dataclasses with `load_catalog` / `save_catalog` JSON round-trip.
  Stdlib `@dataclass` + `field(default_factory=...)` per project
  convention (`src/aegir/config.py`, `src/aegir/data/serialization.py`,
  `src/aegir/models/config.py`).
- `src/aegir/ontology/SLOT_DSL.md` — typed-slot grammar
  (`{name:Type}`), OWL-primitive type lattice (Layer 1) plus
  optional subtype constraints (Layer 2), substitution semantics,
  validation order, examples.
- `src/aegir/ontology/PROVENANCE.md` — reviewer's checklist for
  authorship-judgment review, referenced from charter and migration.
- `src/aegir/ontology/catalog/__init__.py` — empty package marker.
- `src/aegir/ontology/catalog/examples.json` — four hand-authored
  template rows exercising the schema's variability (subclass,
  existential, equivalent-with-intersection, cardinality). Not the
  real P1a catalog; placeholder rows to exercise schema and
  validator.
- `src/aegir/synth/__init__.py` — package marker for the synth
  module that lands at P1+ when synth generators arrive.
- `scripts/check_ontology_schema.py` — tier-0 validator. Loads every
  `catalog/*.json`, instantiates `CatalogTemplate(**row)` per row,
  cross-checks slot placeholders against `slot_types`. Exits non-zero
  on failure.

### Edits

- `Justfile` — added `check-ontology-schema` recipe; made `bdd-0`
  depend on it so the tier-0 gate validates the catalog schema first.
- `src/aegir/data/table_dataset.py` — `TASK_NUM_CLASSES["sotab"] =
  91 → 82`. Verified at the union of v2 train / val / test /
  robustness CSVs.
- `train.py` — two co-located default updates: `SyntheticTableDataset`
  constructor `num_classes=91 → 82`, and the smoke-test fallback
  `args.num_classes or TASK_NUM_CLASSES.get(args.task, 91 → 82)`.

## Verification results

| # | Step | Result |
|---|---|---|
| 1 | `from aegir.ontology.schema import CatalogTemplate, Catalog` | ✓ clean (with `sys.path.insert`, matching project convention for `scripts/*.py`) |
| 2 | `python scripts/check_ontology_schema.py` | ✓ `4 templates validated, 0 errors` |
| 3 | `just check-ontology-schema` | ✓ runs validator, same output |
| 4 | `just bdd-0` | partial — `check-ontology-schema` dep ran first and passed; downstream `behave` step failed because `behave` is not in the active venv |
| 5 | `python train.py --smoke-test --model-size tiny --epochs 1` | blocked — `numpy` not in active venv |
| 6 | `mdbook build docs/` | ✓ clean |
| 7 | `git diff pyproject.toml uv.lock` | non-empty but **all changes predate this session** (gateway deps added previously); zero new deps from this work |

Steps 4 + 5 are blocked on the same root cause: the active venv does
not have core training / BDD deps (`numpy`, `behave`, `torch`, etc.)
installed. This is an environment-state concern that the project's
`uv sync` recipe handles at devenv-up time — but the project's
CLAUDE.md is explicit that `uv sync` is never invoked autonomously
because it clobbers patched CUDA wheels. The blocked steps are
therefore deferred to the next session where the user can rebuild the
venv via the devenv flow.

## What's deliberately not done

- **No DeepOnto wiring.** `is_complex`, `verbal_template`,
  `mean_verbal_length` are placeholder-fillable in `CatalogTemplate`
  (defaults to `False`, `""`, `0.0`). The offline DeepOnto pass that
  populates them is P1a, scoped after P0 lit review exits.
- **No new `pyproject.toml` dependencies.** The schema-only work uses
  only stdlib (`json`, `dataclasses`, `pathlib`, `re`, `sys`).
- **No real catalog content.** `examples.json` is four hand-authored
  rows that exercise the schema's variability; the real P1a catalog
  authors hundreds of templates against the lit-review-derived
  positioning.
- **No mdbook navigation entry.** `SLOT_DSL.md` and `PROVENANCE.md`
  are reviewer-facing artifacts that live alongside the code they
  govern, not browsing destinations in the published book.
- **No literature review.** That's the headline next unit on the
  intellectual track, scoped at `docs/scratch/2026-05-09/` with its
  own deliverable.

## Path forward

P0 lit review remains the committed next unit per
`docs/current/ontology/concept_brief.md` v0.3. It produces a
positioning document at `docs/scratch/YYYY-MM-DD/HHMMSS_lit_review.md`
with explicit "we are not aware of prior work that..." statement;
its exit gate determines whether v0.3's contribution claim survives
or the brief is revised.

P1a (catalog construction proper) and P1b (runtime verifier) are
unblocked once P0 exits. The schema this session committed is the
contract those phases author into.

# Ægir — agent routing (pairing work)

Read this before editing. Long-form model/ontology docs: `CLAUDE.md` and
`docs/current/src/`. Scratch notes: `docs/scratch/YYYY-MM-DD/`.

## Current objective (2026-09-16)

Emit a **disjoint NHSVM-train reference sample** for Atelier’s holdout
target. The accept gate is Atelier `assert_pair`, not a new Ægir checker.

**Brief (start here):**
`docs/current/src/operations/atelier_holdout_reference.md`
(scratch copy: `docs/scratch/2026-09-16/054649_atelier_holdout_reference_brief.md`)

| | |
|--|--|
| Target (holdout) | `sdg-corpora/b24ef9f60660_macbook` |
| Reference (train) | `sdg-corpora/b24ef9f60660_reference` |
| Pin | `b24ef9f606605dd7bb8877bddd2d9672c78282ad` |
| Forbidden slugs | `computational-material-record-044e3d9a`, `position-ac7bde24` |
| Gate | `/home/rch/local/src/zndx/atelier` → `atelier.sdg.pair.assert_pair` |

Do **not** train on the target. Do **not** overlap collections. Genus
IRI miss is `#AT.00000021.PAIRIRI`. SKOS leftovers are `skos_missing`
(Ægir emit list). `perfect_possible` must be true before Atelier DST
on the target.

## Where to work

| Area | Paths |
|------|--------|
| Collection emission | `scripts/build_collections.py`, `scripts/relational_collections.py` |
| FinePDFs / SchemaPile | `scripts/emit_finepdfs_postings.py`, `scripts/emit_schemapile_postings.py` |
| SKOS | `scripts/build_skos_vocab.py`, `scripts/author_skos_surfaces.py` |
| Atelier release / score | `scripts/build_atelier_release.py`, `scripts/score_atelier_predictions.py` |
| Lattice peer | `docs/current/src/operations/peer-unit.md` (`:50151`) |

## Invariants

- sdg-corpora pin is the SoR; do not silently retarget another commit.
- Collections are RI-closed bundles; never split a bundle to fit a budget.
- Entity names without SKOS stay gaps until a term exists — do not invent
  codes in Atelier.
- Product GPU/LLM for Atelier classify is Engine/Complete thinking|instruct
  (OIP); this Ægir task is corpus emission, not a second classify clock.

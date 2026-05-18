# Structural alignment proxy — column vocabulary by ablation arm

**Date:** 2026-05-18 00:24 UTC
**Script:** `scripts/structural_alignment.py`
**Inputs:** 30-chapter ablation pilot (10/arm)

## What this measures (cheap pre-training proxy)

The full downstream evaluation requires training Aegir-tiny on each arm's corpus and measuring CTA/CPA F1. With only 200 chapters per arm (~1M tokens), that's 270× under-Chinchilla — the signal would be noise.

This proxy doesn't require training. For each chapter, parse markdown tables, extract column headers, and measure each column name's overlap with two vocabulary sets:

- **SOTAB-CTA Schema.org vocabulary** (78 terms — Person, Place, Event, Product, Offer slot fields)
- **Our 540-template ontology slot-vocabulary** (451 terms — drawn from slot_types + template_id keywords)

A column is "aligned" if ≥50% of its words appear in the vocabulary (substring matching, either direction).

## Result

| arm | chapters | tables | columns | sotab_align% | mean_sotab | onto_align% | mean_onto |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 10 | 48 | 208 | 28.4% | 0.200 | **88.5%** | **0.853** |
| **no-ontology** | 10 | 48 | 256 | **35.2%** | **0.334** | 57.4% | 0.559 |
| no-schema | 10 | 34 | 147 | 19.7% | 0.119 | 89.1% | 0.728 |

## Interpretation

**1. Ontology IS load-bearing for ontology grounding.** The full arm's 88.5% ontology-alignment (vs no-ontology's 57.4%) confirms the axioms ARE flowing through to table headers, not just decorating prose. The 31-pp gap is the mechanical effect of asking the model to use slot-typed vocabulary.

**2. Ontology HURTS Schema.org alignment.** The no-ontology arm produces +6.8 pp more SOTAB-CTA-aligned columns. Reason: our ontology vocabulary (governance, attestation, audit, requirement, control, evidence...) overlaps poorly with Schema.org's commerce/person/event world.

**3. No-schema regresses on both axes** for the obvious reason — it has 30% fewer tables (34 vs 48) and 29% fewer columns (147 vs 208). The schema instruction is what gets the model to materialize the LIMS structure into tables; remove it and the prose stays prose.

## What this answers for the user's question

> Does ontology push relational corpus to SOTA, or is it ceremony?

**Neither, exactly.** Ontology pushes the corpus toward **its own vocabulary regime** and away from Schema.org's. Whether that's "SOTA" or "ceremony" depends on the downstream task:

- For **SOTAB-CTA** specifically (the v0.2 benchmark in our roadmap): no-ontology produces tables structurally closer to Schema.org. Full pipeline data may actively hurt SOTAB-CTA performance.
- For **ontology-grounded annotation** (CPA-style tasks against our own 540-template axioms): full pipeline produces the only data that carries the signal.
- For **"general LIMS column-type annotation"** as a downstream task category: neither vocab is canonical; whichever the eval-target uses wins.

The implication: the v0.2 SOTAB-CTA benchmark may be the *wrong evaluation target* for the v0.3 thesis. Publishing the chapter corpus + ontology-grounded CPA eval (which we control) is more aligned with the actual claim than chasing SOTAB-CTA F1.

## Top-12 column headers per arm — qualitative

```
full (governance ontology vocabulary dominates):
  [.O] requirement_id, control_id, verification_id, evidence_id, audit_id,
  [.O] attestation_id, policy_id, enforcer_id, control_name
  [SO] name, person_id  (these double-hit because "name"/"person" are Schema.org too)

no-ontology (LIMS-of-the-moment, more commerce-shaped):
  [SO] candidateid, reviewid, reviewerid, name, fullname
  [.O] status, requestid, unitid
  [..] applicationid, meetingid, qualified  (uncovered terms)

no-schema (fewer tables, when present still ontology-shaped):
  [.O] retention period, attestation class, specifies target
  [SO] control identifier, requirement identifier, policy identifier
```

The qualitative read confirms the quantitative finding: removing axioms collapses ontology-vocab usage; removing the schema-prompt collapses table density; the full pipeline produces ontology-dense but Schema.org-sparse tables.

## What we should add to the verifier

R_axiom (table-header slot-vocab match) already captures the "are tables ontology-aligned" half. We should add an `R_downstream` factor parameterized by the *target* downstream task's vocabulary, so we can:

- Score chapters against multiple downstream-task vocabularies simultaneously
- Choose the corpus mix based on which downstream task we're targeting
- Detect when the corpus diverges from the intended eval-target before scaling

This is a small extension to `scripts/verify_chapters.py` — one new scorer + an optional `--target-vocab` flag.

## Next steps tonight

The ablation v1 (600 chapters across 3 arms) is running in background. Once it completes:
1. Re-run topic correspondence + verifier + structural alignment on full 200-per-arm data
2. The structural-alignment numbers above should sharpen with 20× more data — if they flip sign, the pilot was noise; if they hold, this is the production result for tomorrow's discussion
3. Decide on Aegir-tiny training: train at 200-chapter scale knowing signal will be weak, or wait for 5K-chapter scale (~$30, 1 day) for meaningful differential

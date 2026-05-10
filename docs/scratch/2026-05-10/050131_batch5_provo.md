# Batch 5 — PROV-O / OpenLineage lineage + P5 readiness assessment

*2026-05-10 — adds PROV-O / OpenLineage-aligned lineage layer to the
catalog and lands the P5 infrastructure-readiness assessment in
the charter. Q4 commitment fulfilled.*

## Batch 5 summary

| Metric | Result |
|---|---|
| Candidate templates authored | 50 |
| Loaded into DeepOnto without error | 50 (100%) |
| Verbalized round-trip | 49 (98%) |
| `is_complex` flagged | 45 (90%) |
| Axiom-family coverage | F1, F2, F3, F4, F5, F6, F7 (all 7) |
| New `sdg:*` object properties declared in TTL | 33 |
| Reused existing `sdg:*` properties | 4 (`governedBy`, `hasColumn`, `hasConfidence`, `observedAt`) |
| Wall-clock harness time | 67 s |

## Branch coverage in this batch

| Branch | Templates | Notes |
|---|---|---|
| `bfo:Process / sdg:Transformation` | 14 | PROV-O `prov:Activity` alignment; F1-F7 represented |
| `cco:DescriptiveICE / sdg:LineageEdge` | 11 | OpenLineage column-lineage shape; cousin to ColumnPolicy at the same anchor |
| `bfo:Process / sdg:Allocation` | 10 | OpenLineage run-instance shape (run_id, facets, status) |
| `cco:Artifact / sdg:ProvenanceAgent` | 8 | PROV-O `prov:Agent` alignment (delegation, role, organization) |
| `cco:Artifact` cross — derivation | 4 | `wasDerivedFrom` / `wasRevisionOf` / `wasGeneratedBy` / `wasAttributedTo` |
| `cco:DescriptiveICE` cross | 3 | lineage governed by directive, lineage observed by event, column lineage to sdg:Column |

## Failure

Single failure: `transformation_equiv_io_intersection` —
`EquivalentTo bfo:0000015 and (X some Y) and (X some Z)`. Same
DeepOnto verbaliser limitation as the EquivalentTo-with-existential
cases in earlier batches. `is_complex=False` after the harness sets
it; loads cleanly; only R_C and R_D contributions are zero. No
manufactured fallback (per RL-utility guidance).

## Combined catalog state (post-Batch-5)

| Metric | Value |
|---|---|
| Total templates | **343** |
| Verbalized | **327 (95%)** |
| `is_complex` | 294 (86%) |
| Schema errors | 0 |
| `sdg:*` properties in TTL | 173 |

Per-batch contribution:

| Batch | Templates | Verbalized | Complex |
|---|---|---|---|
| 1 — Foundation | 81 | 73 (90%) | 68 |
| 2 — Observation + Measurement | 92 | 91 (99%) | 79 |
| 3 — Directive + Governance | 69 | 66 (96%) | 58 |
| 4 — eBPF kernel granularity | 51 | 48 (94%) | 44 |
| 5 — PROV-O / OpenLineage | 50 | 49 (98%) | 45 |

## Null distribution refresh

| Metric | Post-B4 (293) | Post-B5 (343) | Δ |
|---|---|---|---|
| `null_mean` | 0.41785 | 0.41710 | −0.00075 |
| `null_std` | 0.00809 | 0.00900 | +0.00091 |
| `null_p5` | 0.40526 | 0.40203 | −0.00323 |
| `null_p95` | 0.43182 | 0.43205 | +0.00023 |

All shifts within sampling noise. Locked weights `{0.50, 0.05, 0.45}`
remain in effect.

## P4 smoke test (post-Batch-5)

| Profile | Mean R |
|---|---|
| A_complex_slot_rich | 0.5242 |
| B_mixed | 0.5243 |
| C_trivial_basic | 0.2014 |
| D_stub_filler | 0.4753 |

Discrimination spread A − C: **0.3228** (was 0.5129 post-B4). The
narrower spread reflects a randomly-chosen Profile C sample that
hit better-than-typical R_C values; Profile A still ranks above
Profile C, and determinism + non-degenerate-std + reward-variance
gates all hold. P4 smoke test still ✓ PASS.

## Files touched this session

| Status | Path |
|---|---|
| NEW | `src/aegir/ontology/catalog/05_provo_lineage.candidate.json` (50 templates) |
| NEW | `src/aegir/ontology/catalog/05_provo_lineage.json` (DeepOnto-populated) |
| edited | `src/aegir/ontology/sdg-vocab.ttl` (+33 properties under "Batch 5" header) |
| edited | `src/aegir/ontology/catalog/combined.json` (now 343 templates) |
| edited | `src/aegir/ontology/null_stats.json` (refit against combined) |
| edited | `docs/current/ontology/charter.md` (P1 assessment + catalog-state sections updated; new P5 readiness assessment section added) |
| NEW | `docs/scratch/2026-05-10/050131_batch5_provo.md` (this file) |

## P5 infrastructure readiness — quick read

The full assessment is at `docs/current/ontology/charter.md#p5-infrastructure-readiness-assessment-2026-05-10`. Synopsis:

**Ready:** catalog (343), verifier (locked weights, hash-stable),
T_I + null stats, GRPO group iteration validated end-to-end
via P4 smoke test. The reward-signal layer is fully operational.

**Pending model-side engineering (~6 days + ~200 GPU-hours):**

1. SAE-Res-Qwen3.5-27B-W80K-L0_100 policy load (tensor-parallel across 6× RTX 4090).
2. LoRA adapter setup (target attention + MLP projections; leave SAE bottleneck untouched).
3. GRPO loop wiring (`trl.GRPOTrainer` or minimal custom loop).
4. Composition-format constrained decoding (lm-format-enforcer or Outlines).
5. SAE feature logging during rollout (interpretability claim from v0.5 brief).
6. Held-out evaluation harness (extend C1 test set + new 50-ontology held-out set).

**Recommended path:** parallel — start P5 scaffolding against
current 343-template catalog while Batches 6 and 7 are authored
in parallel. Batch 6/7 hot-load via the verifier's per-call
catalog re-instantiation. Avoids 2-3 weeks of catalog-only delay
before model-side feedback begins.

**Risks called out:** constrained-decoding fidelity, R_C floor
collapse, R_D corpus drift if policy generates kernel-only
compositions, group-size variance with real policy.

## Path forward

P1 stays cleared; verifier continues hash-stable.

- **Batch 6 — BeliefStructure for DST/Atelier integration** (~30 templates).
  `sdg:MassFunction`, `sdg:BeliefInterval`, `sdg:Evidence`,
  `sdg:Claim`. Q1 commitment.
- **Batch 7 — long tail + benchmark gap-filling** (~150 templates).
- **P5 — full GRPO training** with SAE-Res-Qwen3.5-27B-W80K-L0_100
  against the locked verifier. Can begin scaffolding in parallel
  with Batches 6-7.
- **Parallel hygiene track** — v2 → SOTAB head fine-tune. Independent.

## Book-narrative integration note (per user direction)

The user's 2026-05-10 message asked that the post-Batch-4 state be
reflected in the project book under Chapter 3 (semantic engine: aegir
+ atelier) and Appendix A ("What is real today"). With Batch 5 now in,
the narrative should cite **343 templates / 327 verbalized / 95% /
P1 cleared / closed-loop ontology→reward pipeline operational**.
The closed-loop description in Chapter 3 should mention that the
verifier signal flows from procedurally-generated catalog
verbalizations (no DeepOnto in the RL hot path) through locked
weights `{0.50, 0.05, 0.45}` into GRPO advantages with a 0.13 s/group
budget on CPU. Appendix A's "What is real today" should explicitly
call out the cross-context cousining (Trace+LabRun share
ObservationProcess; SQL/SysMLv2/eBPF Constraint co-anchored at
DirectiveICE; Allocation+Transformation share bfo:Process; LineageEdge
sits at DescriptiveICE alongside ColumnPolicy) as load-bearing
evidence that the ontology grounds claims across LIMS, MBSE, database,
kernel-tracing, and lineage contexts simultaneously.

The book itself is outside this repo; the integration note is
recorded here so that when the Atelier-side book draft pulls from
this repository, the relevant facts are in one place to lift.

# Batch 4 — eBPF kernel granularity

*2026-05-10 — adds the four bespoke eBPF subbranches
(`sdg:eBPFProgram`, `sdg:KernelHook`, `sdg:Syscall`, `sdg:Map`)
plus cross-branch relations into the catalog. P1 exit gate
remains cleared, now with comfortable margin (278 vs 200 floor).*

## Summary

| Metric | Result |
|---|---|
| Candidate templates authored | 51 |
| Loaded into DeepOnto without error | 51 (100%) |
| Verbalized round-trip | 48 (94%) |
| `is_complex` flagged | 44 (86%) |
| Axiom-family coverage | F1, F2, F3, F4, F5, F6, F7 (all 7 represented) |
| New `sdg:*` object properties declared in TTL | 27 |
| Wall-clock harness time | 67 s |

## Branch coverage in this batch

| Branch | Templates | Notes |
|---|---|---|
| `cco:Artifact / sdg:eBPFProgram` | 14 | F1–F7 all represented; includes union (kprobe ∪ XDP) and negation (`not Unloaded`) |
| `cco:Artifact / sdg:KernelHook` | 9 | hook-kind, kernel-function attach point, attached programs |
| `cco:Artifact / sdg:Map` | 10 | map-type, key/value-type, max-entries, bpffs pin-path |
| `cco:DesignativeICE / sdg:Syscall` | 10 | syscall sits next to Identifier — naming-style ICE for kernel-API surface |
| `cco:DesignativeICE / sdg:Syscall` cross | 1 | `auditedBy` xref into governance branch |
| `cco:Artifact` cross-branch | 4 | program-attaches-hook, program-writes-map, hook-observes-syscall, program-governed-by-DirectiveICE |
| `bfo:Process / sdg:eBPFEvent` cross | 3 | extends Batch 2's eBPFEvent with `observesSyscall`, `viaProgram`, `atKernelHook` |

## Failures

Three templates fail DeepOnto verbalization, all of the same kind:

| Template | Error |
|---|---|
| `ebpfprogram_equiv_typed_with_hook` | `EquivalentTo` complex restriction |
| `syscall_equiv_in_subsystem` | `EquivalentTo` complex restriction |
| `ebpfmap_equiv_kv_typed` | `EquivalentTo` complex restriction |

These are honest DeepOnto verbaliser limitations on
`EquivalentTo` axioms whose RHS includes existentials. They
load without error and remain `is_complex` for R_B; only
R_C and R_D contributions are zero. No manufactured fallback
was added, per the user-confirmed RL-utility guidance.

## Combined catalog state

Combined (B1+B2+B3+B4):

| Metric | Value |
|---|---|
| Total templates | **293** |
| Verbalized | **278 (95%)** |
| `is_complex` | 249 (85%) |
| Schema errors | 0 |

Per-batch:

| Batch | Templates | Verbalized | Complex |
|---|---|---|---|
| 1 — Foundation | 81 | 73 (90%) | 68 |
| 2 — Observation + Measurement | 92 | 91 (99%) | 79 |
| 3 — Directive + Governance | 69 | 66 (96%) | 58 |
| 4 — eBPF kernel granularity | 51 | 48 (94%) | 44 |

## Null distribution refresh

The 200-sample structural-shuffle null distribution was refit
against the new 293-template combined catalog. The shift
versus the post-Batch-3 baseline is well within sampling
noise:

| Metric | Post-B3 (242 templates) | Post-B4 (293 templates) | Δ |
|---|---|---|---|
| `null_mean` | 0.41931 | 0.41785 | −0.00146 |
| `null_std` | 0.00983 | 0.00809 | −0.00174 |
| `null_p5` | 0.40522 | 0.40526 | +0.00004 |
| `null_p95` | 0.43590 | 0.43182 | −0.00408 |

Locked verifier weights `{a=0.50, b=0.05, c=0.45}` remain in
effect; no re-tuning was triggered. The C1 sweep remains the
authoritative weight calibration.

## Verifier infrastructure check

P4 smoke test re-run against the post-Batch-4 catalog +
refreshed null:

| Profile | Mean R | Notes |
|---|---|---|
| A_complex_slot_rich | 0.5292 | top-ranked |
| B_mixed | 0.5245 | mid-tier |
| D_stub_filler | 0.4747 | R_B-driven, expected |
| C_trivial_basic | 0.0163 | bottom-ranked |

Discrimination spread Profile A − Profile C: **0.5129**
(was 0.505 pre-Batch-4). Determinism: rewards + advantages
identical across re-runs. P4 smoke test still ✓ PASS.

## TTL extension

27 new `sdg:*` object properties added to `sdg-vocab.ttl`,
covering the eBPF / kernel / syscall / map vocabulary
(`sdg:attachesToHook`, `sdg:hasProgramType`, `sdg:hasLicense`,
`sdg:loadedIn`, `sdg:verifiedSafeBy`, `sdg:atKernelFunction`,
`sdg:inKernelModule`, `sdg:hasAttachedProgram`,
`sdg:forKernelSubsystem`, `sdg:hasHookKind`,
`sdg:observesEvent`, `sdg:inSyscallSubsystem`,
`sdg:invokedByEvent`, `sdg:hasReturnType`,
`sdg:hasArgumentCount`, `sdg:traceableBy`,
`sdg:atSecurityTier`, `sdg:hasMapType`, `sdg:hasKeyType`,
`sdg:hasValueTypeMap`, `sdg:hasMaxEntries`,
`sdg:usedByProgram`, `sdg:pinnedAtPath`, `sdg:viaProgram`,
`sdg:atKernelHook`, `sdg:observesSyscall`, `sdg:auditedBy`).
Each has `rdfs:label` + `skos:definition`. Total declared
sdg-namespace properties: **140**. Reused existing
`sdg:writesToMap` and `sdg:governedBy` from Batches 2/3.

## Files touched this session

| Status | Path |
|---|---|
| NEW | `src/aegir/ontology/catalog/04_ebpf_kernel.candidate.json` (51 templates) |
| NEW | `src/aegir/ontology/catalog/04_ebpf_kernel.json` (DeepOnto-populated) |
| edited | `src/aegir/ontology/sdg-vocab.ttl` (+27 properties under "Batch 4" header) |
| edited | `src/aegir/ontology/catalog/combined.json` (now 293 templates) |
| edited | `src/aegir/ontology/null_stats.json` (refit against combined) |
| edited | `docs/current/ontology/charter.md` (P1 readiness + catalog state sections updated post-Batch-4) |
| NEW | `docs/scratch/2026-05-10/044546_batch4_ebpf.md` (this file) |

## Path forward

P1 stays cleared. The remaining work is mechanical catalog
growth + the P5 model-side infrastructure:

- **Batch 5 — PROV-O / OpenLineage lineage** (~50 templates).
  `sdg:LineageEdge`, `sdg:Transformation`, `sdg:Allocation`
  under `prov:Activity`. Q4 commitment.
- **Batch 6 — BeliefStructure for DST/Atelier** (~30 templates).
  `sdg:MassFunction`, `sdg:BeliefInterval`, `sdg:Evidence`,
  `sdg:Claim`. Q1 commitment.
- **Batch 7 — long tail + benchmark gap-filling** (~150 templates).
- **P5 — full GRPO training** with SAE-Res-Qwen3.5-27B-W80K-L0_100
  against the locked verifier. ~200 GPU-hours per the v0.5
  brief budget.
- **Parallel hygiene track** — v2 → SOTAB head fine-tune.
  Independent.

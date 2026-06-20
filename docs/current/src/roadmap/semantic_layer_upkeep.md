# Semantic-Layer-Upkeep — the local-first quality loop

**Status: SPEC (2026-06-19, RH).** The procedure for keeping the **semantic layer** (ontology → DDL →
views + verbalizations) *valuable enough to spend paid-API budget scaling out*. The last cycle established
**structure** (RI-true tables/views, SKOS-native names) but the *semantic* content is thin — and semantic
content is what makes the corpus worth pretraining on. This spec adds an **embedded-view semantic-quality
gate** and an upkeep loop we run **entirely on local resources** before any paid scale-out.

## The problem (audited 2026-06-19)
- **Verbalizations are low-entropy.** Baseline (`scripts/audit_verbalization_entropy.py`): 522 templates,
  **60 distinct syntactic frames**, **top-5 frames = 69%** (`§ is a · that · §` alone = 167). "X is a Y"
  monotony. Root cause is *our under-use of DeepOnto* (it is **not** a black box): we call
  `OntologyVerbaliser` with defaults and take only its single `.verbal` string — never its config
  (`add_quantifier_word`, `vocab`), never its `OntologySyntaxParser`→`RangeNode` **parse tree**, never the
  relational verbalisers (`object_property_domain/range/assertion`).
- **Cell values ~67% placeholders** (`Process 01`) + a `start_time > end_time` bug; ~33% (enums) are real.
- **Column vocabularies "canned"** — all BFO anchors below SchemaPile p10 (de-canning), because same-anchor
  tables inherit an identical attribute set.

## The three quality dimensions (metric · floor · lever)
| Dimension | Metric | Floor (provisional, ratchet up) | Lever (local) |
|---|---|---|---|
| **Verbalization diversity** | skeleton-frame entropy + top-5 share + relational share (`audit_verbalization_entropy.py`) | top-5 share ↓, frame entropy ↑ vs baseline | DeepOnto parse-tree re-render (config + relational verbalisers) → diverse set; **local-LLM** elaboration |
| **Value semantics** | placeholder-ratio + domain-term fraction | placeholder-ratio ↓ | richer enums (sdg-vocab), date-bug fix; **local-LLM**-seeded domain values |
| **Column-name diversity** | de-canning `distinct_ratio` vs SchemaPile (`check_decanning_entropy.py`) | ≥ provisional floor (ratchet toward p25→p50) | per-template stratified anchor-attributes |

The **embedded-view semantic-quality gate** (`scripts/semantic_layer_gate.py`) composes the three into one
per-dimension pass/fail, pre-registered in `EVIDENCE.md`. **A gate is a floor to clear on the way — not the
objective** (see Non-goals).

## Provisional scaffolding / NON-GOALS (load-bearing — RH 2026-06-19)
The simplifications below are **expedient scaffolding to get an early result over the line — they are NOT
goals**, and must never be codified as design targets (the "illustrative, not definitive" discipline; cf.
the Provenance DAG). See memory `provisional_scaffolding_not_goals`.

- **"entity columns are never FKs ⇒ LLM-seeded values are RI-safe"** — holds only for *today's* simple
  schemas. The real product has entity columns that **are** foreign keys in dense webs.
- **one-FK-per-table** (`cross_family_fks` takes `refs[0]`), **slot-derived** structure, **RI=1.0 by
  construction** over simple tables — current floors, not the shape of the target.
- **`distinct_ratio` floor, placeholder/deterministic values, realization-CPA firing only on object-property
  templates** — proxies/guards/current-scope, not the destination.

**North star:** the true final data product carries **significant real-world relational complexity** —
dense many-to-many relations, FK-bearing entity columns, complex multi-table schemas, and domain-real values
and prose. The upkeep loop's job is to *advance toward* that; when the work matures, the scaffolding is
**retired, not enshrined**.

## Local LLM substrate — the Aegir capability/gRPC engine
LLM-using levers run on a **local** capability/gRPC engine (mirroring Gaius; `src/aegir/engine/`), serving
**Qwen 3.6+** via vLLM. **Strict layering:** the engine is the **sole** vLLM client and owns the
**capability→model mapping**; **workloads connect only to the gRPC engine** (`Complete`), never to vLLM,
never handed an endpoint URL. Federation with Gaius's engine is the roadmap. This is *normal local
overhead*, not a gate.

**Thinking-trace retention.** Qwen 3.6 reasons verbosely, and the reasoning trace is a **corpus value-add**
(cf. Cerebras GLM reasoning-trace retention in published datasets) — so the engine **retains** it rather
than suppressing it. `CompleteResponse` carries `reasoning_content` (the separated trace, when a
model/parser splits it cleanly) alongside `text` and `finish_reason`; for a checkpoint that embeds its
trace inline with no parseable delimiter, the trace is retained within `text`. The engine is sized for long
traces *without OOM*: `max_model_len × max_num_seqs` is held at the proven-safe KV footprint (e.g. 16384×8 ≡
8192×16), and token budgets are generous (the workload accepts the wait). Use `client.complete_detailed()`
to capture the trace for the corpus; `complete()` returns just the answer text.

## Resource principle / sequencing
**Local GPU, local LLM, and HermiT/JVM are normal programming overhead — used freely. The only gate is a
paid remote API.** So the entire upkeep loop runs + is gated locally:

```
iterate (deterministic + local-LLM levers) → re-run the semantic-quality gate → confirm in the lineup
   → repeat until gate green → ONLY THEN scale out the end-stage corpus with paid Grok/Cerebras
```

## Confirmation surface — the lineup
The lineup **Schema lens** is where a curator *sees* the layer advancing: `build.py::project_relational`
surfaces sample rows (from `base_rows.parquet`), the verbalization, and per-table/anchor quality badges.
`just kb-build` rebuilds; browse `/lineup`.

## Where this sits
Inserted **before the paid corpus scale-out** that feeds M2/M3. M1 (H-Net isolation, local GPU) is
independent and may proceed in parallel. Full implementation plan: `~/.claude/plans/unified-noodling-flurry.md`.

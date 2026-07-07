# Phase Gate — The Inverted Topic Layer (R1 topic-layer build)

**Date:** 2026-07-07 · **Decision: PASS** · **Commits:** `70e3b23` → `05c1503` → `3835d3c`
(increments 1–3) · `edc261b` (the authorless (f)-iteration) · `a9d7ced`, `ff6a45c`, `a593382`
(the authored half + M8) · `bdadbc0` (τ\* + M7) · `407ce0e` (punchlist) · `cb8b54c` (lineup
items). Ratified design: RH 2026-07-07, rulings (a)–(h).

This gate certifies the **inverted topic layer** as the live topic instrument: every FinePDFs
passage window (item) maps to **at most one ontology-grounded topic**, by late-interaction
MaxSim against the live catalog's term anchors, under a **pre-registered, null-calibrated
unambiguity gate**, with every association **content-addressed to the collection state that
adjudicated it** — and the whole lineage walkable in the lineup. It replaces the corpus-fitted
(BERTopic-era) topic model as conceived for R1.

## The thesis it certifies

> Topics are not fitted to the corpus; the corpus is measured against the ontology. A topic
> **≡** a concept anchor in the registry (first-order fungible — the qdrant collection IS the
> topic registry; no fitted topic-model artifact). An input window aligns to **one specific
> topic or none** — ambiguous mass is the **error signal**, and the **lexicon is the
> parameter**: when input fails to align, the topics iterate (definitional rigor), never the
> input. This inverts the LDA direction while keeping every definitional artifact legible,
> editable, and reasoner-adjacent.

## Methods (the final version — what an implementer inherits)

### 1. The registry (`topic_layer.build_term_registry` → `sdg_topics`)

- **Anchors** = the 29 rich SKOS domains (the aperture overlay) ∪ **all 433 live catalog
  terms**, each embedded as ColBERT multivectors (colbertv2.0, 128-d/token, qdrant-native
  MAX_SIM). Terms nest under their SKOS domain via `ancestor_codes`, so the hierarchical gate
  treats domain↔member-term as one lineage while **sibling terms compete** — that competition
  is the rigor signal.
- **Anchor text** = `term_anchor_text()` — the *single shared composition* (the membrane
  judges exactly what MaxSim embeds): prefLabel · slot/alt labels · verbalization frames ·
  **the axiom's referenced vocabulary resolved to labels/definitions** (CCO/FHIR via the
  grounding-anchors index; a static gloss map for BFO numerics) · grounding `source_span` ·
  scope. Everything *declared* is exposed; nothing is invented outside the membranes.
- **Self-declared granularity:** an anchor's multivector length IS its token count — the
  registry announces its own granularity (`anchor_tokens_p50`), which sizes the items.
- **Content-addressing:** `collection_state()` digests the sorted anchor identities
  `(id, code, iri, label, vector_sha)` → a 16-hex `collection_sha`; annotations (token
  counts) ride outside the digest. Lens identity fields (`encoder`, `projection`) stamp every
  state and record (ladder rung L1) — accumulated associations remain valid supervision
  across future lens generations.

### 2. Items — anchor-proportional sliding windows (`windows` / `window_plan`)

- The FedWiki item granularity, realized: documents slice into token windows **proportional
  to the anchors' median token count** (`window = ratio × anchor_p50`; late interaction is
  well-conditioned when both sides speak at comparable length), cut with the **same
  tokenizer MaxSim sees**, char-span-faithful to the original text, tail-covered. This also
  fixed a real defect: whole-document classification silently truncated at the encoder's
  512-token cap.
- A/B discipline: runs against an *evolved* registry **pin** the window
  (`--window-tokens`) so per-item span joins hold.

### 3. Adjudication — one item, one topic, or none (`associate_items` / `_adjudicate`)

- qdrant-native MaxSim; **hierarchical unambiguity** gates assignment: `rel_margin_h` =
  margin over the **nearest non-ancestor competitor** (a parent/child near-miss is not
  ambiguity — the subtree is the unambiguous assignment). Flat margin is recorded alongside.
- The `Association` record is the lineage:
  `(passage_hash, doc_hash, span, topic, score, margins, competitor, τ, collection@sha,
  encoder, projection)` — assigned or not (the ambiguous mass must be inspectable; it is
  the rigor loop's food). Stores are **sha-keyed**: a re-run against an evolved registry
  lands *beside*, never over, prior adjudications.

### 4. The definitional-rigor loop — both halves, membrane-gated

- **Authorless first (increment 4):** expose the declared-but-unseen surface (frames, axiom
  vocabulary, spans). Measured effect: term anchors p50 48→78 tokens; alignment +42%
  relative; assigned mass moved domain→term; the thin-intermediate smell case collapsed.
  The honest first move is exposure, not authorship.
- **Authored half (4b):** the closed loop — engine proposes `{alt_labels, scope_note,
  definition}` per term (gRPC, **guided-JSON** structured output), the **annotation
  membrane disposes with a reason**, rejects re-prompt carrying that reason. Gates:
  **M1** shape · **M2** token bounds vs the registry's granularity · **M3** no smuggled
  ontological claims (annotations never assert taxonomy; HermiT has no stake, so the
  membranes are the whole gate) · **M4** vocabulary grounding (the term's declared
  vocabulary + gloss resolution + top-k grounding-anchor retrieval) · **M5** clean-room
  tripwire · **M6** discrimination, including the **self-retrieval margin gate** (the
  composed surface must retrieve its own anchor at rank-1 with margin over the nearest
  non-ancestor — its failure message names the sibling to differentiate from, the
  highest-value re-prompt feedback) · **M8** *positive voice*.
- **M8 — the methodological finding (RH):** *definition-by-negation is an embedding
  anti-pattern.* A similarity filter has no negation operator: "do not use for general
  travel acts" **injects** the excluded content as attractor mass, and "distinct from
  ⟨sibling⟩" injects the sibling's vocabulary into this anchor. Measured: 190/194
  first-pass surfaces carried negation (prompt-induced); one anti-pattern explained both
  the basin explosion (3→38) and the margin compression. **Durable rule: anchor surfaces
  are positive-voice only — discriminate by specific positive vocabulary (participants,
  artifacts, settings, instruments), never by exclusion or contrast.**
- Authored surfaces live on `CatalogTemplate` (`alt_labels`, `scope_note`, elaborated
  `verbal_templates`) with a provenance `skos_authoring` record (date, model, membranes,
  registry sha, dispositions) — catalog.json stays the single source of truth.

### 5. Calibration — pre-registered, report-not-tune (`derive_tau`)

- **τ\* = the (1−α) quantile of `rel_margin_h` under the shuffled-window null** (the
  reference items, token-shuffled at fixed seed 44641 — vocabulary and length preserved,
  topical coherence destroyed), α = 0.05, method registered *before* computing
  (`docs/scratch/2026-07-07/165051_…`). Result: **τ\* = 0.1065** — the inherited 0.10 was
  already right; rate recovery via a lower gate would admit vocabulary leakage. Sobering
  corollary: the null median (0.0253) nearly meets the real median (0.0289) — the bulk of
  harvest windows are near-null connective prose; only the tail is genuinely topical. The
  aligned rate is a corpus-composition fact, not a registry deficiency. τ\* is
  lens-identity-scoped; new lens ⇒ re-derive.

### 6. The M7 basin gate (`basin_calibration`)

- Registry changes are gated on the **input side**: between two pinned runs over the same
  items, a topic flags when its assigned mass explodes (`new ≥ 5 ∧ new ≥ 3× base`). The
  self-retrieval gate (anchor-vs-anchor) is necessary but **not sufficient** — an authored
  surface can pass M1–M6 yet explode its attraction basin.
- Offenders carry their evidence: `n_docs`, dominant document, its share, and a reading —
  **single-source concentration** (N items from one long document; inspect the document,
  likely legitimate — both live flags resolved this way: a municipal traffic ordinance, a
  religious text) vs **corpus-wide attraction** (basin suspect; inspect the anchor). The
  gate flags; disposition (strip / re-author / accept-with-justification) is the loop's and
  is recorded in provenance. **The gate never silently mutates.**

### 7. Lineage in the lineup

- Assigned associations project as **item notes** (trunk/scratch): the window's original
  text, its adjudication, its registry pin — with the topic as a live term wikilink. Term
  notes are hubs (**Aligned items**); `item/index` is the trailhead (Scratch → Content →
  Items × Topics). The unassigned mass stays report-side by design.

## Gate evidence (verified)

| Criterion | Result |
|---|---|
| Registry: live-catalog topics, content-addressed | **462 anchors** (29 domains + 433 terms) @ `0e88f134…`, lens identity stamped ✅ |
| Full-store adjudication (453 docs, τ\*, pinned 170) | **15,976 items · 1,629 aligned (10.2%) · 175/462 topics hit** ✅ |
| Items bind to individual terms (not just domains) | top terms carry 20–380 items each; domain roll-up absorbs 10 (was 64 pre-rigor) ✅ |
| Four-state A/B (same 1,635 items) | declared 13.9%/34 → neg-authored 11.1%/28 (taxi 38) → **positive-authored 9.0%/36** — rate traded for **resolution** ✅ |
| Authoring convergence (closed loop) | 195/201, then 181/197 positive-voice (M8 fired 5×), then 7/16 stubborn (widened M4); **9 honest holdouts** stay thin by design ✅ |
| τ\* pre-registered + derived | 0.1065 (null p95), inherited 0.10 validated; report-not-tune held ✅ |
| M7 at scale (full store, pre/post authoring) | **PASS, zero flags**; both historical flags resolved by doc-concentration evidence ✅ |
| Lineage walkable | 1,403 item notes + term hubs + trailhead, serving via `?root=` scoped resolution ✅ |
| BERTopic-era retired | `topic_alignment.py`/`build_topic_model.py` deprecated → `topic_layer` + `congruence`; live flow confirmed clean ✅ |

## Significance / what this unblocks

- **R1's topic layer exists** — coverage regen no longer means a BERTopic re-fit; the
  corpus census over ontology-grounded topics is live and re-runnable per window.
- **The (f)-loop is a working instrument**: unaligned input indicts topics; the lexicon
  improves through membranes that return reasons; the improvement is measured, gated, and
  provenance-recorded. This is the agent-mediated-feedback doctrine operating on retrieval
  surfaces.
- **Supervision for the latent-lens ladder accumulates by operation** (design:
  `docs/scratch/2026-07-07/152519_latent_lens_design.md`): every association record is a
  training pair for the anchor-projection head (L3); lens identity fields are already on
  the records (L1). L2 — the latent-prediction auxiliary for Path-B pretraining — is the
  next major move.
- **Anti-mimicry-compatible by construction:** topics are named, legible, editable text —
  never opaque vectors — so the definitional layer stays inside the verification membrane
  the programme depends on.

## Known issues / deferred (none gate-blocking)

- **9 holdout intermediates** (maximally generic: `serviceprovider`, `regulatoryauthority`…)
  are M6-blocked — correctly: terms that abstract cannot own discriminative surfaces. They
  stay thin on the worklist rather than fattened into attractors.
- **Intermediates carry `domain: null`** → no `ancestor_codes` → they compete flat at the
  root and never receive domain-lineage forgiveness. Assigning domains is a
  semantics-changing move (not annotation) — a candidate for the next authoring increment
  under full membranes.
- The passage store remains whole-document at harvest (the stream cursor is the outer
  window); item windowing is projection-side. Persisting items as first-class harvest
  artifacts is open.
- M7 dispositions are manual by design; only the evidence is automated.

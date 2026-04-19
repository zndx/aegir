# Latent-Guided Training of a Tokenizer-Free H-Net RWKV Model (Aegir)
### v3 — Phased, conditional, de-risked

## Objective

Train the H-Net/RWKV-7 Aegir architecture from scratch on raw bytes, deriving
a tokenizer-free vocabulary directly from corpus statistics. Measure it
against a clean baseline, then evaluate two increasingly ambitious knowledge-
transfer paths — weight-space fusion via Mergekit, and activation-space
latent supervision from Nemotron-3 Nano — each gated on the previous phase's
outcome. Reserve expensive GLM-4-7 reasoning traces for a fixed offline
replay corpus.

The novel claim is *conditional*: whichever phase produces the larger
measured uplift becomes the headline. If Phase-1.5 already gets us there,
the Nano path remains a research option rather than a commitment.

## Phase structure (committed plan)

| Phase | Scope | Expected cost | Headline claim |
|---|---|---|---|
| **Phase 1 — Aegir-only** | Train single-task Aegir on SOTAB CTA / CPA (Schema.org + DBpedia) and GitTables-DBpedia with the validated SSD kernel. | ~1 week, mostly unattended background runs. | Baseline numbers per task. Leaderboard row each. Known: this is where MLFlow/W&B would live, and we're deliberately replacing that with in-process JSON sidecars + Bokeh snapshots. |
| **Phase 1.5 — Aegir specialist fusion via Mergekit** | Apply `mergekit-pytorch` (task-arithmetic, TIES, evolutionary search) to fuse Phase-1 single-task checkpoints into a multi-task Aegir. Test whether RWKV + H-Net checkpoints merge cleanly at all — an *open empirical question*: the recurrent-state parameters and H-Net dynamic-chunking weights have non-trivial geometric structure that weight-space linear interpolation may or may not preserve. | ~3 days once Phase 1 checkpoints land. | "Multi-task Aegir" entry on the leaderboard that beats each single-task baseline on out-of-task transfer. If *not*, that's itself publishable — negative result bounds the design space. |
| **Phase 2 — Aegir + cross-architecture distillation** (conditional) | Run only if Phase 1.5 falls short by more than ~3 points of the best hypothetical Nano-supervised number (or if Mergekit Phase 1.5 produces incoherent checkpoints). Begin with a **tokensurgeon spike** (below). | 2-3 weeks. | Depends on which sub-path the spike unlocks. |

Each phase produces an independent, citable measurement. The tripartite
leaderboard (solo / merged / aligned) is the paper structure.

## Phase 2, Step 0 — Tokensurgeon spike (half-day, high information value)

Before committing to LatentMAS-style span alignment, run the cheaper
experiment:

1. Download Nemotron-3 Nano weights (gated; verify license first).
2. Apply `mergekit-tokensurgeon` to transplant Aegir's byte-level "vocabulary"
   onto Nano's embedding matrix. Tokensurgeon is *explicitly* designed for
   cross-tokenizer knowledge distillation — this is its stated use case.
3. Sanity-check: feed the transplanted Nano the same 1000 raw-byte sequences
   Aegir sees. Measure the resulting output quality on held-out perplexity.
4. If transplantation produces a coherent byte-level Nano → use it directly as
   the teacher signal. Aegir and transplanted-Nano consume identical byte
   inputs → positional alignment is *trivial* (same positions), `W_a` may not
   be needed at all, and we can do standard distillation.

If tokensurgeon dissolves the dual-tokenizer problem, the rest of v2's
architecture (span-based alignment, W_a refit policy) is replaced by plain
distillation loss on matched positions. Massive simplification.

**If tokensurgeon does not work** (likely scenarios: Nano's hybrid
Mamba-Transformer architecture isn't supported, or the transplant destroys
too much of Nano's learned geometry), fall back to v2's span-based latent
alignment path.

## Phase 2, Step 1 — The 1-day de-risker (only if tokensurgeon fails)

Take 1,000 raw-text samples. Run both tokenizers (BPE Nano, byte Aegir).
Compute W_a via ridge regression on 800 paired samples under each candidate
alignment strategy:

| Alignment | Works when | Cost |
|---|---|---|
| Char-boundary (Nano token i → byte position B[i]) | always | wastes signal for non-boundary bytes |
| Semantic-boundary (H-Net dynamic chunks ↔ Nano tokens) | H-Net has learned chunks | chicken-and-egg at cold start |
| Pooled (mean-pool Nano hidden across Aegir-chunk window) | always | drops positional detail |

On held-out 200 pairs, measure R²: does projected(Nano_hidden) predict
baseline-Aegir's hidden at the paired position better than random?

- R² > ~0.2 with any strategy → hypothesis survives; proceed to Phase 2
  full training.
- R² ~0 across strategies → geometry is the bottleneck. Pivot to output-logit
  or attention-map distillation (TinyBERT-style) before committing.

## Phase 2 — Full training (only if de-risker passes)

### Span-Based Latent Alignment

Each Nano BPE token maps back to its byte span in the raw text. Hidden
states aggregate via mean-pooling (or attention-weighted pooling) per span,
project into Aegir's embedding space via `W_a`.

### Deferred-Periodic Refit Policy for W_a

- **Defer** initial fit until Aegir has completed 10% of total pretraining
  steps (embeddings have begun to stabilize).
- **Refit** every 5% of training steps thereafter.

Stabilizes against the noise-dominated cold-start problem; keeps the
alignment matrix relevant to Aegir's evolving representation space.

### Live Inner Training Loop

1. Nano forward pass on raw text → span-aggregate hidden states → project
   via current `W_a`.
2. Aegir forward pass ingests the identical raw bytes through its learned
   H-net embeddings; receives aligned Nano latents via additive injection
   or cross-attention fusion at matching byte / chunk positions.
3. GLM-4-7 traces replayed only periodically, outside the Nano hot path,
   as a stronger distillation signal for harder reasoning samples.

### Loss Structure

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \alpha(t)\,\mathcal{L}_{\text{latent}} + \beta\,\mathcal{L}_{\text{GLM-replay}}$$

- $\mathcal{L}_{\text{main}}$: next-embedding (or learned-token) prediction
  on Aegir's native representation.
- $\mathcal{L}_{\text{latent}}$: **cosine similarity** between Aegir's hidden
  states and projected Nano latents (basis-invariant; raw MSE has the
  coordinate-system problem).
- $\alpha(t)$: curriculum schedule. $\alpha_0 = 1.0$ early, linearly decays
  to ~0.1 by the final 20% of training. Prevents H-Net chunking from
  degenerating toward Nano-like BPE boundaries.
- $\beta$: small constant (e.g. 0.2), applied only on GLM-replay minibatches.

## Evaluation

### Falsifiable claims

**Phase 1 vs 1.5**: "Mergekit-fused multi-task Aegir matches or beats the
average of single-task Aegirs on their respective tasks, and beats each
individually on at least one out-of-task benchmark."

**Phase 2**: "Aegir trained with Nano latent alignment beats Aegir-only and
Mergekit-fused Aegir on held-out corpus perplexity AND on at least one of
{CTA, CPA, DED} downstream benchmarks, by a margin that justifies the 10×
training budget."

### Primary metrics

- Held-out corpus perplexity (raw-bytes, matched split).
- SOTAB-CTA micro + macro F1 (target: REVEAL's 0.815 micro as the
  public benchmark).
- SOTAB-CPA F1.
- GitTables-DBpedia F1.
- Cross-table DED B-cubed F1 (once that task registry entry lands).

### Baselines

1. Aegir-only (no Nano, no Mergekit).
2. Aegir + Mergekit fusion.
3. REVEAL (external ML baseline, re-implemented).
4. Published external LLM numbers on the same splits (GLM-4-7 / Nemotron
   Nano one-shot), for context — these are *not* the same training budget
   as Aegir, so the comparison is framed accordingly.

## Deliberately out-of-scope

- **Weight-space Aegir↔Nano merges.** Architecturally heterogeneous; doesn't
  work by construction. Mergekit only fuses compatible shapes.
- **KV-cache transfer from Nano to Aegir.** Attracting but expensive.
  Reserve for a Phase 3 investigation after Phase 2 lands.
- **Training Nano itself.** Nano stays frozen throughout. We are a consumer
  of its representation space, not a modifier of it.
- **MLFlow / W&B integration.** Leaderboard gateway is the observability
  surface — in-repo JSON sidecars + Bokeh plots, no tracking daemon.

## Why this structure

Three design pressures:

1. **Every claim should be independently measurable.** Each phase has its
   own leaderboard row and its own falsifiable claim. If Phase 2 doesn't
   ship, Phase 1 + 1.5 is still a paper.
2. **Do the cheap experiment before the expensive one.** Tokensurgeon
   costs ~4 hours to try; span alignment + W_a policy is weeks of careful
   engineering. If tokensurgeon works, the Phase 2 architecture is a
   simpler system.
3. **Keep the native Aegir geometry intact.** The curriculum schedule
   ($\alpha(t)$), the deferred W_a refit, and the Phase 1 → 1.5 → 2
   escalation are all about not contaminating Aegir's tokenizer-free
   H-Net chunking with Nano's BPE bias.

## Immediate next actions

1. Land the phase-1 SOTAB sweep (background, ~1 week unattended).
2. Build/UI work now targets Phase 1 observability: status cards,
   Classifications view (Apache Atlas vernacular), Ontologies view
   (BERTSubs-style ontology subsumption for user-supplied vocab mapped
   into our ICE/BFO training ontology).
3. After Phase 1 lands, attempt Phase 1.5 Mergekit fusion.
4. Tokensurgeon spike scheduled for the Phase 1 → 1.5 transition. The
   spike result, not the phase timing, decides whether Phase 2 starts.

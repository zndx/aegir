# P0 — Literature review and positioning (v1)

*2026-05-09 — supersedes v0 at
`223537_lit_review.md`. Adds three priority-axis sweeps:
(1) ISWC/ESWC proceedings 2018–2026 + citation-forward from
OntoTune, Bakker et al., LLMs4OL 2025;
(2) RL on graph-structured outputs with verifiable rewards;
(3) confirmation that no TBox-specific flat-pretraining
verbalization corpus precedent exists beyond the KELM lineage.*

## Status

- **v1 — this document.** Three targeted research-agent sweeps
  conducted in parallel; ~16 new candidate papers vetted; six
  added to the adjacent-prior-art list; contribution claim
  reframed to a sharper, more defensible form.
- **P0 exit gate: firmly green.** v0 had it provisionally green;
  v1 confirms the chain-of-three claim survives a depth-of-search
  expansion across the three highest-risk axes.
- The brief at `docs/current/ontology/concept_brief.md` should be
  promoted v0.4 → v0.5 to incorporate the v1 findings (three new
  differentiation paragraphs + sharper contribution claim
  framing).

## Headline change vs. v0

The contribution claim is sharper. v0 framed the novelty as the
chain-of-three (verifier + RL + downstream pretraining slice). v1
finds that *each link in the chain has direct prior art* on its
own — what's novel is the specific combination of components,
and **especially** the use of a *semantically grounded
deterministic verifier* (i.e., one that runs a description-logic
reasoner over OWL) rather than the extrinsic LLM-judge or
execution-based verifiers that dominate adjacent RL-on-structured-
output work.

The reframed novelty claim:

> The OWL artifact is the only graph-structured output where the
> verifier can run a sound-and-complete reasoner producing both
> structural and semantic verdicts (consistency, entailment,
> class-hierarchy coverage). That combination — graph-structured
> output with a *semantically grounded* deterministic verifier
> targeting an LLM policy under RLVR, plus the verbalization-
> corpus pretraining application — is the gap.

This framing is more defensible against reviewers because it
admits each component recipe (RL on graphs; deterministic
schema/execution verifier RL; verbalization corpora) and
positions Aegir as the synthesis where the verifier acquires
deductive semantics that none of the precedents have.

## v1 axis 1 — ISWC/ESWC + citation-forward sweep

### New must-cite hits

**OLLM — End-to-End Ontology Learning with Large Language Models.**
Lo, Jiang, Li, Jamnik. NeurIPS 2024. Closest prior art on
*end-to-end LLM ontology generation*. Fine-tunes an LLM with a
custom regulariser that reduces overfitting on high-frequency
concepts; produces taxonomic backbones from scratch. Differs on
two axes: (a) plain SFT with a regulariser, no RL; (b) taxonomic
backbones only, not full OWL with restrictions or
equivalentClass axioms. Should be cited as the **primary
baseline** the brief's RL approach is being compared against.

**Knowledge-to-Verification (K2V).** Yuan et al., ICLR 2026
submission. Closest *methodological cousin*: builds a KG from
text and frames KG completion as a verifiable QA task to derive
dense rule-based rewards for LLM reasoning. Trained via RLVR.
Differs in artifact and verifier shape: the policy emits QA
reasoning traces, not OWL; the verifier checks subtask
correctness, not ontology loadability / axiom density / topic
alignment. Strong adjacent precedent that proves the
RLVR-+-KG-derived-reward shape works.

**Heterogeneous LLM Methods for Ontology Learning.** Beliaeva,
Rahmatullaev. arXiv 2508.19428, Aug 2025. LLMs4OL 2025
submission combining RAG, ensemble cosine-similarity classifiers,
and a cross-attention is-a predictor. No RL, no full-OWL
generation; cite as part of the LLMs4OL 2025 landscape evidence
that the field has not converged on RL.

### Confirmed negative space

The 2024 and 2025 LLMs4OL challenge corpus (40+ papers across
two TIB-OP volumes including representative samples like
RWTH-DBIS, silp_nlp, Phoenixes, IRIS, LABKAG) was sampled.
**Zero use RL/PPO/GRPO/DPO** — all rely on prompting, RAG,
embedding ensembles, or supervised fine-tuning. None target
full OWL axiom generation; none use ontology verbalizations as
a flat pretraining slice. This is positive evidence the gap is
real.

ESWC 2024/2025 ontology-LLM track papers (NeOn-GPT, LLMs4OM,
concept-placement) — all prompting/embedding pipelines, no RL.
Bakker et al. (axiom-identification benchmark, Dec 2025) is
benchmark-only; provides an evaluation harness the brief could
borrow but does not overlap.

## v1 axis 2 — RL on graph-structured outputs

### New must-cite hits

**AutoGraph-R1 — End-to-End Reinforcement Learning for Knowledge
Graph Construction.** Tsang, Bai, Huang, Xiao, Zheng, Xu, Liu,
Song. arXiv 2510.15339, Oct 2025 (ICLR 2026 submission).

> **This is the single closest precedent in v1.** A GRPO-trained
> LLM policy emits a knowledge graph from text; reward is a
> "Knowledge-Carrying Reward" computed from the graph's downstream
> RAG utility — an external LLM-judge checks whether gold answers
> are deducible from triples retrieved over the constructed graph.

Differentiation:

- **Artifact.** AutoGraph-R1 emits flat (head, rel, tail) triples
  — ABox-style instance facts. Aegir emits OWL TBox compositions
  with class axioms, restrictions, equivalentClass intersections.
- **Verifier.** AutoGraph-R1's reward is *extrinsic* (downstream
  QA utility judged by an LLM); Aegir's verifier is *intrinsic*
  and *semantically grounded* (DL reasoner consistency check +
  programmatic structural property checks). Sound-and-complete
  reasoning is unavailable to AutoGraph-R1's flat-triple output by
  construction.
- **Downstream use.** AutoGraph-R1's graph is consumed by a RAG
  retriever; Aegir's ontology is consumed as a pretraining-corpus
  generator (verbalizations) and a leaderboard-prediction
  ancestor.

This is the most-likely-to-be-confused precedent and the brief's
related-work section must address it head-on.

**Schema Reinforcement Learning (SRL).** Lu, Li, Cong, Zhang, Wu,
Lin, Liu, Liu, Sun. ACL 2025 Long. Trains LLMs with RL against a
fine-grained JSON Schema validator; reward is correctness ratio
over JSON tokens. Strong methodological adjacency on the
*constrained-output + deterministic-validator* axis.
Differentiation: JSON Schema is tree-structured (not graph);
the schema is given (not inferred); no semantic reasoning over
output. Cite as "schema-RL where the artifact gets DL semantics
and the verifier runs an OWL reasoner."

**RL-Struct.** arXiv 2512.00319, late 2025. GRPO with hierarchical
reward (structural integrity, format, content, validity) for
reliable JSON structured output. Cite alongside SRL.

**Rewarding Smatch — Transition-Based AMR Parsing with
Reinforcement Learning.** Naseem, Shah, Wan, Florian, Roukos,
Ballesteros. ACL 2019. Earliest direct precedent of "RL policy →
graph artifact → structural reward." Stack-LSTM transition parser
with policy learning; reward is Smatch F1 of sampled AMR graphs
against gold. **Modernized by Llama-GRPO at ACL XLLM 2025**, which
modernizes the recipe with LLM + GRPO and a multi-component
structural reward. Cite as foundational and modern AMR-RL precedent.

**CodeRL** (Le et al., NeurIPS 2022), **PPOCoder** (TMLR 2023),
**RLEF**, **Reasoning-SQL** (arXiv 2503.23157, 2025). The
canonical RLVR-on-code / RLVR-on-SQL chain that Aegir's recipe
literally inherits, replacing `unit_test(program)` with
`owl_consistency_check(ontology) + property_coverage(ontology)`.
Cite the canonical 2 (CodeRL + RLEF) plus Reasoning-SQL.

**Concept Learning for EL++ by Refinement and Reinforcement.**
Lehmann & Haase, 2012. Only prior "RL + DL ontology" hit;
symbolic refinement-operator RL over a small concept space. Worth
a citation as the legacy bound; clearly pre-dates and is
methodologically distinct from LLM-policy-based work.

### Bottom line for axis 2

The brief's specific shape — RL-trained LLM policy emits OWL;
deterministic OWL-reasoner verifier scores consistency,
coverage, and structural axioms — appears genuinely novel. But
the *recipe* is well-precedented across the three precursor lines
(graph-output RL, schema-validator RL, code/SQL execution RL),
and the brief should frame its contribution as **the synthesis of
the three precursors onto an artifact (OWL) where the verifier
acquires DL deductive semantics that none of them have**. That
synthesis is the load-bearing novelty.

## v1 axis 3 — TBox flat-pretrain verbalization sweep

### Bottom line

**The brief's TBox-flat-pretrain claim (Claim B) survives v1.**
~12 search axes including TBox+pretrain combinations, OWL
pretraining queries, KELM citation-forward, ontology
verbalization corpus, knowledge-enhanced pretraining surveys —
zero papers found that:

1. construct a corpus of verbalized OWL TBox axioms,
2. mix it with general-purpose text, and
3. pretrain (from scratch or continued) under a flat next-token
   / byte-level objective with held-out perplexity / bpb
   evaluation.

### New adjacent-prior-art hits to cite

**Language Models as Ontology Encoders (OnT).** Yang, Chen, He,
Gao, Horrocks. arXiv 2507.14334, 2024. Compositional
verbalization of OWL class expressions feeding a pretrained
Sentence Transformer (all-MiniLM-L12-v2), re-trained via
hyperbolic-space hierarchy/role/conjunction losses.
Differentiation: **auxiliary-objective embedding**, not flat
next-token loss; underlying LM is not pretrained on the
verbalizations themselves.

**Language Model Analysis for Ontology Subsumption Inference.**
He et al., ACL Findings 2023. **Probing/evaluation** of
pretrained LMs on TBox subsumption with verbalized class axioms;
some few-shot fine-tuning. Differentiation: probing + fine-tune,
not pretraining-corpus construction.

**DRAGON — Deep Bidirectional Language-Knowledge Graph
Pretraining.** NeurIPS 2022. Closest "pretraining + KG" hit:
co-pretrains over text + raw KG triples with bidirectional MLM +
link prediction. Differentiation: dual-stream auxiliary-objective
pretraining over raw triples (not verbalizations); no TBox focus
(uses ConceptNet / UMLS subgraphs, ABox-leaning); no flat
verbalization slice.

**Verbalisation Techniques of Ontologies for Use with LLMs.**
Ganellari, TU Wien Diplomarbeit, 2025. Survey/evaluation
methodology study comparing DeepOnto, OntoLearn, Zaitoun for
verbalization quality. No pretraining experiment.

## Updated adjacent prior art (cumulative)

The brief's adjacent-prior-art table should now contain seven
explicit entries with differentiation paragraphs:

| Citation | Closest axis | Differentiation |
|---|---|---|
| KELM/TEKGEN (Agarwal et al., NAACL 2021) | verbalization corpus | ABox triples not TBox; retrieval not pretrain; QA not stratified bpb |
| OntoTune (Liu et al., WWW 2025) | optimization on ontology | SFT not RL; emits text not OWL; implicit reward not deterministic |
| Zaitoun et al. (AAAI Symp 2024) | OWL verbalization | Paired SFT not flat self-supervised pretrain |
| **OLLM (Lo et al., NeurIPS 2024)** | end-to-end OWL gen | SFT + regulariser, no RL; taxonomic backbone only, not full OWL |
| **AutoGraph-R1 (Tsang et al., 2025)** | RL + graph output | Flat KG triples not OWL TBox; extrinsic LLM-judge reward not DL reasoner; downstream RAG not pretrain |
| **K2V (Yuan et al., ICLR 2026)** | RLVR + KG-derived reward | Policy emits QA traces not OWL; verifier checks subtask correctness not ontology properties |
| **OnT (Yang et al., 2024)** | TBox verbalization | Auxiliary-objective embedding via hyperbolic loss, not flat next-token pretrain |

Plus secondary citations to anchor methodological lineage:

- **SRL (Lu et al., ACL 2025) + RL-Struct** — JSON-Schema-RL
  precedent for the constrained-output + deterministic-validator
  axis.
- **Rewarding Smatch (Naseem et al., ACL 2019) + Llama-GRPO (ACL
  XLLM 2025)** — foundational + modern AMR-RL precedent for "RL
  policy → graph artifact → structural reward."
- **CodeRL (NeurIPS 2022) + RLEF + Reasoning-SQL** — RLVR-on-code
  / RLVR-on-SQL canonical chain.
- **Concept Learning for EL++ by Refinement and Reinforcement
  (Lehmann & Haase, 2012)** — legacy "RL + DL ontology" bound.
- **DRAGON (NeurIPS 2022)** — closest "pretraining + KG" hit.

## Refined "we are not aware of prior work that..." statement

> **We are not aware of prior work that:**
>
> **(a)** defines a deterministic, semantically grounded verifier
> *R*(*O*, *I*) over an OWL ontology *O* and a target text corpus
> *I* whose components are *(a-i)* DL-reasoner-checked
> consistency / loadability, *(a-ii)* complex-asserted-class
> density, *(a-iii)* DeepOnto-derived verbalization quality, and
> *(a-iv)* cross-corpus topic alignment between *I* and the
> ontology's verbalizations, with thresholds derived from a
> structural-shuffle null distribution;
>
> **(b)** trains an LLM policy via GRPO against *R* as a strict
> verifiable reward, where the policy generates OWL ontology
> compositions over a typed-slot template catalog (DeepOnto-
> validated offline, JVM-free at training time); *or*
>
> **(c)** uses such an RL-trained policy's gate-passing outputs
> as an ontology-grounded synthetic corpus slice for byte-level
> pretraining of a hierarchical sequence model, with stratified
> held-out evaluation that isolates ontology-construct
> contribution.

The novel synthesis is **(a) + (b) + (c) on OWL with a DL-reasoner
verifier**. Each component recipe has prior precedent on adjacent
artifacts (flat KG, JSON, code, SQL, AMR), but the OWL-specific
combination — where the verifier acquires sound-and-complete
deductive semantics not available to those precursor artifacts —
is the load-bearing gap.

## Risks remaining for v2

v1 closes the highest-risk uncertainties from v0. The remaining
soft uncertainties:

1. **Topic-model evaluation of ontologies (L5 in v0)** is still
   under-surveyed. v1 sampled ESWC/ISWC ontology+LLM tracks but
   did not separately sweep topic-model-evaluation venues
   (CIKM, KDD topic-modeling tracks). Low priority since the L5
   gap was confirmed; no prior art surfaced contradicts the
   topic-alignment-as-verifier-component framing.

2. **AutoGraph-R1 reproducibility / methodology depth.** The
   brief's positioning against AutoGraph-R1 depends on careful
   reading of their reward shape and graph emission protocol.
   v2 should include a depth-read of the AutoGraph-R1 paper
   when it's accepted (currently ICLR 2026 submission) — if
   their reward function turns out to include any structural
   verifier component, the differentiation language sharpens.

3. **ICLR 2026 final-round reveals.** Several v1-surfaced papers
   (K2V, AutoGraph-R1) are ICLR 2026 submissions; final
   acceptance / rejection and any methodology revisions in
   camera-ready will affect the citations. v2 should refresh
   after ICLR notifications (Jan 2026 if standard timeline).

These are quality-of-positioning concerns, not contribution-
claim-survival concerns. The chain-of-three claim is firmly
established as the synthesis of multiple precursor recipes onto
an artifact (OWL) where the verifier's deductive semantics make
the synthesis distinctive.

## Decision

**P0 exit gate: firmly green.**

The contribution claim survives v1 with sharper framing. The
brief should be promoted v0.4 → v0.5 with:
- Three new differentiation paragraphs (OLLM, AutoGraph-R1, K2V).
- Sharper contribution-claim language emphasizing the DL-reasoner
  verifier as the load-bearing distinction.
- Citations index updated to include the secondary methodological
  precedents (SRL/RL-Struct, AMR-RL chain, CodeRL/Reasoning-SQL,
  DRAGON, OnT, Lehmann & Haase 2012).

P1a catalog construction is now unblocked from a literature-
review perspective. Author-week and engineering-effort gating
remain; those are operational rather than research concerns.

## Citations index (v1 additions)

For paper-drafting `.bib` once paper writing starts:

- **Lo, J., Jiang, A. Q., Li, Y., Jamnik, M.** *End-to-End
  Ontology Learning with Large Language Models.* NeurIPS 2024.
- **Tsang, K., Bai, X., Huang, S., Xiao, Y., Zheng, K., Xu, Z.,
  Liu, J., Song, Y.** *AutoGraph-R1: End-to-End Reinforcement
  Learning for Knowledge Graph Construction.* arXiv:2510.15339,
  Oct 2025 (ICLR 2026 sub).
- **Yuan et al.** *Knowledge-to-Verification (K2V).* ICLR 2026
  submission.
- **Lu, X., Li, M., Cong, Z., Zhang, F., Wu, J., Lin, Z., Liu, P.,
  Liu, X., Sun, M.** *Schema Reinforcement Learning.* ACL 2025
  Long.
- **(anonymous).** *RL-Struct.* arXiv:2512.00319, 2025.
- **Naseem, T., Shah, A., Wan, H., Florian, R., Roukos, S.,
  Ballesteros, M.** *Rewarding Smatch: Transition-Based AMR
  Parsing with Reinforcement Learning.* ACL 2019.
- **(authors).** *Enhancing AMR Parsing with Group Relative Policy
  Optimization (Llama-GRPO).* ACL 2025 XLLM workshop.
- **Le, H., et al.** *CodeRL: Mastering Code Generation through
  Pretrained Models and Deep Reinforcement Learning.* NeurIPS
  2022.
- **(authors).** *Reasoning-SQL.* arXiv:2503.23157, 2025.
- **Lehmann, J., Haase, P.** *Concept Learning for EL++ by
  Refinement and Reinforcement.* 2012.
- **Yang, J., Chen, J., He, Y., Gao, Z., Horrocks, I.** *Language
  Models as Ontology Encoders (OnT).* arXiv:2507.14334, 2024.
- **He, Y., et al.** *Language Model Analysis for Ontology
  Subsumption Inference.* Findings of ACL 2023.
- **Yasunaga, M., Bosselut, A., Ren, H., Zhang, X., Manning,
  C. D., Liang, P., Leskovec, J.** *DRAGON: Deep Bidirectional
  Language-Knowledge Graph Pretraining.* NeurIPS 2022.
- **Beliaeva, V., Rahmatullaev, A.** *Heterogeneous LLM Methods
  for Ontology Learning.* arXiv:2508.19428, Aug 2025.
- **Ganellari, A.** *Verbalisation Techniques of Ontologies for
  Use with LLMs.* TU Wien Diplomarbeit, 2025.

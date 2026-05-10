# P0 — Literature review and positioning (v0)

*2026-05-09 — first-pass positioning document for the v0.3 concept
brief at `docs/current/ontology/concept_brief.md`. This is **v0**:
written in one session against training knowledge plus a focused web
gap-check on the two highest-risk novelty claims, not against the
~2-week deep-reading exercise the brief committed to. v1 follows
when that deeper reading completes.*

## Status of this document

- **v0 — this draft.** Surveys five literatures at the resolution my
  training and a one-shot web gap-check provide. Names representative
  works, distinguishes what's done from what's open, drafts the
  intersection statement.
- **v1 — committed deliverable.** Proper structured review against
  ACL Anthology, ISWC proceedings, NeurIPS/ICLR/ICML, recent arXiv,
  and at minimum spot-checks of citation graphs around the closest
  adjacent work. Goes here at v1.0.

The brief's contribution claim survives v0. Two papers must be
addressed head-on in the brief regardless of where v1 lands:
**KELM / TEKGEN (Agarwal et al., NAACL 2021)** for Claim B and
**OntoTune (Liu et al., WWW 2025)** for Claim A. Specific
differentiation language drafted at the end of this document.

## The five committed literatures

The concept brief commits review across five areas. Each is surveyed
below at v0 resolution.

### L1 — Verifiable reward in language models (RLVR)

**Representative works:**

- Lambert, N., et al. *Tülu 3: Pushing Frontiers in Open Language
  Model Post-Training.* 2024. Popularized the RLVR framing in open
  post-training; verifiable reward over math, instruction following,
  format compliance.
- DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in
  LLMs via Reinforcement Learning.* 2025. GRPO at scale on math /
  code with deterministic verifiers.
- Shao, Z., et al. *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models.* 2024. Original GRPO algorithm
  paper; group-relative advantage estimation without a learned
  critic.
- *awesome-RLVR* community list (github/opendilab/awesome-RLVR).
  Confirms breadth of recent RLVR work; primary domains are math,
  code, format compliance, instruction-following.

**What's done.** Verifiable rewards applied to *model outputs*:
correctness of math solutions, code-execution pass/fail,
format-conformance, multi-turn instruction following, tool-call
correctness. GRPO is the dominant algorithm; PPO with shaped reward
is the older alternative. Verifiers are typically external programs
(test suites, math graders, regex format checkers).

**What's open / our intersection.** Verifiable rewards applied to
*generated artifacts that are not direct task outputs* — e.g., an
ontology, a database schema, a configuration file — where the
reward function evaluates structural and content properties of the
artifact rather than its task-correctness. The closest published
RLVR work targets symbolic outputs (math, code) where correctness is
binary; ontology generation has continuous quality dimensions
(complex-class density, verbalization quality, topic alignment) that
require an aggregated verifier of the kind this brief defines.

**Risk-checked.** A web search across arXiv, ACL Anthology, ISWC
LLMs4OL 2025 challenge, and the awesome-RLVR list found no direct
overlap. The 2025 LLMs4OL challenge overview characterizes
participant approaches as "prompt engineering, RAG, fine-tuning, and
ensembles" — RL is conspicuously absent.

### L2 — Ontology engineering with deep learning

**Representative works:**

- He, Y., Chen, J., Antonyrajah, D., Horrocks, I. *DeepOnto: A
  Python package for ontology engineering with deep learning.*
  Semantic Web Journal 2023. The package this brief consumes for
  catalog-construction-time loadability + complex-class +
  verbalization.
- Babaei Giglou, H., D'Souza, J., Auer, S. *LLMs4OL: Large Language
  Models for Ontology Learning.* ISWC 2023; LLMs4OL 2025 challenge
  at TIB-OP. Prompting / SFT-based ontology learning; no RL.
- Liu, Z., et al. *OntoTune: Ontology-Driven Self-training for
  Aligning Large Language Models.* WWW 2025 (arXiv:2502.05478).
  Self-training loop that uses an ontology to guide LLM refinement;
  reward is implicit (does-LLM-already-know-this gating); model
  emits text answers, not OWL.
- Bakker, M., et al. *Ontology Learning with LLMs: A Benchmark
  Study on Axiom Identification.* arXiv:2512.05594, 2025.
  Eval-only benchmark; no policy training.
- Zaitoun, H., Sagi, T., Peleg, M. *Generating Ontology-Learning
  Training Data through Verbalization.* AAAI Symposium Series Vol.
  4 No. 1, 2024. LLM-assisted verbalization of OWL axioms to create
  text→OWL **SFT pairs** (not pretraining corpus slice).
- *Ontology Generation using Large Language Models* (arXiv:2503.05388,
  2025). Prompting-based generation; no RL.
- Caufield, J. H., et al. *OntoGPT.* GitHub project, ongoing.
  Prompting-based extraction.

**What's done.** LLMs author ontology fragments via prompting and
SFT. Verbalizers turn OWL axioms into natural language. Verifiers
exist for ontology *alignment* (BERTMap; AML at OAEI). Ontology
*evaluation* metrics include logical consistency, coverage, and
expert-judged quality.

**What's open / our intersection.** A deterministic verifier for
*generated ontologies* that combines DeepOnto loadability,
complex-class density, verbalization quality, and topic alignment
into a single scalar reward — and an RL training loop that targets
that reward — has not been published to my knowledge. OntoTune
(WWW 2025) is the closest hit on the optimization side: it iterates
an LLM against an ontology-grounded objective, but the optimization
is SFT-style (does-LLM-already-know-this), not RL with a
deterministic verifier.

### L3 — Verbalization-augmented language modeling

**Representative works:**

- Petroni, F., et al. *Language Models as Knowledge Bases?* EMNLP
  2019. LAMA probing; KG-derived cloze tasks for **evaluation**, not
  corpus.
- Logan, R., et al. *Barack's Wife Hillary: Using Knowledge Graphs
  for Fact-Aware Language Modeling.* ACL 2019. KGLM — KG triples
  injected as auxiliary supervision.
- Zhang, Z., et al. *ERNIE: Enhanced Language Representation with
  Informative Entities.* ACL 2019. KG entities concatenated at the
  embedding layer.
- Wang, X., et al. *KEPLER: A Unified Model for Knowledge Embedding
  and Pre-trained Language Representation.* TACL 2021. Joint
  entity-link + LM pretraining objective.
- Liu, W., et al. *K-BERT: Enabling Language Representation with
  Knowledge Graph.* AAAI 2020. KG injection at inference, not
  pretrain.
- **Agarwal, O., Ge, H., Shakeri, S., Aharoni, R. *Knowledge Graph
  Based Synthetic Corpus Generation for Knowledge-Enhanced Language
  Model Pre-training* (KELM/TEKGEN). NAACL 2021 (arXiv:2010.12688).
  Verbalizes the entire English Wikidata into ~18M synthetic
  sentences; integrates as a REALM retrieval corpus; evaluates on
  Natural Questions and WebQuestions.** This is the closest hit to
  Claim B and the most likely "isn't this just KELM with OWL?"
  reviewer challenge.

**What's done.** KG triples injected as: auxiliary supervision
(KGLM, KEPLER), embedding-layer concat (ERNIE, K-BERT), inference-
time context (K-BERT, GraphRAG, Chain-of-Verbalization). One paper
— KELM/TEKGEN — verbalizes a full KG to text and uses the result as
a retrieval corpus, evaluated downstream on QA.

**What's open / our intersection.** Specifically:

1. **TBox axioms, not ABox triples.** KELM verbalizes Wikidata
   *facts* (subject-predicate-object triples — ABox). This brief
   verbalizes OWL *class expressions* (subClassOf with restrictions,
   equivalentClass with intersections — TBox). The semantic content
   differs structurally; TBox verbalizations encode taxonomic and
   logical structure that ABox verbalizations do not.
2. **Flat pretraining mix, not retrieval augmentation.** KELM's
   verbalized corpus is fed to REALM's retriever. This brief's
   verbalization corpus is concatenated as a flat byte-level
   pretraining slice into a multi-domain mix; evaluated by stratified
   held-out bpb, not by retrieval accuracy.
3. **Stratified-by-construct eval, not QA accuracy.** KELM evaluates
   on Natural Questions / WebQuestions. This brief evaluates on bpb
   over held-out verbalizations and on stratified eval slices that
   isolate per-domain contribution.

These three differentiations are the framing the brief and any v1
literature review must defend explicitly. Without addressing KELM
head-on, reviewers will dismiss the contribution as "KELM with OWL."

### L4 — Data curation with verifiable signals

**Representative works:**

- Albalak, A., et al. *A Survey on Data Selection for Language
  Models.* arXiv:2402.16827, 2024. Comprehensive survey of data-
  selection methods, classified by signal type (heuristic, classifier,
  perplexity, deduplication).
- Penedo, G., et al. *FineWeb: Decanting the Web for the Finest Text
  Data at Scale.* 2024. Production CommonCrawl curation with
  deterministic + classifier-based filters.
- Marion, M., et al. *When Less Is More: Investigating Data Pruning
  for Pretraining LLMs at Scale.* arXiv:2309.04564, 2023. Perplexity-
  based pruning of pretraining data.
- *DataComp / DCLM 2024* (Gadre et al.). Benchmark for data-curation
  recipes for both image-text and language modeling.

**What's done.** Quality filters (perplexity, classifier-based,
heuristic dedup) over *existing* corpora; curriculum learning over
data-quality signal; active filtering during pretraining; recipes
that combine heuristic and learned signals.

**What's open / our intersection.** Treating verifier-passing
artifacts as the *generation mechanism* for new corpus slices —
not filtering existing text but generating new text whose admission
is verifier-gated. The brief's chain (ontology → verifier → admit
verbalizations → pretrain) inverts the curation direction: rather
than starting with text and curating it down, it starts with a
verified ontology and synthesizes corpus from it. The verifier-as-
generator idea sits adjacent to but distinct from the survey's
curation taxonomy.

### L5 — Topic-model evaluation of ontologies

This is the thinnest committed literature; v1 should expand it.

**Representative works:**

- Lau, J. H., Newman, D., Baldwin, T. *Machine Reading Tea Leaves:
  Automatically Evaluating Topic Coherence and Topic Model Quality.*
  EACL 2014. NPMI-based topic coherence metric.
- Newman, D., et al. *Automatic Evaluation of Topic Coherence.*
  NAACL 2010. Topic coherence as a proxy for human judgment.
- Grootendorst, M. *BERTopic: Neural topic modeling with a class-
  based TF-IDF procedure.* 2022. The default topic model the brief
  commits to (with NMF as ablation).
- Pesquita, C., et al. *Semantic Similarity in Biomedical
  Ontologies.* PLoS Comp Bio 2014. Ontology-based semantic similarity
  measures (uses ontologies to *assess* text similarity, not the
  reverse).

**What's done.** Topic coherence metrics evaluating topic models;
some use of ontology-based similarity to assess text quality;
BERTopic / NMF / LDA as standard topic models.

**What's open / our intersection.** Cross-corpus topic alignment
between an input corpus and an ontology's verbalization corpus,
used as a *deterministic quality signal for the ontology*, with
null-distribution-derived thresholds and best-match Hungarian
matching. The inverse direction from typical KG-aware topic models
(which use a KG to *condition* a topic model). I am unaware of
prior work using topic-model alignment as a verifier component for
ontology generation; v1 should search ESWC / ISWC proceedings more
systematically since this is the most likely venue for such a
proposal.

## Adjacent prior art the brief must address head-on

**KELM / TEKGEN (Agarwal et al., NAACL 2021).** This is the
single most important citation. Section 4 of the eventual paper
must contain a paragraph distinguishing this work along the three
axes named in L3:

> KELM verbalizes ABox triples from Wikidata into a retrieval
> corpus evaluated on QA. We verbalize TBox axioms from a bespoke
> OWL ontology into a flat byte-level pretraining slice evaluated
> on stratified held-out bits-per-byte. The structural content of
> the verbalized text differs (taxonomic and logical class
> expressions vs. instance-level facts); the integration mechanism
> differs (pretrain mix vs. retrieval); the evaluation isolates
> ontology-specific contribution rather than generic QA lift.

**OntoTune (Liu et al., WWW 2025).** Closest hit on the
optimization side. The differentiation:

> OntoTune iterates an LLM against an ontology-grounded objective
> via SFT, with the reward implicit in does-LLM-already-know-this
> gating. The model emits natural-language answers, not OWL. We
> train an LLM policy via GRPO with an explicit deterministic
> verifier whose output is a continuous reward in [0, 1], and the
> policy emits OWL ontology compositions whose well-formedness is
> guaranteed by the catalog's typed-slot grammar.

**Zaitoun et al. (AAAI Symposium 2024).** Closest OWL-specific
hit on verbalization. The differentiation:

> Zaitoun et al. use LLM-assisted verbalization of OWL axioms to
> create text→OWL supervised fine-tuning pairs. We treat the
> verbalizations as a flat self-supervised byte-level corpus
> mixed with general pretraining text, with no instruction-pair
> framing.

These three citations should appear in the related-work section of
any paper that emerges from this brief, with explicit
differentiation language matching the above.

## "We are not aware of prior work that..." (drafted)

Three composing claims; the chain of all three is the contribution.

> **We are not aware of prior work that:**
>
> **(a)** defines a deterministic verifier *R*(*O*, *I*) over an
> OWL ontology *O* and a target text corpus *I* whose components
> are *(a-i)* DeepOnto loadability, *(a-ii)* complex-asserted-class
> density, *(a-iii)* verbalization quality, and *(a-iv)*
> cross-corpus topic alignment between *I* and the ontology's
> DeepOnto verbalizations, with thresholds derived from a
> structural-shuffle null distribution;
>
> **(b)** trains an LLM policy via GRPO against *R* as a strict
> verifiable reward, where the policy generates OWL ontology
> compositions over a typed-slot template catalog (DeepOnto-
> validated offline, JVM-free at training time); *or*
>
> **(c)** uses such an RL-trained policy's gate-passing outputs as
> an ontology-grounded synthetic corpus slice for byte-level
> pretraining of a hierarchical sequence model, with stratified
> held-out evaluation that isolates ontology-construct contribution.

Adjacent prior art (KELM, OntoTune, Zaitoun et al., LLMs4OL 2025
challenge submissions, KGLM/ERNIE/KEPLER) covers parts of *(a)* or
*(c)* in non-overlapping ways. The chain *(a) + (b) + (c)* is, on
v0 evidence, novel.

## Risk-flags for v1 deep review

Where v0 is most likely to be wrong:

1. **L5 is under-surveyed.** ISWC and ESWC proceedings (2018–2025)
   have not been swept; a topic-model-based ontology evaluation
   proposal there would not necessarily appear in arXiv or ACL
   Anthology searches. Highest priority for v1.
2. **The 2024–2026 ontology-LM frontier moves quickly.** OntoTune
   (WWW 2025), Bakker et al. (Dec 2025), and the 2025 LLMs4OL
   challenge are all post-most-training-data-cutoffs; v1 must run a
   citation-forward search from these to catch their followups.
3. **ABox-corpus precedents beyond KELM.** v1 should sweep the
   2021–2025 KG-pretraining literature more systematically — there
   may be papers that took KELM's recipe and applied it at flat
   pretrain (rather than retrieval) without making it the headline.
4. **RL on structured-output generation.** RL training for
   structured-output generation (SQL, code, JSON, math) is well-
   covered; v1 should sweep whether anyone has trained RL policies
   for *graph-structured* output (KGs, ontologies, ERMs) against
   verifiable rewards. Adjacent but not identical to L1.
5. **The "verbalization corpus as flat pretrain" specific claim.**
   v1 should run a focused search on "ontology pretraining" + "OWL
   pretraining" + "axiom verbalization pretraining" against arXiv
   2022–2026; if KELM's recipe has been ported to OWL anywhere, v0
   missed it.

## Decision

The contribution claim survives v0 review with two adjacent works
(KELM, OntoTune) that must be addressed head-on. The brief at
`docs/current/ontology/concept_brief.md` v0.3 does not yet have
the differentiation language drafted at the end of this document;
v1 of the brief should add a "Related work" section (or annotate
the existing prior-art table) with the KELM and OntoTune
differentiation paragraphs verbatim or close to.

**P0 exit gate: provisionally green.** v1 of this lit review should
land before P1a catalog construction begins; the depth of the
adjacent-work survey may surface one or two more citations that
need addressing, but the chain-of-three claim is unlikely to
collapse based on v0 evidence.

## Next actions

- [ ] Update concept brief v0.3 → v0.4 with KELM + OntoTune +
      Zaitoun et al. differentiation paragraphs in the prior-art
      section.
- [ ] Schedule v1 literature review block (ISWC/ESWC sweep, citation-
      forward search from OntoTune and Bakker et al., RL-on-graph-
      structured-output search). Estimated effort 1–2 weeks of
      focused reading.
- [ ] Until v1 lands, treat P0 exit as provisional; new prior art
      surfaced may revise the contribution claim.

## Citations index (v0)

For copy-paste into a `.bib` file when paper drafting starts:

- Lambert et al. 2024 — Tülu 3.
- DeepSeek-AI 2025 — DeepSeek-R1.
- Shao et al. 2024 — DeepSeekMath / GRPO.
- He et al. 2023 — DeepOnto, Semantic Web Journal.
- Babaei Giglou et al. 2023, 2025 — LLMs4OL, ISWC.
- Liu et al. 2025 — OntoTune, WWW.
- Bakker et al. 2025 — Axiom Identification benchmark, arXiv 2512.05594.
- Zaitoun, Sagi, Peleg 2024 — Verbalization-derived OWL training data,
  AAAI Symposium Series.
- Caufield et al. — OntoGPT, ongoing.
- Petroni et al. 2019 — LAMA, EMNLP.
- Logan et al. 2019 — KGLM, ACL.
- Zhang et al. 2019 — ERNIE, ACL.
- Wang et al. 2021 — KEPLER, TACL.
- Liu et al. 2020 — K-BERT, AAAI.
- **Agarwal et al. 2021 — KELM/TEKGEN, NAACL.** (Required citation.)
- Albalak et al. 2024 — Data selection survey, arXiv 2402.16827.
- Penedo et al. 2024 — FineWeb.
- Marion et al. 2023 — Data pruning, arXiv 2309.04564.
- Gadre et al. 2024 — DCLM.
- Lau et al. 2014 — Topic coherence, EACL.
- Newman et al. 2010 — Topic coherence, NAACL.
- Grootendorst 2022 — BERTopic.
- Pesquita et al. 2014 — Semantic similarity in biomedical ontologies.

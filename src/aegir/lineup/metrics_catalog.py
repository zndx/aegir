"""The project's quantitative controls — the gates, measures, and KPIs that steer the THREE coupled products
(ontology · corpus · model) toward higher-quality results, and that drive model fine-tuning and the H-Net+RWKV
ground-up training.

This is a CURATED reference (definition · formula · gate/threshold · role · where-defined), surfaced as a
navigable catalog under Training → Metrics in the lineup (panel-trail: root → category → metric). It is
authored knowledge, not a live read — the live values live in the corpus/coverage/training artifacts and the
verifier/eval scripts. Keep it honest and current: when a gate's threshold or a weight changes in code, update
the matching entry here.

Each category: ``{slug, title, blurb, metrics: [{slug, name, formula, gate, role, src}]}``.
``gate`` is the admission threshold when the metric is a gate (else "—"). ``src`` is the file that computes it.
"""
from __future__ import annotations

METRICS: list[dict] = [
    {"slug": "verifier", "title": "Ontology efficacy — verifier R",
     "blurb": "The runtime efficacy score for an ontology template against the corpus. Weights locked from a "
              "P2 sweep; `tune_verifier_weights.py` re-derives them.",
     "metrics": [
         {"slug": "r", "name": "R — composite", "formula": "R = R_A·(0.50·R_B + 0.05·R_C + 0.45·R_D)",
          "gate": "—", "role": "the single efficacy number per template; drives template selection",
          "src": "ontology/verifier.py"},
         {"slug": "r-a", "name": "R_A — type gate", "formula": "hard structural gate: slots present + type-match",
          "gate": "multiplicative (R_A=0 ⇒ R=0)", "role": "kills ill-typed fills before they can score",
          "src": "ontology/verifier.py"},
         {"slug": "r-b", "name": "R_B — complexity saturation", "formula": "axiom complexity, saturating",
          "gate": "—", "role": "rewards expressivity (50% of the inner sum)", "src": "ontology/verifier.py"},
         {"slug": "r-c", "name": "R_C — semantic richness", "formula": "semantic richness of the fill",
          "gate": "—", "role": "small weight (0.05) — anti-degeneracy nudge", "src": "ontology/verifier.py"},
         {"slug": "r-d", "name": "R_D — topic alignment", "formula": "encode (MiniLM) → KMeans → Hungarian-matched "
          "cosine between generated-text centroids and the cached corpus topic model T_I",
          "gate": "—", "role": "ties templates to the real corpus topic distribution (45%)",
          "src": "ontology/topic_alignment.py"}]},

    {"slug": "chapter-verify", "title": "Chapter verification — 4 scorers",
     "blurb": "The post-generation verification loop scoring each chapter; composite is the geometric mean, "
              "status = accepted / borderline / rejected.",
     "metrics": [
         {"slug": "r-topic", "name": "R_topic", "formula": "sentence-transformer cosine to the style anchors",
          "gate": "—", "role": "does the prose read like its target topic", "src": "scripts/verify_chapters.py"},
         {"slug": "r-iri", "name": "R_iri", "formula": "cited-template keyword presence in the prose",
          "gate": "—", "role": "did the chapter actually realize the templates it cites", "src": "scripts/verify_chapters.py"},
         {"slug": "r-density", "name": "R_density", "formula": "markdown-table structure (≥2 tables GLM / ≥3 + "
          "cross-FKs Grok)", "gate": "—", "role": "relational density of the chapter", "src": "scripts/verify_chapters.py"},
         {"slug": "r-axiom", "name": "R_axiom", "formula": "table headers match slot types",
          "gate": "—", "role": "structural fidelity of the tables to the axioms", "src": "scripts/verify_chapters.py"},
         {"slug": "composite", "name": "composite + τ", "formula": "geometric mean of the four",
          "gate": "accept ≥ τ_accept 0.50 · reject < τ_review 0.30", "role": "admission of a generated chapter",
          "src": "scripts/verify_chapters.py"}]},

    {"slug": "refine-membrane", "title": "Refinement membrane — gates",
     "blurb": "The deterministic gates of the membrane-gated refinement loop: the agent PROPOSES, these "
              "DISPOSE. A chapter promotes only when all pass.",
     "metrics": [
         {"slug": "value-disjointness", "name": "value disjointness (HermiT)", "formula": "# value-level HermiT "
          "disjointness violations under the bootstrapped value-ontology fragment", "gate": "0",
          "role": "the sharpest gate — rejects cross-domain cells (the ASHRAE-in-imaging defect)",
          "src": "refine/value_gate.py"},
         {"slug": "placeholder-rate", "name": "placeholder_rate", "formula": "fraction of placeholder cells",
          "gate": "0", "role": "no canned/placeholder values in the tables", "src": "refine/eval.py"},
         {"slug": "ri-ok", "name": "ri_ok", "formula": "every FK referent resolves",
          "gate": "True", "role": "referential integrity of the materialized tables", "src": "refine/eval.py"},
         {"slug": "prose-entailment", "name": "prose_entailment", "formula": "0.7·structural (table/column "
          "names mentioned) + 0.3·value-sample (≤2 representative cells/col)", "gate": "≥ 0.5",
          "role": "prose↔table correspondence, calibrated for realized schemas — structural-weighted so a "
          "many-table chapter isn't penalized for not naming every one of its hundreds of cells", "src": "refine/eval.py"},
         {"slug": "length-ok", "name": "length_ok", "formula": "prose length within the FinePDFs band",
          "gate": "in-band", "role": "no truncated / runt chapters", "src": "refine/eval.py"}]},

    {"slug": "family-complex", "title": "Family simplicial complex",
     "blurb": "The family-set a chapter cites is a simplex; generation is gated on the empirically-measured "
              "complex so it samples only co-realizable family combinations.",
     "metrics": [
         {"slug": "r-axiom-floor", "name": "R_axiom floor", "formula": "per-simplex axiom-score floor for admission",
          "gate": "R_axiom ≥ floor, n ≥ min_count", "role": "which family-sets are allowed to co-occur",
          "src": "ontology/complex.py"},
         {"slug": "maximal-simplices", "name": "maximal_simplices", "formula": "the maximal allowed family faces",
          "gate": "—", "role": "the sampler's admissible co-citation space", "src": "ontology/family_complex.json"},
         {"slug": "below-floor", "name": "measured_below_floor", "formula": "observed punctures (failed simplices)",
          "gate": "—", "role": "holes that do NOT propagate to supersets (best_face drops the offender)",
          "src": "ontology/family_complex.json"}]},

    {"slug": "corpus-quality", "title": "Corpus quality — Semantic-Layer-Upkeep",
     "blurb": "The SLU scorers that keep the corpus from going mechanical — verbalization variety, value "
              "realism, de-canning, and realized structural complexity.",
     "metrics": [
         {"slug": "verbalization-entropy", "name": "verbalization entropy", "formula": "# distinct verbalization "
          "skeletons / top-1 share (multi-frame recomposition vs DeepOnto single string)", "gate": "—",
          "role": "axiom prose isn't one canned template (top-1 32%→7%)", "src": "ontology/verbalization.py"},
         {"slug": "value-semantics", "name": "value semantics", "formula": "placeholder / typed / domain-value rates",
          "gate": "—", "role": "real domain values, not placeholders (domain 0.14→0.80, placeholder 0.45→0)",
          "src": "ontology/entity_value_pools.json"},
         {"slug": "h-colset", "name": "h_colset — de-canning entropy", "formula": "column-set (Shannon) entropy",
          "gate": "≥ SchemaPile median", "role": "tables aren't canned clones — floors on ENTROPY, not "
          "distinct_ratio (ontology tables legitimately share typed attrs)", "src": "scripts/check_decanning_entropy.py"},
         {"slug": "structural-kpi", "name": "structural-complexity KPI", "formula": "realized vs flat growth "
          "(tables 3.21× · rows 5.0× · FKs 5.2×)", "gate": "🟢 thresholds", "role": "schema realization yields "
          "real relational breadth (EAV/junction/star)", "src": "scripts/check_structural_complexity.py"}]},

    {"slug": "coverage", "title": "Coverage — corpus generation",
     "blurb": "How well the corpus covers the ontology and the FinePDFs topic space — the generator's binding "
              "objective (coverage-close, not admit-rate).",
     "metrics": [
         {"slug": "topic-coverage", "name": "topic_coverage", "formula": "cosine of each FinePDFs topic to all "
          "540 templates", "gate": "—", "role": "which ontology regions a topic exercises", "src": "scripts/ontology_coverage_audit.py"},
         {"slug": "template-family-density", "name": "template / family density", "formula": "per-template & "
          "per-family coverage density", "gate": "—", "role": "where coverage is thin", "src": "scripts/ontology_coverage_audit.py"},
         {"slug": "r1-coverage-close", "name": "R1 — coverage-close", "formula": "register-fair coverage-close score",
          "gate": "G-cov R1 floor", "role": "THE generator metric (corpus-as-deliverable gate)", "src": "G-cov"}]},

    {"slug": "eval", "title": "Downstream eval instruments",
     "blurb": "The model-side instruments — built to be shortcut-hardened and CI-honest (the 'random ≈ "
              "pretrained' null is an instrument problem, not a verdict).",
     "metrics": [
         {"slug": "relational-edge-probe", "name": "relational edge-probe (E2)", "formula": "CPA realization-as-"
          "classification probe", "gate": "—", "role": "does grounding help relational/CPA prediction", "src": "G-rel / inc-2d"},
         {"slug": "mdl-selectivity", "name": "MDL probing + selectivity", "formula": "MDL description length + "
          "control-task selectivity (Voita&Titov, Hewitt&Liang)", "gate": "—", "role": "guards against the probe "
          "memorizing — the named fix for the calibration null", "src": "ontology-cpa methodology"},
         {"slug": "sample-efficiency", "name": "sample-efficiency curves", "formula": "k-shot accuracy & "
          "selectivity vs shots (not full-data points)", "gate": "—", "role": "the right axis for a data-value claim",
          "src": "path-a eval"},
         {"slug": "bits-per-byte", "name": "bits/byte", "formula": "byte-level LM loss (label-aware variant)",
          "gate": "—", "role": "general-LM fitness (non-degeneracy)", "src": "train_pretrain.py"},
         {"slug": "stats", "name": "bca_ci · paired_permutation", "formula": "BCa bootstrap CIs · paired "
          "permutation test", "gate": "CI-clean / p", "role": "every comparison reports a CI, not a point",
          "src": "utils/train.py"}]},

    {"slug": "token-economics", "title": "Token economics — measurable impact",
     "blurb": "The mixing math that sets how big the bespoke corpus must be to measurably move the RWKV-7 models.",
     "metrics": [
         {"slug": "y-eff", "name": "Y_eff — max yield", "formula": "max non-repetitive token yield of the corpus",
          "gate": "—", "role": "the corpus's information budget (~7.93M prior)", "src": "issue #42"},
         {"slug": "alpha-mix", "name": "α — mixing fraction", "formula": "corpus tokens / total run tokens, ≤4 "
          "epochs", "gate": "target α≈5% (CI-clean) · cap ≤15%", "role": "how much of the training mix is ours",
          "src": "gated-scaling math"},
         {"slug": "impact-target", "name": "measurable-impact target", "formula": "Y = α·T/E ≈ 100M (0.1B/8B) / "
          "200M (0.4B/16B)", "gate": "~150–200M tokens", "role": "the corpus-size goal (NOT ~1M)", "src": "gated-scaling math"}]},

    {"slug": "naturalness", "title": "Naturalness norms",
     "blurb": "A re-runnable tracker of the corpus's mechanical character vs a FinePDFs anchor — prose and "
              "table utility.",
     "metrics": [
         {"slug": "prose-mechanicalness", "name": "prose mechanicalness", "formula": "4-arm (qwen/grok/glm/"
          "finepdfs-anchor) prose-character panel", "gate": "—", "role": "how natural the prose reads (GLM closest)",
          "src": "scripts/compare_corpus_naturalness.py"},
         {"slug": "table-utility", "name": "table utility", "formula": "trivial-col / placeholder-cell / numeric "
          "rates", "gate": "—", "role": "tables carry real content — a rows.py pipeline lever, not an LLM choice",
          "src": "scripts/compare_corpus_naturalness.py"}]},

    {"slug": "meta-harness", "title": "Meta-harness — ontology admission gates",
     "blurb": "The membrane gates that admit a generated ontology primitive — HermiT is the oracle, the rest "
              "are structural/quality checks.",
     "metrics": [
         {"slug": "consistent", "name": "consistent (HermiT)", "formula": "HermiT consistency of the candidate "
          "axioms", "gate": "True", "role": "the reasoner beachhead — logically admissible or rejected", "src": "ontology/reasoning_gates.py"},
         {"slug": "deeponto-ok", "name": "deeponto_ok", "formula": "verbalizer produces a faithful surface",
          "gate": "True", "role": "the axiom can be put into prose", "src": "ontology/deeponto_harness.py"},
         {"slug": "polyglot-ok", "name": "polyglot_ok", "formula": "the DDL lowering parses (polyglot SQL)",
          "gate": "True", "role": "the syntactic (SQL) axis of admission", "src": "meta_harness"},
         {"slug": "r1-ci-low", "name": "r1_ci_low", "formula": "domain-grounding R1 CI lower bound",
          "gate": "binding v0.4 constraint", "role": "the generator must clear domain-grounding, CI-honest", "src": "G-cov R1"}]},

    {"slug": "ontology-rigor", "title": "Ontology rigor — IOF/BFO metrology + OQuaRE",
     "blurb": "The IOF-derived rigor dimensions + the OQuaRE 1-5 quality gate on the realized, FinePDFs-derived "
              "ontology (corpora/ontology/sdg-ontology.owl). Live values stamped from ontology_metrology.compute "
              "+ ontology_oquare; benchmarked against the IOF/BFO signature (structurally clean → also rigorous).",
     "metrics": [
         {"slug": "defin-complete", "name": "definitional completeness", "formula": "≡-defined classes / domain classes",
          "gate": "OQ-Rigor ≥0.45", "role": "genuine definitions (necessary+sufficient) vs subClassOf primitives — the IOF discriminator",
          "src": "scripts/ontology_metrology.py"},
         {"slug": "bfo-grounded", "name": "BFO grounding", "formula": "classes whose subsumption chain reaches a BFO category / domain classes",
          "gate": "OQ-Structure ≥0.95", "role": "every domain class anchored in BFO (heads AND fillers)", "src": "scripts/ontology_metrology.py"},
         {"slug": "realizable-machinery", "name": "realizable machinery", "formula": "role/disposition/function restriction + realizes/inheres property uses",
          "gate": "OQ-Rigor >0", "role": "phase-sortals modeled as BFO ROLES, not rigid subclasses (the IOF discipline)", "src": "scripts/ontology_metrology.py"},
         {"slug": "def-annotation", "name": "definition-annotation coverage", "formula": "classes with iao:0000115 / skos:definition / rdfs:comment / domain classes",
          "gate": "OQ-Structure ≥0.90", "role": "NL/FOL definitions — the IAO/OBO documentation convention", "src": "scripts/ontology_metrology.py"},
         {"slug": "oquare-aggregate", "name": "OQuaRE aggregate (1-5)", "formula": "mean of the 6 SQuaRE characteristic scores; each metric IOF-anchored → [1,5]",
          "gate": "🟢 ≥3.5 (aim 3.9, Brick/RealEstateCore) AND FunctionalAdequacy ≥3.0", "role": "the hard publish gate on the ontology Data Product (sync._gate)",
          "src": "scripts/ontology_oquare.py"},
         {"slug": "taxonomic-cleanliness", "name": "taxonomic cleanliness", "formula": "1 − (subsumption-cycles + OntoClean-violations) / subClassOf",
          "gate": "—", "role": "un-gameable taxonomic correctness — reasoner-INVISIBLE defects a generic LLM cannot fake", "src": "scripts/ontology_metrology.py"},
         {"slug": "ontoclean-violations", "name": "OntoClean violations", "formula": "edges where an anti-rigid (role) class subsumes a non-anti-rigid (rigid) one",
          "gate": "—", "role": "the OntoClean anti-rigidity constraint (a role cannot subsume a kind) — the un-fakeable rigor discriminator", "src": "scripts/ontology_metrology.py"},
         {"slug": "subsumption-cycles", "name": "subsumption cycles", "formula": "classes reachable from themselves via subClassOf (OOPS! P06)",
          "gate": "must be 0", "role": "a hard taxonomic-correctness floor — the hierarchy must be a DAG", "src": "scripts/ontology_metrology.py"}]},
]

#!/usr/bin/env python
"""Empirical check: run BERTopic on the hand-written prose to see whether
topic structure is identifiable at all in a single ~400-word passage
scaffolded by 5 verbaliser-derived concepts.

If topics ARE identifiable, we have an objective feedback signal that
a smaller bulk-generation LLM could be evaluated against. If NOT, we've
learned the corpus floor is somewhere larger than one passage.
"""

from __future__ import annotations

import re

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

PROSE = """\
The Committee operates under written authority delegated by the Board. \
That authority takes the form of a board-approved policy whose effect is \
conditioned on the existence of an accountable enforcer: a policy that \
names no identifiable party responsible for its application is treated \
as advisory rather than binding. For every policy item the Committee \
places under active oversight, the Chair retains standing accountability \
for enforcement unless the Board explicitly designates a delegate.

The Annual Audit is the Committee's principal recurring engagement. \
Each Annual Audit covers a defined audit period and is conducted by an \
appointed Auditor who carries personal accountability for the engagement's \
conduct and conclusions. The audit period is fixed in the engagement \
letter and may not be extended without the Committee's written approval. \
The Auditor's appointment is renewed annually; rotation is required at \
least every five years to preserve independence.

Where an engagement's scope includes the specific examination of a named \
control — for example, an entity-level access control or a financial-\
reporting reconciliation procedure — the audit is also designated a \
Control Audit. The two designations are concurrent rather than mutually \
exclusive: an Annual Audit that incorporates targeted review of one or \
more controls inherits the Control Audit's reporting requirements with \
respect to those controls.

The output of any audit engagement is one or more Signed Attestations. \
A Signed Attestation is a written statement affirming the audit's \
findings; it must be signed by the Auditor and supported by at least \
one piece of preserved evidence that a subsequent reviewer could examine \
to re-derive the stated conclusion. Attestations issued without retained \
supporting evidence are not recognized by the Committee and will be \
returned to the Auditor for substantiation.

Each obligation carried by this Charter is a Targeted Requirement: it \
specifies the control whose operation satisfies the requirement and \
identifies the scope of operations or assets to which the requirement \
applies. A requirement that names neither its specifying control nor \
its application scope is not actionable; the Committee Secretary is \
responsible for revising or retiring any such requirement before the \
next Committee meeting.
"""

SEED_VERBALIZATIONS = [
    "policy that enforced by person",
    "audit that conducted by auditor and for audit period audit period",
    "audit that audits control control",
    "attestation that has supporting evidence evidence and signed by auditor",
    "requirement that applies to control and specifies control",
]

SEED_LABELS = [
    "EnforceablePolicy",
    "AnnualAudit",
    "ControlAudit",
    "SignedAttestation",
    "TargetedRequirement",
]


def split_sentences(text: str) -> list[str]:
    # Crude sentence-split on .!? followed by whitespace. Good enough for
    # corporate-prose register.
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]


def main() -> int:
    sentences = split_sentences(PROSE)
    print(f"# sentences: {len(sentences)}")
    print()

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                    device="cuda")
    sent_embs = embedder.encode(sentences, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)
    seed_embs = embedder.encode(SEED_VERBALIZATIONS,
                                 normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)

    # --- Part 1: BERTopic on the sentences (does structure emerge?) ---
    # For a ~15-sentence corpus, we have to push min_cluster_size to its
    # floor and reduce UMAP dimensionality. This is what someone
    # evaluating bulk-generation quality would do for small batches.
    model = BERTopic(
        umap_model=UMAP(n_components=2, random_state=42, n_neighbors=5),
        hdbscan_model=HDBSCAN(min_cluster_size=2, min_samples=1),
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(sentences, embeddings=sent_embs)

    print("=" * 72)
    print("BERTopic clusters on hand-written prose")
    print("=" * 72)
    info = model.get_topic_info()
    print(info.to_string(index=False))
    print()

    # Sentences per topic, with c-TF-IDF top words
    for tid in sorted(set(topics)):
        rep = model.get_topic(tid) or []
        top_words = ", ".join(w for w, _ in rep[:6])
        print(f"\n--- Topic {tid} — keywords: [{top_words}] ---")
        for s, t in zip(sentences, topics):
            if t == tid:
                print(f"   • {s[:120]}")

    # --- Part 2: seed-alignment check (does each sentence's nearest seed
    # match the seed that semantically generated it?) ---
    print()
    print("=" * 72)
    print("Sentence → nearest seed verbalisation")
    print("=" * 72)
    sims = sent_embs @ seed_embs.T  # (n_sentences, 5)
    nearest = sims.argmax(axis=1)
    nearest_sim = sims.max(axis=1)
    for i, (s, n, sim) in enumerate(zip(sentences, nearest, nearest_sim)):
        print(f"  [{i:2d}] nearest={SEED_LABELS[n]:<20} "
              f"sim={sim:.3f}  «{s[:90]}»")

    print()
    print("=" * 72)
    print("Per-seed coverage")
    print("=" * 72)
    for i, label in enumerate(SEED_LABELS):
        mask = nearest == i
        n = int(mask.sum())
        mean_sim = float(nearest_sim[mask].mean()) if n > 0 else 0.0
        print(f"  {label:<20}  n={n:2d}  mean_sim={mean_sim:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

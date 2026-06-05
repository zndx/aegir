#!/usr/bin/env python
"""BERTopic on a single whole document — what does it return?

Three runs:
  (1) Default BERTopic on [the_one_doc] — likely degenerate; let's see.
  (2) BERTopic with clustering replaced by "everything-is-cluster-0"
      (so we get the c-TF-IDF representation of the doc as its own topic).
  (3) Pre-fit BERTopic on FinePDFs-style reference corpus, then
      transform() the doc to ask "which reference topic is it nearest?"
"""

from __future__ import annotations

import sys

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Reuse the prose from the prior experiment.
sys.path.insert(0, "scripts")
from experiment_bertopic_my_prose import PROSE  # noqa: E402

DOC = " ".join(PROSE.split())  # collapse to one line


class _SingletonCluster:
    """Stand-in for both UMAP and HDBSCAN. fit/transform pass through;
    labels are all 0. BERTopic calls .fit(X, y=...) on the umap model,
    .fit_transform(X) on the umap, and .fit(X)/.labels_ on the cluster
    model — so we implement all three."""

    def fit(self, X, y=None, **kwargs):
        self.embedding_ = np.asarray(X, dtype=np.float32)
        self.labels_ = np.zeros(len(X), dtype=int)
        return self

    def transform(self, X, **kwargs):
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y=y)
        return self.embedding_

    def fit_predict(self, X, **kwargs):
        self.fit(X)
        return self.labels_

    def predict(self, X, **kwargs):
        return np.zeros(len(X), dtype=int)


def main() -> int:
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                    device="cuda")

    print(f"# document length: {len(DOC)} chars, {len(DOC.split())} words")
    print()

    # --- Run 1: default BERTopic on [single_doc] ---
    print("=" * 72)
    print("RUN 1: Default BERTopic on [single_doc]")
    print("=" * 72)
    try:
        m1 = BERTopic(verbose=False)
        t1, _ = m1.fit_transform([DOC])
        print(f"topics: {t1}")
        print(m1.get_topic_info().to_string(index=False))
        for tid in sorted(set(t1)):
            print(f"  topic {tid} keywords: {m1.get_topic(tid)}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()

    # --- Run 2: BERTopic with HDBSCAN replaced by singleton cluster ---
    print("=" * 72)
    print("RUN 2: BERTopic with everything-in-cluster-0 (skip HDBSCAN)")
    print("=" * 72)
    try:
        m2 = BERTopic(hdbscan_model=_SingletonCluster(),
                       umap_model=_SingletonCluster(),  # also bypass UMAP
                       verbose=False)
        emb = embedder.encode([DOC], normalize_embeddings=True,
                               convert_to_numpy=True).astype(np.float32)
        t2, _ = m2.fit_transform([DOC], embeddings=emb)
        print(f"topics: {t2}")
        print(m2.get_topic_info().to_string(index=False))
        for tid in sorted(set(t2)):
            rep = m2.get_topic(tid) or []
            print(f"  topic {tid} keywords (top 15):")
            for word, weight in rep[:15]:
                print(f"    {weight:.3f}  {word}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()

    # --- Run 3: pre-fit on FinePDFs reference, transform() the doc ---
    print("=" * 72)
    print("RUN 3: Pre-fit BERTopic on 200 audit-topic representatives, ")
    print("       then transform(our_doc) to find its nearest topic")
    print("=" * 72)
    try:
        import pyarrow.parquet as pq
        audit = pq.read_table(
            "/raid/checkpoints/aegir-artifacts/coverage_v0/"
            "232ea5460ce6e0bf/topic_coverage.parquet"
        ).to_pandas()
        ref_texts = audit["topic_repr_text"].fillna("").tolist()
        ref_topic_ids = audit["topic_id"].to_numpy()
        ref_top_templates = audit["top_template_id"].tolist()

        ref_embs = embedder.encode(ref_texts, normalize_embeddings=True,
                                    convert_to_numpy=True,
                                    show_progress_bar=False).astype(np.float32)

        m3 = BERTopic(hdbscan_model=_SingletonCluster(),
                       umap_model=_SingletonCluster(),
                       verbose=False)
        # Fit BERTopic on the reference corpus
        m3.fit(ref_texts, embeddings=ref_embs)

        # Embed the doc and find nearest reference topic by cosine
        doc_emb = embedder.encode([DOC], normalize_embeddings=True,
                                   convert_to_numpy=True).astype(np.float32)[0]
        sims = ref_embs @ doc_emb
        nearest_idx = int(np.argmax(sims))
        print(f"\nNearest reference topic (by cosine):")
        print(f"  audit topic_id:    {int(ref_topic_ids[nearest_idx])}")
        print(f"  top template_id:   {ref_top_templates[nearest_idx]}")
        print(f"  similarity:        {float(sims[nearest_idx]):.4f}")
        print(f"  representative text (first 220 chars):")
        print(f"    {ref_texts[nearest_idx][:220]}")
        # Show top-5 nearest
        top5 = np.argsort(-sims)[:5]
        print(f"\nTop-5 nearest reference topics:")
        for j, idx in enumerate(top5):
            print(f"  [{j+1}] topic_id={int(ref_topic_ids[idx]):3d}  "
                  f"sim={float(sims[idx]):.4f}  "
                  f"top_template={ref_top_templates[idx]}")
    except Exception as e:
        import traceback
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Stage C cold-start investigation — head-LR saturation + the deeper
## pretraining-objective gap

The hierarchical fine-tune (Stage C) produced a hard plateau at leaf
F1 ≈ 0.01-0.03. Three remediations were tried. None broke out. The
pattern across all of them isolates the cause to the pretraining
objective itself, not to the fine-tuning regime.

## Experiments

All variants: same `small` arch, same Stage B pretrained backbone
(`outputs/pretrain/20260420T002455Z/final.pt`, 744/745 tensors
restored), same SOTAB CTA dataset, same `max_length=1024`, batch 16,
same 3-epoch budget.

| Variant | parent loss w | backbone LR | head LR | @ step 1000 leaf F1 | @ stable plateau |
|---|---|---|---|---|---|
| **C**   | 0.5  | 1e-4 | 1e-4 (equal) | 0.0107 / 0.0002 | 0.0107 / 0.0002 (frozen thru step 11000) |
| **C.2** | 0.0  | 1e-4 | 1e-4 (equal) | 0.0107 / 0.0002 | 0.0107 / 0.0002 (frozen thru step 7000) |
| **C.3** | 0.0  | 5e-5 | 5e-4 (10×)   | 0.0283 / 0.0006 | 0.01-0.03 oscillating |
| **C.4** | 0.0  | 5e-5 | 5e-3 (100×)  | 0.0277 / 0.0006 | — (running) |

Reference: Stage A (direct CTA from random, hygiene bundle, no
pretraining) plateau'd at 0.011 / 0.000. Stage C variants' plateau is
in the same order of magnitude as "no pretraining at all."

## What this isolates

- **C vs C.2**: parent head is not the blocker. Adding a parent
  auxiliary signal or removing it gives identical leaf F1.
- **C.3 vs C.4**: head-LR multiplier saturates between 10× and 100×.
  Same F1. Whatever capacity the classifier head can extract from the
  pretrained embeddings, it extracts by step 1000 regardless of LR.
- **All C variants vs Stage A**: pretraining produces *some* benefit
  (~3× higher leaf F1 than Stage A) but the improvement is still
  inside the collapsed regime. Not a breakthrough.

The negative result matters: head-LR-cold-start was a *credible*
remediation. "Pretrained backbone + classifier head trained from
scratch, backbone at small LR, head at large LR" is the standard
recipe for BERT-style fine-tuning. It does not work here. That
eliminates the class of fixes that tune optimizer hyperparameters.

## Where the remaining possibilities live

Four candidate causes, ordered by plausibility given the evidence.

### 1. The pretraining objective doesn't produce column-semantic clusters

**Most likely.** Next-byte prediction on tabular serializations
teaches byte-level statistics (what characters co-occur, common cell
shapes, cell-delimiter patterns). It does NOT teach that two email
columns from different tables — say `alice@site-a.com, bob@site-a.com`
and `x@example.org, y@example.org` — are the same *semantic type*
despite having different byte distributions.

At Stage B loss 2.26 the model has learned substantial byte-level
structure. At Stage C leaf F1 0.03 the classifier can't read column
type out of that structure. The representation is rich but wrong —
rich on the axis of byte co-occurrence, impoverished on the axis of
column semantics.

Fix direction: **contrastive pretraining**. Pair columns by type
(e.g., any two `schema:Email` columns form a positive pair); train
the backbone so same-type columns produce similar pooled embeddings
and different-type columns produce different ones. SimCLR /
SupCon-style objective on pooled column embeddings. Requires some
type labels — SOTAB's own training set or GitTables' DBpedia
annotations provide them for free.

Alternative fix: **masked-cell prediction with column-type condition**.
Show the model columns `[CLS] type=email; [val] alice@... [val] bob@...`;
mask random cells, predict them. The conditioning forces the model
to represent type information in the pooled embedding.

### 2. SOTAB Schema.org is too long-tail for 91-way fine-tuning at this scale

**Possible contributor.** 91 classes with order-of-magnitude
imbalance means rare classes get minimal gradient signal. But this
would be a gradual descent, not a hard plateau. The plateau shape
argues more against this.

Mitigation if this turns out significant: class-balanced sampling,
focal loss, or per-class reweighting in CE.

### 3. The pretraining corpus is too small

**Possible.** 100 M bytes is enough to see the loss descend
meaningfully (5.68 → 2.26) but not enough for byte-level semantics
to compound into column-level clustering. RWKV-LM from-scratch
recipes routinely use 10-100× this volume.

Mitigation: scale Stage B to 1-10 B bytes. Cheap on compute (~hours
on 6 GPUs with DDP), doesn't require new code. But this is a
*brute force* bet — if the objective itself is wrong (cause 1),
more data of the same kind won't fix it.

### 4. An architectural bug specific to sparse-supervision fine-tuning

**Unlikely.** Stage B pretraining descended cleanly, representation
is verified non-collapsed, heads restore cleanly. If an architectural
bug exists, it would likely also have broken pretraining.

## Implication for the v3 plan

v3's Phase 1 ("Aegir-only baseline") implicitly assumed "pretraining +
fine-tuning will produce a baseline." What the three C variants show is
that *generic* next-byte pretraining is insufficient; the pretraining
objective needs to be task-*aware* even for the baseline.

This doesn't invalidate the phased structure but revises what Phase 1
requires. The honest Phase 1 is:

1. Byte-level pretraining (Stage B, done).
2. **Column-contrastive mid-training** on labeled or self-labeled
   column pairs — new addition.
3. Leaf-level fine-tuning (Stage C proper).

The mid-training step is new. It adds a research question (does
SupCon-style column contrastive produce the cluster geometry MCL
would recover?) that aligns naturally with the dual-center-loss
discussion — SupCon IS the contrastive form of dual center loss.

## Next concrete experiments

1. **Stop C.3 and C.4 once C.4 confirms plateau at step 2000.**
2. **Implement column-contrastive mid-training** (~day of work).
   Dataset: same SOTAB train split, sample positive/negative column
   pairs by label. Head: a projection on pooled embeddings. Loss:
   SupCon. Expected signal: intra-class cosine distance drops below
   inter-class in the diagnostic audit.
3. **If #2 works**, re-run Stage C fine-tune on top of the
   contrastive-pretrained backbone. Predict: leaf F1 breaks out of
   the 0.01-0.03 plateau.
4. **If #2 also plateaus**, Byte-level pretraining scale (cause 3)
   becomes the next lever — grow Stage B from 100 M to 1 B bytes
   before any fine-tuning.

## What we learned vs what we expected

Expected: pretraining fixes the from-scratch collapse, fine-tuning
just needs standard BERT-recipe hyperparameters.

Learned: pretraining fixes the *collapse* but not the *discrimination*
problem. The backbone outputs vary with input (Stage B's post-training
embedding diagnostic confirmed this) but the variation isn't aligned
with label semantics. Head-LR tuning can't rescue a geometric
misalignment between representation and labels.

This is exactly the kind of negative result the MCL diagnostic
framework was built to detect — intra-class cosine distance on Stage
C's best checkpoint will almost certainly match inter-class,
confirming the geometry is "alive but not semantically clustered."
A follow-on MCL audit on C.4's `best.pt` is the cheap next step
before committing to the contrastive mid-training direction.

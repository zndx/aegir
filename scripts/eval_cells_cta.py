#!/usr/bin/env python
"""Cells-only few-shot CTA value-add search over gt-signals-dbpedia (real tables).

The hardened, transfer + sample-efficiency instrument for the calibration. The
question is NOT "does the ontology help (y/n)" but **WHERE does an ontology-pretrained
backbone's value-add appear**, so we can amplify it. Per the methodology
(TAPAS/TAPEX/OmniTab + Voita & Titov): grounding value-add is maximal at *few-shot*
and ~vanishes at full data, must show on *real* tables (non-template transfer),
*cells-only* (no header shortcut), and only counts if it clears a *control task*
(selectivity) with a bootstrap CI that excludes the random floor.

Design — linear probe on a FROZEN backbone (the clean measure of what the
representation already encodes), features cached once for speed:
  1. Featurize: frozen arm backbone → pooled feature per column. Cells-only =
     headers blanked + no context columns. A fixed (seeded) pooler is shared across
     arms so the comparison is apples-to-apples.
  2. Few-shot curve: for N in the grid × seeds, fit a linear probe on N cached train
     features, evaluate on the full held-out test set.
  3. Control task (Hewitt & Liang 2019): refit on PERMUTED labels → selectivity =
     real − control (surface memorization can't lift selectivity).
  4. Bootstrap 95% CI on test accuracy.

Run once per checkpoint (``--pretrained <arm>`` ; omit → random-init floor); compare
arms at each N. Value-add = arm accuracy/selectivity above the random floor with
non-overlapping CIs, largest in the low-N regime.

Usage::
    uv run --no-sync python scripts/eval_cells_cta.py \\
        --pretrained /raid/.../ablation_v1_ckpts/full/runs/<ts>/best_model.pt --arm full
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.data.table_dataset import GitTablesSignalsDataset  # noqa: E402
from aegir.data.tokenizer import ByteTokenizer  # noqa: E402
from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig  # noqa: E402
from aegir.models.heads import AegirForColumnAnnotation  # noqa: E402

log = logging.getLogger("eval-cells-cta")
HEAD_PREFIXES = ("role_embeddings.", "pooler.", "classifier.")


def tiny_config(vocab_size: int, num_labels: int) -> AegirConfig:
    """Verbatim train_pretrain.py make_pretrain_model('tiny') + a CTA head."""
    return AegirConfig(
        arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
        d_model=[128, 192, 192], d_intermediate=[0, 0, 0],
        vocab_size=vocab_size,
        ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
        attn_cfg=AttnConfig(num_heads=[2, 3, 3], rotary_emb_dim=[16, 24, 24], window_size=[]),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=num_labels, task_type="cta",
    )


class CellsOnlyGTSignals(GitTablesSignalsDataset):
    """gt-signals-dbpedia with headers blanked (cells-only shortcut-hardening)."""

    def _load_tables(self):
        tables = super()._load_tables()
        for t in tables:
            t["column_names"] = ["" for _ in t["column_names"]]
        return tables


def load_backbone(model, ckpt_path) -> dict:
    """Copy matching-shape backbone+embedding tensors; head stays at seeded-random."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            loaded += 1
    model.load_state_dict(own)
    return {"loaded": loaded, "ckpt_total": len(state)}


def collate(batch):
    B = len(batch)
    L = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    role_ids = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    cls_indexes = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        role_ids[i, :n] = b["role_ids"]
        mask[i, :n] = True
        cls_indexes[i] = b["cls_index"]
        labels[i] = b["label"]
    return input_ids, role_ids, mask, cls_indexes, labels


@torch.no_grad()
def featurize(model, ds, device, batch_size=32):
    """Frozen backbone → MEAN-POOLED hidden state per column.

    NB: the backbone is causal (RWKV-7), so the head's CLS-at-position-0 extraction
    never sees the cells (they follow position 0) — a useless column feature, and a
    likely contributor to the original flat calibration. Mean-pooling the masked hidden
    states gives every column a representation that integrates its cells (incl. the last
    token, which under the recurrence has seen the whole sequence).
    """
    from torch.utils.data import DataLoader
    feats, labs = [], []
    for input_ids, role_ids, mask, cls_indexes, labels in DataLoader(
            ds, batch_size=batch_size, collate_fn=collate):
        input_ids, role_ids, mask = input_ids.to(device), role_ids.to(device), mask.to(device)
        h = model.embeddings(input_ids) + model.role_embeddings(role_ids)
        B, L, D = h.shape
        hs, _ = model.backbone(h, cu_seqlens=None, max_seqlen=None, mask=mask)
        hs = hs.view(B, L, D).float()
        m = mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
        feats.append(pooled.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def probe_once(Xtr, ytr, Xte, yte, num_labels, seed, control=False):
    """Fit a linear probe on (Xtr,ytr); return test (acc, macro_f1, top5, preds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y = ytr.copy()
    if control:
        y = rng.permutation(y)  # Hewitt & Liang control task
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, y)
    proba = clf.predict_proba(Xte)
    classes = clf.classes_
    pred = classes[proba.argmax(1)]
    acc = float((pred == yte).mean())
    f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
    top5 = classes[np.argsort(-proba, 1)[:, :5]]
    top5_acc = float(np.mean([yte[i] in top5[i] for i in range(len(yte))]))
    return {"acc": acc, "macro_f1": f1, "top5_acc": top5_acc}, (pred == yte)


def bootstrap_ci(correct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", default=None, help="Arm byte-LM ckpt; omit → random floor")
    ap.add_argument("--arm", default="random")
    ap.add_argument("--data-dir", default=None, help="gt-signals drop (default: cldr/signals)")
    ap.add_argument("--out-dir", default="/raid/checkpoints/aegir-artifacts/cells_cta_v0")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--shots", type=int, nargs="+", default=[128, 256, 512, 1024, 0],
                    help="few-shot train sizes; 0 = full train")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.shots, args.seeds, args.max_length = [32, 0], [1], 256

    torch.manual_seed(4649)  # identical fresh pooler across arms → fair comparison
    device = torch.device(args.device)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    tok = ByteTokenizer()

    log.info("loading cells-only gt-signals-dbpedia (headers blanked, no context) ...")
    common = dict(data_dir=args.data_dir, tokenizer=tok, max_length=args.max_length, max_context_cols=0)
    tr_ds = CellsOnlyGTSignals(split="train", **common)
    te_ds = CellsOnlyGTSignals(split="test", **common)
    num_labels = tr_ds.num_classes
    log.info("train %d / test %d columns | %d labels", len(tr_ds), len(te_ds), num_labels)

    vocab_size = ByteTokenizer.vocab_size
    if args.pretrained:
        peek = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        if "embeddings.weight" in peek:
            vocab_size = peek["embeddings.weight"].shape[0]
        del peek

    model = AegirForColumnAnnotation(tiny_config(vocab_size, num_labels)).to(device=device, dtype=dtype)
    if args.pretrained:
        log.info("load backbone [%s]: %s", args.arm, load_backbone(model, args.pretrained))
    else:
        log.info("RANDOM-INIT floor (no --pretrained)")
    model.eval()

    log.info("featurizing (frozen backbone, cached) ...")
    Xtr, ytr = featurize(model, tr_ds, device)
    Xte, yte = featurize(model, te_ds, device)

    curve = []
    full_n = len(ytr)
    for shot in args.shots:
        N = full_n if shot == 0 else min(shot, full_n)
        per_seed = []
        for sd in args.seeds:
            rng = np.random.default_rng(sd)
            idx = rng.permutation(full_n)[:N]
            real, correct = probe_once(Xtr[idx], ytr[idx], Xte, yte, num_labels, sd)
            ctrl, _ = probe_once(Xtr[idx], ytr[idx], Xte, yte, num_labels, sd, control=True)
            per_seed.append({"acc": real["acc"], "macro_f1": real["macro_f1"],
                             "top5_acc": real["top5_acc"], "control_acc": ctrl["acc"],
                             "selectivity": real["acc"] - ctrl["acc"], "_correct": correct})
        accs = np.array([s["acc"] for s in per_seed])
        all_correct = np.concatenate([s["_correct"] for s in per_seed])
        lo, hi = bootstrap_ci(all_correct)
        row = {"N": N, "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
               "acc_ci95": [lo, hi],
               "macro_f1_mean": float(np.mean([s["macro_f1"] for s in per_seed])),
               "top5_acc_mean": float(np.mean([s["top5_acc"] for s in per_seed])),
               "selectivity_mean": float(np.mean([s["selectivity"] for s in per_seed]))}
        curve.append(row)
        log.info("N=%-5d acc %.4f±%.4f CI[%.3f,%.3f] top5 %.4f sel %.4f", N,
                 row["acc_mean"], row["acc_std"], lo, hi, row["top5_acc_mean"], row["selectivity_mean"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"arm": args.arm, "pretrained": args.pretrained, "num_labels": num_labels,
              "n_train": full_n, "n_test": len(yte), "feature_dim": int(Xtr.shape[1]),
              "shots": args.shots, "seeds": args.seeds, "curve": curve}
    (out / f"result_{args.arm}_cells.json").write_text(json.dumps(result, indent=2))
    log.info("wrote %s", out / f"result_{args.arm}_cells.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run the ontology-CPA eval for one pretrained ablation arm.

Downstream-validation stage of the v0.3 procedure (pairs with
``build_ontology_cpa_eval.py``). Loads a byte-level pretraining checkpoint
for one arm, attaches the column-annotation head, fine-tunes (or
linear-probes with ``--freeze-backbone``) on the SHARED ontology-CPA task,
and reports multi-label metrics on the held-out test split.

Run it once per arm against the same built task to get the downstream
ranking, then compare to the proxy ranking (verifier ``r_composite`` in each
arm's ``verification.parquet``). That comparison IS the calibration: does
corpus-level ontology/schema grounding translate into a model that better
recognizes ontological structure, and does the proxy predict it?

Metrics. The task is sparse multi-label (~4 of 59 templates per chapter, a
~7% positive rate), so a 0.5-threshold F1 collapses to the trivial
all-negative predictor and reads 0 for *every* model — it hides the signal.
The ground-truth signal is therefore carried by **ranking** metrics, which
are threshold-free:
  * micro-AUC          — random = 0.50
  * micro-mAP          — random ~= positive rate (~0.07)
  * R-precision        — per chapter, fraction of the true-count top-scored
                         labels that are correct; random ~= positive rate
F1@0.5 is kept as a diagnostic. Training uses ``pos_weight`` so the head
actually fires rather than collapsing.

The tiny config is a verbatim copy of ``train_pretrain.py``'s
``make_pretrain_model('tiny')`` so the arm checkpoints (produced by
``train_ablation_all.sh`` -> ``train_pretrain.py``) load with identical
backbone shapes and behavior; only the head (role_embeddings/pooler/
classifier) is fresh.

Usage::

    uv run --no-sync python scripts/eval_ontology_cpa.py \\
        --eval-dir   /raid/checkpoints/aegir-artifacts/ontology_cpa_v0 \\
        --pretrained /raid/checkpoints/aegir-artifacts/ablation_v1_ckpts/full/runs/<ts>/best_model.pt \\
        --arm full
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.data.tokenizer import CLS_TOKEN_ID, ByteTokenizer  # noqa: E402
from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig  # noqa: E402
from aegir.models.heads import AegirForColumnAnnotation  # noqa: E402
from aegir.utils.train import f1_score_multilabel  # noqa: E402

log = logging.getLogger("eval-ontology-cpa")

# Heads are the only fresh tensors; everything else loads from the byte-LM ckpt.
HEAD_PREFIXES = ("role_embeddings.", "pooler.", "classifier.")


def tiny_config(vocab_size: int, num_labels: int) -> AegirConfig:
    """Verbatim train_pretrain.py make_pretrain_model('tiny') + a CPA head."""
    return AegirConfig(
        arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
        d_model=[128, 192, 192],
        d_intermediate=[0, 0, 0],
        vocab_size=vocab_size,
        ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
        attn_cfg=AttnConfig(num_heads=[2, 3, 3], rotary_emb_dim=[16, 24, 24], window_size=[]),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=num_labels,
        task_type="cpa",
    )


class OntologyCPADataset(Dataset):
    """Chapter text -> (CLS + bytes), multi-hot template labels."""

    def __init__(self, rows, num_labels, tokenizer, max_length):
        self.rows = rows
        self.num_labels = num_labels
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = [CLS_TOKEN_ID] + self.tok.encode(r["response_text"])[: self.max_length - 1]
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        for idx in r["label_idx"]:
            if 0 <= idx < self.num_labels:
                y[idx] = 1.0
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "label": y, "cls_index": 0}


def collate(batch):
    B = len(batch)
    L = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    role_ids = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    cls_indexes = torch.zeros(B, dtype=torch.long)
    labels = torch.stack([b["label"] for b in batch])
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        mask[i, :n] = True
        cls_indexes[i] = b["cls_index"]
    return {"input_ids": input_ids, "role_ids": role_ids, "mask": mask,
            "cls_indexes": cls_indexes, "labels": labels}


def load_pretrained(model, ckpt_path) -> dict:
    """Copy matching-shape tensors (embeddings + backbone) from a byte-LM ckpt."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    return {"loaded": len(loaded), "skipped": len(skipped),
            "pretrained_total": len(state), "first_skipped": skipped[:5]}


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


@torch.no_grad()
def evaluate(model, loader, device, num_labels, threshold=0.5) -> dict:
    """Threshold-free ranking metrics carry the signal (see module docstring);
    F1@0.5 kept as a diagnostic."""
    if len(loader.dataset) == 0:
        return {"val_loss": 0.0, "micro_f1": None, "macro_f1": None,
                "mAP_micro": None, "micro_auc": None, "r_precision": None,
                "n": 0, "skipped": True}
    model.eval()
    lossf = nn.BCEWithLogitsLoss()
    scores, trues = [], []
    tot_loss, n = 0.0, 0
    for batch in loader:
        out = model(input_ids=batch["input_ids"].to(device),
                    role_ids=batch["role_ids"].to(device),
                    cls_indexes=batch["cls_indexes"].to(device),
                    mask=batch["mask"].to(device))
        logits = out.logits.float()
        y = batch["labels"].to(device)
        tot_loss += lossf(logits, y).item() * y.shape[0]
        n += y.shape[0]
        scores.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(y.cpu().numpy())
    model.train()

    S = np.concatenate(scores)
    Y = np.concatenate(trues).astype(int)
    P = (S >= threshold).astype(int)
    micro, macro, _ = f1_score_multilabel(Y.tolist(), P.tolist(), num_classes=num_labels)
    res = {"val_loss": tot_loss / max(1, n), "micro_f1": float(micro),
           "macro_f1": float(macro), "n": n}

    # Threshold-free ranking signal.
    yf, sf = Y.ravel(), S.ravel()
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if 0 < yf.sum() < yf.size:
            res["micro_auc"] = float(roc_auc_score(yf, sf))
            res["mAP_micro"] = float(average_precision_score(yf, sf))
    except Exception as e:  # degenerate batch / sklearn edge
        res["rank_err"] = str(e)
    rp = []
    for i in range(Y.shape[0]):
        k = int(Y[i].sum())
        if k == 0:
            continue
        topk = np.argsort(-S[i])[:k]
        rp.append(float(Y[i][topk].sum()) / k)
    res["r_precision"] = float(np.mean(rp)) if rp else None
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval-dir", required=True, help="Output of build_ontology_cpa_eval.py")
    ap.add_argument("--pretrained", default=None,
                    help="Arm byte-LM checkpoint. Omit -> random init (smoke / floor baseline).")
    ap.add_argument("--arm", default="unknown")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr-mult", type=float, default=5.0,
                    help="LR multiplier for the cold-start head vs the pretrained backbone.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0,
                    help="Clamp on per-label neg/pos weighting (multi-label imbalance).")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Linear-probe: train only the head — a cleaner measure of "
                         "what the pretraining representation already encodes.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4649)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="Tiny fast end-to-end pipeline check.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap per-split samples (debug / smoke).")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = 2
        args.max_length = 256
        args.max_samples = args.max_samples or 6

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    eval_dir = Path(args.eval_dir)
    vocab = json.loads((eval_dir / "label_vocab.json").read_text())
    num_labels = len(vocab)

    import pyarrow.parquet as pq
    ds = pq.read_table(eval_dir / "eval_dataset.parquet").to_pandas()
    rows = ds.to_dict("records")
    for r in rows:
        r["label_idx"] = [int(x) for x in r["label_idx"]]
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if args.max_samples:
        for s in by_split:
            by_split[s] = by_split[s][: args.max_samples]
    log.info("task: %d labels | train %d val %d test %d", num_labels,
             len(by_split["train"]), len(by_split["val"]), len(by_split["test"]))
    if not by_split["train"]:
        raise SystemExit("empty train split — check the built task / --max-samples")

    tok = ByteTokenizer()
    vocab_size = ByteTokenizer.vocab_size
    if args.pretrained:
        peek = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        if "embeddings.weight" in peek:
            vocab_size = peek["embeddings.weight"].shape[0]
            log.info("inferred vocab_size=%d from checkpoint", vocab_size)
        del peek

    config = tiny_config(vocab_size=vocab_size, num_labels=num_labels)
    model = AegirForColumnAnnotation(config).to(device=device, dtype=dtype)
    if args.pretrained:
        stats = load_pretrained(model, args.pretrained)
        log.info("loaded pretrained: %s", stats)
        if stats["loaded"] == 0:
            log.warning("0 tensors loaded — config/checkpoint mismatch?")
    else:
        log.warning("no --pretrained: RANDOM INIT (smoke / floor baseline only)")

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad_(any(name.startswith(h) for h in HEAD_PREFIXES))
        log.info("freeze-backbone: linear-probe (head-only training)")

    def make_loader(split, shuffle):
        d = OntologyCPADataset(by_split[split], num_labels, tok, args.max_length)
        return DataLoader(d, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=args.num_workers)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if any(name.startswith(h) for h in HEAD_PREFIXES)
         else backbone_params).append(p)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "_base_lr": args.lr})
    if head_params:
        groups.append({"params": head_params, "_base_lr": args.lr * args.head_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    # Multi-label imbalance: ~4 of 59 positive per chapter. Without pos_weight,
    # BCE collapses to the all-negative predictor (F1 -> 0). Weight positives by
    # the per-label neg/pos ratio (clamped) so the head actually fires.
    counts = np.zeros(num_labels, dtype=np.float64)
    for r in by_split["train"]:
        for idx in r["label_idx"]:
            if 0 <= idx < num_labels:
                counts[idx] += 1.0
    n_train = max(1, len(by_split["train"]))
    pos_weight = torch.tensor(
        np.clip((n_train - counts) / np.clip(counts, 1.0, None), 1.0, args.pos_weight_cap),
        dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val_map = -1.0
    history = []
    last_loss = float("nan")
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        role_ids=batch["role_ids"].to(device),
                        cls_indexes=batch["cls_indexes"].to(device),
                        mask=batch["mask"].to(device))
            loss = lossf(out.logits.float(), batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scale = lr_scale(step)
            for pg in opt.param_groups:
                pg["lr"] = pg["_base_lr"] * scale
            opt.step()
            opt.zero_grad()
            step += 1
            last_loss = loss.item()
        ev = evaluate(model, val_loader, device, num_labels)
        history.append({"epoch": epoch + 1, "val": ev})
        log.info("epoch %d/%d  train_loss %.4f  val mAP %s  R-prec %s  AUC %s  F1@.5 %s",
                 epoch + 1, args.epochs, last_loss,
                 _fmt(ev.get("mAP_micro")), _fmt(ev.get("r_precision")),
                 _fmt(ev.get("micro_auc")), _fmt(ev.get("micro_f1")))
        if isinstance(ev.get("mAP_micro"), float):
            best_val_map = max(best_val_map, ev["mAP_micro"])

    test = evaluate(model, test_loader, device, num_labels)
    log.info("TEST [%s]  mAP %s  R-prec %s  AUC %s  F1@.5 %s  (%.1fs)",
             args.arm, _fmt(test.get("mAP_micro")), _fmt(test.get("r_precision")),
             _fmt(test.get("micro_auc")), _fmt(test.get("micro_f1")), time.time() - t0)

    result = {
        "arm": args.arm,
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "num_labels": num_labels,
        "max_length": args.max_length,
        "epochs": args.epochs, "lr": args.lr, "head_lr_mult": args.head_lr_mult,
        "best_val_mAP_micro": best_val_map,
        "test": test,
        "history": history,
        "splits": {s: len(by_split[s]) for s in by_split},
    }
    res_path = out_dir / f"result_{args.arm}.json"
    res_path.write_text(json.dumps(result, indent=2))
    log.info("wrote %s", res_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

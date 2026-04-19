#!/usr/bin/env python3
"""Synthetic DED smoke trainer — proves AegirForDED + SupCon train end-to-end.

Generates a toy cross-table dataset where each sample has N columns
distributed across K ground-truth clusters, serialized as random byte
patterns with cluster-dependent prefixes. A correctly-working contrastive
objective should pull same-cluster columns close in the embedding space,
so the B-cubed F1 of a k-means clustering over the learned embeddings
should rise monotonically during training.

This is a scaffolding sanity-check, not a benchmark — the synthetic task
is easy by construction. Its value is catching regressions in head
shape, loss sign, or training-loop wiring before they hit real data.

Usage:
    uv run --no-sync python scripts/train_ded_smoke.py
    uv run --no-sync python scripts/train_ded_smoke.py --epochs 5 --device cuda
"""
from __future__ import annotations

import argparse
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import AegirForDED
from aegir.utils.train import bcubed_f1, supcon_loss


class SyntheticDEDDataset(Dataset):
    """Cross-table DED dataset with known cluster ground truth.

    Each sample packs N_cols columns into a single byte sequence. Columns
    from the same ground-truth cluster share a deterministic byte-pattern
    prefix so a successful model learns to map them near each other in
    the contrastive embedding space.

    Args:
        num_samples: Number of samples to generate.
        n_clusters: Size of the ground-truth cluster universe.
        cols_per_sample: Number of columns per sample.
        seq_len: Byte-serialization length (incl. per-column CLS).
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 256,
        n_clusters: int = 8,
        cols_per_sample: int = 6,
        seq_len: int = 128,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.n_clusters = n_clusters
        self.cols_per_sample = cols_per_sample
        self.seq_len = seq_len
        self.rng = random.Random(seed)

        # Deterministic per-cluster byte prefix — 4 bytes is enough signal.
        self._cluster_prefix = {
            c: bytes(self.rng.getrandbits(8) for _ in range(4))
            for c in range(n_clusters)
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r = random.Random(idx)
        # Per-column region = seq_len // cols_per_sample bytes.
        region = self.seq_len // self.cols_per_sample
        input_bytes = bytearray(self.seq_len)
        role_ids = torch.zeros(self.seq_len, dtype=torch.long)
        cls_indexes = []
        cluster_ids = []

        for col_idx in range(self.cols_per_sample):
            c = r.randrange(self.n_clusters)
            cluster_ids.append(c)
            prefix = self._cluster_prefix[c]
            start = col_idx * region
            # First 4 bytes = cluster prefix, rest = noise
            input_bytes[start : start + 4] = prefix
            for j in range(4, region):
                input_bytes[start + j] = r.getrandbits(8)
            # CLS position = last byte of region
            cls_indexes.append(start + region - 1)
            role_ids[start : start + region] = col_idx + 1

        return {
            "input_ids": torch.tensor(list(input_bytes), dtype=torch.long),
            "role_ids": role_ids,
            "cls_indexes": torch.tensor(cls_indexes, dtype=torch.long),
            "cluster_ids": torch.tensor(cluster_ids, dtype=torch.long),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "role_ids": torch.stack([b["role_ids"] for b in batch]),
        "cls_indexes": torch.stack([b["cls_indexes"] for b in batch]),
        "cluster_ids": torch.stack([b["cluster_ids"] for b in batch]),
        "mask": torch.ones(len(batch), batch[0]["input_ids"].shape[0], dtype=torch.bool),
    }


def evaluate(model, loader, device) -> tuple[float, float]:
    """Report SupCon loss + B-cubed F1 (via k-means on predictions)."""
    from sklearn.cluster import KMeans  # type: ignore[import]

    model.eval()
    losses = []
    all_embs, all_cids, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                role_ids=batch["role_ids"],
                cls_indexes=batch["cls_indexes"],
                mask=batch["mask"],
            )
            loss = supcon_loss(
                out.embeddings.float(), batch["cluster_ids"], out.col_mask
            )
            losses.append(loss.item())
            all_embs.append(out.embeddings.float().cpu())
            all_cids.append(batch["cluster_ids"].cpu())
            all_masks.append(out.col_mask.cpu())

    emb = torch.cat(all_embs, dim=0)
    cid = torch.cat(all_cids, dim=0)
    mask = torch.cat(all_masks, dim=0)
    B, N, D = emb.shape
    flat_emb = emb.view(B * N, D).numpy()
    flat_mask = mask.view(B * N).bool().numpy()

    k = int(cid[mask].max().item()) + 1
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
    kmeans.fit(flat_emb[flat_mask])
    preds = torch.full((B * N,), -1, dtype=torch.long)
    preds[torch.from_numpy(flat_mask)] = torch.from_numpy(kmeans.labels_).long()
    preds = preds.view(B, N)

    _, _, f1 = bcubed_f1(preds, cid, mask)
    return float(sum(losses) / len(losses)), f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--cols-per-sample", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    train_ds = SyntheticDEDDataset(
        num_samples=512, n_clusters=args.n_clusters,
        cols_per_sample=args.cols_per_sample, seq_len=args.seq_len, seed=0,
    )
    val_ds = SyntheticDEDDataset(
        num_samples=64, n_clusters=args.n_clusters,
        cols_per_sample=args.cols_per_sample, seq_len=args.seq_len, seed=1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    config = AegirConfig(
        arch_layout=["w2", ["w2", ["w4"], "w2"], "w2"],
        d_model=[128, 192, 192],
        d_intermediate=[0, 0, 0],
        vocab_size=260,
        ssm_cfg=SSMConfig(d_state=64, chunk_size=64),
        attn_cfg=AttnConfig(num_heads=[2, 3, 3], rotary_emb_dim=[8, 12, 12], window_size=[]),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=1,
        task_type="cta",
    )
    model = AegirForDED(config=config, proj_dim=128, device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"DED smoke: {len(train_ds)} train / {len(val_ds)} val "
          f"| {sum(p.numel() for p in model.parameters()):,} params | device={device}")

    val_loss0, val_f10 = evaluate(model, val_loader, device)
    print(f"  epoch 0 (untrained): val_loss={val_loss0:.3f} bcubed_f1={val_f10:.3f}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                role_ids=batch["role_ids"],
                cls_indexes=batch["cls_indexes"],
                mask=batch["mask"],
            )
            loss = supcon_loss(
                out.embeddings.float(), batch["cluster_ids"], out.col_mask
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss, val_f1 = evaluate(model, val_loader, device)
        print(
            f"  epoch {epoch+1}: train_loss={train_loss:.3f} "
            f"val_loss={val_loss:.3f} bcubed_f1={val_f1:.3f} "
            f"| {time.time()-t0:.1f}s"
        )


if __name__ == "__main__":
    main()

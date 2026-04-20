#!/usr/bin/env python3
"""Stage C — hierarchical fine-tuning from a pretrained backbone.

First pass of the path-prediction CTA: depth-2 hierarchical supervision.
Two classifier heads (Schema.org top-level parent + leaf class) on a
shared pooler. Fine-tune from a Stage B byte-level pretraining checkpoint.

The depth-2 formulation is intentionally the simplest version of the
"path prediction subsumes dual center loss" claim: the parent head is
an explicit coarse-cluster centroid, the leaf head is the fine-grained
centroid, and the joint loss enforces consistency via the shared
pooler. Full autoregressive path decode is Stage C+ (adds more depth;
same skeleton).

Usage:
    uv run --no-sync python scripts/train_hierarchical_cta.py \
        --pretrained outputs/pretrain/20260420T002455Z/final.pt \
        --task sotab --model-size small --epochs 3

Output at outputs/hierarchical/{run_id}/:
    metadata.json      — config + pretrained source + parent bucket map
    metrics.jsonl      — per-step loss, lr, per-head F1
    final.pt           — fine-tuned state dict
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegir.data.table_dataset import SotabCTADataset, TASK_NUM_CLASSES
from aegir.data.tokenizer import ByteTokenizer
from aegir.models.aegir import Aegir
from aegir.models.config import AegirConfig, AttnConfig, RWKVConfig, SSMConfig
from aegir.models.heads import _extract_at_indexes
from aegir.utils.train import group_params


log = logging.getLogger("train_hier_cta")


# ── Parent bucket map (lifted from sotab_diagnostic._PARENT_BUCKETS) ──


_PARENT_BUCKETS: dict[str, tuple[str, ...]] = {
    "Organization": (
        "Organization", "LocalBusiness", "Restaurant", "Hotel", "Airline",
        "Bank", "Bookstore", "ClothingStore", "GasStation", "MovieTheater",
        "Museum", "Library", "Hospital", "Pharmacy", "Dentist",
        "FoodEstablishment", "LodgingBusiness", "NGO", "Corporation",
        "EducationalOrganization", "CollegeOrUniversity", "School",
        "SportsTeam", "SportsClub", "SportsOrganization",
        "AutomotiveBusiness", "AutoDealer", "AutoRepair", "AutoRental",
        "FinancialService", "Store", "ShoppingCenter",
    ),
    "Place": (
        "Place", "City", "Country", "AdministrativeArea", "PostalAddress",
        "TouristAttraction", "Park", "CivicStructure", "Residence",
        "Accommodation", "Apartment", "House", "Beach", "Mountain",
        "River", "Lake", "Cemetery",
    ),
    "CreativeWork": (
        "CreativeWork", "Book", "Article", "BlogPosting", "NewsArticle",
        "Movie", "TVSeries", "TVEpisode", "VideoObject", "ImageObject",
        "MusicRecording", "MusicAlbum", "MusicGroup", "Recipe",
        "Episode", "Review", "Comment", "Dataset", "SoftwareApplication",
        "MobileApplication", "VideoGame", "Photograph", "PhotoAlbum",
        "AudioObject", "WebPage", "WebSite",
    ),
    "Product": (
        "Product", "IndividualProduct", "ProductModel", "SomeProducts",
        "Drug", "MedicalEntity", "FoodProduct", "Offer", "AggregateOffer",
        "Car", "Vehicle", "AutomotiveVehicle", "Boat",
    ),
    "Person": ("Person",),
    "Event": (
        "Event", "BusinessEvent", "ChildrensEvent", "ComedyEvent",
        "DanceEvent", "EducationEvent", "FoodEvent", "LiteraryEvent",
        "MusicEvent", "SocialEvent", "SportsEvent", "TheaterEvent",
        "VisualArtsEvent", "PublicationEvent",
    ),
    "Intangible": (
        "Intangible", "Rating", "AggregateRating", "Quantity", "Distance",
        "Duration", "Mass", "Energy", "ContactPoint", "Brand", "Enumeration",
        "StructuredValue", "PropertyValue", "MonetaryAmount", "PriceSpecification",
        "QuantitativeValue", "Role", "Service", "Language", "Audience",
    ),
    "Action": (
        "Action", "MoveAction", "TradeAction", "PlayAction", "AssessAction",
    ),
    "MedicalEntity": (
        "MedicalEntity", "AnatomicalStructure", "DrugClass", "MedicalCause",
        "MedicalCondition", "MedicalDevice", "MedicalGuideline",
        "MedicalIndication", "MedicalProcedure", "MedicalRiskFactor",
        "MedicalStudy", "MedicalTest", "MedicalTherapy",
    ),
}

_PARENT_NAMES = sorted(_PARENT_BUCKETS.keys()) + ["Thing"]  # Thing = catch-all
_PARENT_TO_IDX = {p: i for i, p in enumerate(_PARENT_NAMES)}


def _parent_of(leaf: str) -> str:
    for parent, members in _PARENT_BUCKETS.items():
        if leaf in members:
            return parent
    return "Thing"


# ── Model ───────────────────────────────────────────────────────


class AegirForHierarchicalCTA(nn.Module):
    """Two-head CTA classifier with shared pooler.

    Load-bearing design: the Stage B pretrained backbone provides
    stable byte-level representations; the new heads specialize that
    shared geometry into two consistent classifications (parent + leaf)
    that enforce hierarchical coherence via the single pooled vector.
    """

    def __init__(
        self,
        config: AegirConfig,
        num_leaf: int,
        num_parent: int,
        max_roles: int = 32,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d = config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, d, **factory_kwargs)
        self.role_embeddings = nn.Embedding(max_roles, d, **factory_kwargs)
        self.backbone = Aegir(config=config, stage_idx=0, **factory_kwargs)
        self.pooler = nn.Linear(d, d, **factory_kwargs)
        self.leaf_head = nn.Linear(d, num_leaf, **factory_kwargs)
        self.parent_head = nn.Linear(d, num_parent, **factory_kwargs)

    def forward(self, input_ids, role_ids, cls_indexes, mask=None, **mixer_kwargs):
        hidden = self.embeddings(input_ids) + self.role_embeddings(role_ids)
        B, L, D = hidden.shape
        if mask is None:
            hidden = hidden.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden.device)
        else:
            cu_seqlens = None
            max_seqlen = None
        hidden, bpred = self.backbone(
            hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            mask=mask, **mixer_kwargs,
        )
        hidden = hidden.view(B, L, D)
        pooled = _extract_at_indexes(hidden, cls_indexes)
        pooled = torch.tanh(self.pooler(pooled))
        leaf_logits = self.leaf_head(pooled)
        parent_logits = self.parent_head(pooled)
        return leaf_logits, parent_logits, bpred, pooled


def _small_config(vocab_size: int = 260) -> AegirConfig:
    return AegirConfig(
        arch_layout=["w4", ["w4", ["w8"], "w4"], "w4"],
        d_model=[256, 384, 384],
        d_intermediate=[0, 0, 0],
        vocab_size=vocab_size,
        ssm_cfg=SSMConfig(),
        attn_cfg=AttnConfig(num_heads=[4, 6, 6], rotary_emb_dim=[16, 24, 24], window_size=[]),
        rwkv_cfg=RWKVConfig(head_size=64, rosa_mode="1bit"),
        num_labels=1,
        task_type="cta",
    )


def _load_pretrained(model: AegirForHierarchicalCTA, ckpt_path: Path) -> dict:
    """Load Stage B backbone + byte embeddings. Return stats about what loaded."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    own_state = model.state_dict()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    for k, v in state.items():
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k].copy_(v)
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)
    model.load_state_dict(own_state)
    return {
        "loaded": len(loaded_keys),
        "skipped": len(skipped_keys),
        "total_pretrained": len(state),
        "own_total": len(own_state),
        "first_skipped": skipped_keys[:5],
    }


# ── Training ────────────────────────────────────────────────────


def _cosine_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    import math
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _f1_scores(preds: list[int], labels: list[int], num_classes: int) -> dict:
    # Standard micro / macro
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for p, l in zip(preds, labels):
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / max(1, total_tp + total_fp)
    micro_r = total_tp / max(1, total_tp + total_fn)
    micro_f1 = 2 * micro_p * micro_r / max(1e-9, micro_p + micro_r)

    f1s = []
    for c in range(num_classes):
        p_c = tp[c] / max(1, tp[c] + fp[c])
        r_c = tp[c] / max(1, tp[c] + fn[c])
        f1_c = 2 * p_c * r_c / max(1e-9, p_c + r_c) if (p_c + r_c) > 0 else 0.0
        f1s.append(f1_c)
    macro_f1 = sum(f1s) / max(1, len(f1s))

    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "accuracy": total_tp / max(1, len(preds))}


@torch.no_grad()
def _evaluate(model, loader, device, leaf_to_parent_arr) -> dict:
    model.eval()
    leaf_preds, leaf_labels = [], []
    parent_preds, parent_labels = [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        cls_idx = batch["cls_indexes"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        parent_gt = leaf_to_parent_arr[labels.cpu()].to(device)

        leaf_logits, parent_logits, _, _ = model(
            input_ids=input_ids, role_ids=role_ids,
            cls_indexes=cls_idx, mask=mask,
        )
        loss_l = F.cross_entropy(leaf_logits.float(), labels)
        loss_p = F.cross_entropy(parent_logits.float(), parent_gt)
        total_loss += (loss_l + loss_p).item() * labels.shape[0]
        n += labels.shape[0]

        leaf_preds.extend(leaf_logits.argmax(-1).cpu().tolist())
        leaf_labels.extend(labels.cpu().tolist())
        parent_preds.extend(parent_logits.argmax(-1).cpu().tolist())
        parent_labels.extend(parent_gt.cpu().tolist())

    model.train()
    return {
        "val_loss": total_loss / max(1, n),
        "leaf": _f1_scores(leaf_preds, leaf_labels, model.leaf_head.out_features),
        "parent": _f1_scores(parent_preds, parent_labels, model.parent_head.out_features),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained", type=Path, required=True)
    ap.add_argument("--sotab-dir", type=Path, default=Path("/raid/datasets/sotab"))
    ap.add_argument("--model-size", default="small", choices=["small"])
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-context-cols", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--parent-loss-weight", type=float, default=0.5,
                    help="α in L = L_leaf + α · L_parent")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=4649)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/hierarchical"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    out_dir = args.out_dir / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    # Data
    tok = ByteTokenizer()
    log.info("Loading SOTAB train+val splits...")
    train_ds = SotabCTADataset(
        data_dir=args.sotab_dir, split="train", tokenizer=tok,
        max_length=args.max_length, max_context_cols=args.max_context_cols,
    )
    val_ds = SotabCTADataset(
        data_dir=args.sotab_dir, split="val", tokenizer=tok,
        max_length=args.max_length, max_context_cols=args.max_context_cols,
    )
    if args.max_train_samples and len(train_ds) > args.max_train_samples:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(args.max_train_samples)))
    if args.max_val_samples and len(val_ds) > args.max_val_samples:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, list(range(args.max_val_samples)))
    log.info("  train %d val %d", len(train_ds), len(val_ds))

    # Resolve leaf→parent mapping using the discovered label vocab.
    vocab = SotabCTADataset._label_vocab_cache.get(
        ("CTA", "schemaorg", str(args.sotab_dir))
    )
    if vocab is None:
        raise RuntimeError("SOTAB label vocab not resolved — dataset init bug")
    leaf_names = [""] * TASK_NUM_CLASSES["sotab"]
    for name, idx in vocab.items():
        if 0 <= idx < len(leaf_names):
            leaf_names[idx] = name
    leaf_to_parent_idx = torch.tensor(
        [_PARENT_TO_IDX[_parent_of(ln)] for ln in leaf_names], dtype=torch.long,
    )
    parent_dist = Counter(_parent_of(ln) for ln in leaf_names)
    log.info("Parent-bucket distribution: %s", dict(parent_dist))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_aegir_train", str(Path(__file__).parent.parent / "train.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import train.py")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    collate_fn = train_mod.collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # Model + load pretrained
    num_leaf = TASK_NUM_CLASSES["sotab"]
    num_parent = len(_PARENT_NAMES)
    config = _small_config(vocab_size=260)
    model = AegirForHierarchicalCTA(
        config=config, num_leaf=num_leaf, num_parent=num_parent,
        max_roles=32, device=device, dtype=dtype,
    )
    stats = _load_pretrained(model, args.pretrained)
    log.info(
        "Loaded %d/%d tensors from %s (skipped %d, first skipped: %s)",
        stats["loaded"], stats["total_pretrained"], args.pretrained,
        stats["skipped"], stats["first_skipped"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model params: %s (leaf=%d parent=%d)", f"{n_params:,}", num_leaf, num_parent)

    param_groups = group_params(model)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = max(100, (len(train_loader) * args.epochs))
    log.info(
        "total_steps=%d warmup=%d eval_every=%d",
        total_steps, args.warmup_steps, args.eval_every,
    )

    metadata = {
        "model_size": args.model_size,
        "num_leaf": num_leaf,
        "num_parent": num_parent,
        "parent_names": _PARENT_NAMES,
        "pretrained": str(args.pretrained),
        "pretrained_load_stats": stats,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "parent_loss_weight": args.parent_loss_weight,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "total_steps_planned": total_steps,
        "cmdline": sys.argv,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    metrics_path = out_dir / "metrics.jsonl"
    best_leaf_macro = -1.0
    best_path = out_dir / "best.pt"

    step = 0
    model.train()
    t0 = time.time()
    loss_acc = 0.0
    loss_l_acc = 0.0
    loss_p_acc = 0.0
    samples_acc = 0

    for epoch in range(args.epochs):
        log.info("=== epoch %d/%d ===", epoch + 1, args.epochs)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            role_ids = batch["role_ids"].to(device, non_blocking=True)
            cls_idx = batch["cls_indexes"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            parent_gt = leaf_to_parent_idx[labels.cpu()].to(device)

            leaf_logits, parent_logits, _, _ = model(
                input_ids=input_ids, role_ids=role_ids,
                cls_indexes=cls_idx, mask=mask,
            )
            loss_l = F.cross_entropy(leaf_logits.float(), labels)
            loss_p = F.cross_entropy(parent_logits.float(), parent_gt)
            loss = loss_l + args.parent_loss_weight * loss_p
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            lr_scale = _cosine_warmup(step, args.warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            loss_acc += loss.item()
            loss_l_acc += loss_l.item()
            loss_p_acc += loss_p.item()
            samples_acc += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                log.info(
                    "step %6d  loss %.4f  (leaf %.4f  parent %.4f)  lr %.2e  %.1fs",
                    step, loss_acc / samples_acc, loss_l_acc / samples_acc,
                    loss_p_acc / samples_acc, args.lr * lr_scale, elapsed,
                )
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "loss": loss_acc / samples_acc,
                        "loss_leaf": loss_l_acc / samples_acc,
                        "loss_parent": loss_p_acc / samples_acc,
                        "lr": args.lr * lr_scale,
                        "epoch": epoch + 1,
                    }) + "\n")
                loss_acc = loss_l_acc = loss_p_acc = 0.0
                samples_acc = 0
                t0 = time.time()

            if step % args.eval_every == 0:
                log.info("Running val eval at step %d...", step)
                ev = _evaluate(model, val_loader, device, leaf_to_parent_idx)
                log.info(
                    "EVAL step %d  val_loss %.4f  leaf F1 %.4f/%.4f  parent F1 %.4f/%.4f  parent_acc %.4f",
                    step, ev["val_loss"],
                    ev["leaf"]["micro_f1"], ev["leaf"]["macro_f1"],
                    ev["parent"]["micro_f1"], ev["parent"]["macro_f1"],
                    ev["parent"]["accuracy"],
                )
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "eval": ev, "epoch": epoch + 1,
                    }) + "\n")
                if ev["leaf"]["macro_f1"] > best_leaf_macro:
                    best_leaf_macro = ev["leaf"]["macro_f1"]
                    torch.save(model.state_dict(), best_path)
                    log.info("  saved best (leaf macro F1 %.4f)", best_leaf_macro)

    # End-of-training eval
    log.info("Final val eval...")
    ev = _evaluate(model, val_loader, device, leaf_to_parent_idx)
    log.info(
        "FINAL  leaf F1 %.4f/%.4f  parent F1 %.4f/%.4f  parent_acc %.4f",
        ev["leaf"]["micro_f1"], ev["leaf"]["macro_f1"],
        ev["parent"]["micro_f1"], ev["parent"]["macro_f1"],
        ev["parent"]["accuracy"],
    )
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"step": step, "eval_final": ev}) + "\n")
    torch.save(model.state_dict(), out_dir / "final.pt")
    if ev["leaf"]["macro_f1"] > best_leaf_macro:
        torch.save(model.state_dict(), best_path)
    log.info("Done. Best leaf macro F1: %.4f", max(best_leaf_macro, ev["leaf"]["macro_f1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

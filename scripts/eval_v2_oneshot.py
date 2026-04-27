#!/usr/bin/env python3
"""One-shot stratified eval of a final.pt against the eval manifest.

Used to fill in slices that weren't in the manifest at training start
(e.g. FinePDFs-lab, which was still downloading when v2 launched).

Usage:
    uv run --no-sync python scripts/eval_v2_oneshot.py \
        --ckpt outputs/mixed-v2/20260426T232240Z/final.pt \
        --eval-manifest /raid/datasets/aegir-corpus-v1/MIXTURE_eval.json \
        --slices eval.finepdfs-lab-held
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegir.models.heads import AegirForCausalLM  # noqa: E402

# Import helpers from pretrain_mixed
sys.path.insert(0, str(Path(__file__).parent))
from pretrain_mixed import _repro_config, _eval_slice  # noqa: E402

log = logging.getLogger("eval_v2_oneshot")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--eval-manifest", type=Path, required=True)
    ap.add_argument("--slices", nargs="*", default=None,
                    help="Slice names to evaluate (default: all in manifest)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-bytes", type=int, default=16_000_000)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--vocab-size", type=int, default=260)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = _repro_config(
        vocab_size=args.vocab_size, n_layer=args.n_layer, d_model=args.d_model,
    )
    model = AegirForCausalLM(config=config, device=device, dtype=dtype)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    log.info("loaded %s on %s", args.ckpt, device)

    eval_manifest = json.loads(args.eval_manifest.read_text())
    slices = eval_manifest["slices"]
    if args.slices:
        slices = [s for s in slices if s["name"] in args.slices]
    log.info("eval %d slice(s)", len(slices))

    results = {}
    for slc in slices:
        nats, seen = _eval_slice(
            model, list(slc["files"]), args.seq_len, args.eval_bytes,
            args.batch_size, device,
        )
        bpb = nats / math.log(2)
        results[slc["name"]] = bpb
        log.info("  %-26s  %.4f bits/byte  (%.1f MB seen)",
                 slc["name"], bpb, seen / 2**20)

    out_path = args.ckpt.parent / "metrics_eval_oneshot.jsonl"
    with open(out_path, "a") as f:
        for name, bpb in results.items():
            f.write(json.dumps({
                "ckpt": str(args.ckpt), "slice": name,
                "bits_per_byte": bpb,
            }) + "\n")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

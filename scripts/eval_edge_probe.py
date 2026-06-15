#!/usr/bin/env python
"""E2 — relational edge-probe instrument (EVIDENCE.md).

Two frozen-backbone linear probes over the corpus's verifiable JSON tables, both
CELLS-ONLY (values only; the corpus already uses positional headers id/x/y, so
there is nothing to leak) and both scored against a Hewitt-Liang control task:

  (a) column → role/SKOS-code (pk / designation / fk-ref).  [secondary]
  (b) FK-pair validity — do (col_A, col_B) stand in a sanctioned FK relation?
      positives = realized spine FK edges (fk-col → referenced pk-col, value-
      overlap ≥ 0.5); HARD negatives = type-compatible id-columns NOT FK-linked
      (fk→wrong-pk, pk→pk). Forces value-overlap signal, not type alone.  [PRIMARY]

Selectivity = real − control (random-permuted labels). PR-AUC for (b). Bootstrap
CIs. Train/test split BY CHAPTER (no leakage). Validity rule: on a known-good
backbone, selectivity > 0 with a CI-clean margin on both heads.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from aegir.data.serialization import serialize_table  # noqa: E402
from aegir.data.tokenizer import ByteTokenizer  # noqa: E402
from aegir.models.heads import AegirForColumnAnnotation  # noqa: E402
from eval_cells_cta import tiny_config, load_backbone, bootstrap_ci  # noqa: E402

JSON_FENCE = re.compile(r"```json\s*(.*?)```", re.S)
DEFAULT_RUNS = [
    "/raid/checkpoints/aegir-artifacts/sdg_corpus_v0_3/811408b392859708",
    "/raid/checkpoints/aegir-artifacts/sdg_corpus_v0_3/d7646714bdd5e16f",
]


def extract(chapters: list[dict], max_chapters: int):
    """Parse JSON fences → column records (with positional/FK role) + FK pairs."""
    cols: list[dict] = []
    pairs: list[tuple[int, int, int]] = []
    ch_used = 0
    for ci_ch, ch in enumerate(chapters[:max_chapters]):
        m = JSON_FENCE.search(ch.get("response_text") or "")
        if not m:
            continue
        try:
            tables = [t for t in json.loads(m.group(1)).get("tables", []) if t.get("rows")]
        except Exception:
            continue
        if len(tables) < 2:
            continue
        pk_set = {t["name"]: {str(r[0]) for r in t["rows"] if r} for t in tables}
        local = []
        for t in tables:
            rows = t["rows"]
            ncol = max(len(r) for r in rows)
            for ck in range(ncol):
                vals = [str(r[ck]) for r in rows if len(r) > ck and r[ck] is not None]
                if not vals:
                    continue
                rec = {"gid": len(cols), "ch": ci_ch, "tname": t["name"], "ci": ck,
                       "vals": vals, "is_pk": ck == 0, "role": "pk" if ck == 0 else "label",
                       "fk_target": None}
                if ck != 0:  # FK = non-pk col whose values mostly land in a SIBLING pk set
                    best, bf = None, 0.0
                    for st in tables:
                        if st["name"] == t["name"]:
                            continue
                        ov = sum(1 for v in vals if v in pk_set[st["name"]]) / len(vals)
                        if ov > bf:
                            bf, best = ov, st["name"]
                    if bf >= 0.5:
                        rec["role"], rec["fk_target"] = "fk", best
                cols.append(rec)
                local.append(rec)
        # pairs (type-compatible id columns only)
        pkcol = {t["name"]: next((c for c in local if c["tname"] == t["name"] and c["is_pk"]), None)
                 for t in tables}
        for fk in [c for c in local if c["role"] == "fk"]:
            tgt = pkcol.get(fk["fk_target"])
            if tgt:
                pairs.append((fk["gid"], tgt["gid"], 1))                       # positive
            others = [pkcol[n] for n in pkcol if n not in (fk["fk_target"], fk["tname"]) and pkcol[n]]
            if others:
                pairs.append((fk["gid"], others[0]["gid"], 0))                 # hard neg: wrong pk
        pks = [c for c in local if c["is_pk"]]
        for i in range(len(pks) - 1):
            pairs.append((pks[i]["gid"], pks[i + 1]["gid"], 0))                # hard neg: pk vs pk
        ch_used += 1
    return cols, pairs, ch_used


@torch.no_grad()
def featurize_columns(model, cols, tok, device, max_length=256, batch=64) -> np.ndarray:
    sers = []
    for c in cols:
        st = serialize_table(c["vals"], "", [], [], tok, max_length=max_length)
        sers.append((st.token_ids, st.role_ids))
    out = []
    for s in range(0, len(sers), batch):
        chunk = sers[s:s + batch]
        L = max(len(t) for t, _ in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        rls = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros(len(chunk), L, dtype=torch.bool)
        for i, (t, r) in enumerate(chunk):
            n = len(t)
            ids[i, :n] = torch.tensor(t)
            rls[i, :n] = torch.tensor(r)
            mask[i, :n] = True
        ids, rls, mask = ids.to(device), rls.to(device), mask.to(device)
        h = model.embeddings(ids) + model.role_embeddings(rls)
        hs, _ = model.backbone(h, cu_seqlens=None, max_seqlen=None, mask=mask)
        hs = hs.view(len(chunk), L, -1).float()
        m = mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out)


def _probe(Xtr, ytr, Xte, yte, seeds, binary):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    real_acc, ctrl_acc, pr, correct = [], [], [], None
    for sd in seeds:
        rng = np.random.default_rng(sd)
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        real_acc.append(float((pred == yte).mean()))
        if correct is None:
            correct = (pred == yte).astype(int)
        if binary:
            classes = list(clf.classes_)
            proba = (clf.predict_proba(Xte)[:, classes.index(1)]
                     if 1 in classes else np.zeros(len(yte)))
            pr.append(float(average_precision_score(yte, proba)))
        cc = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        cc.fit(Xtr, rng.permutation(ytr))
        ctrl_acc.append(float((cc.predict(Xte) == yte).mean()))
    lo, hi = bootstrap_ci(correct)
    sel = np.array(real_acc) - np.array(ctrl_acc)
    return {"acc": float(np.mean(real_acc)), "acc_ci": [lo, hi],
            "control_acc": float(np.mean(ctrl_acc)),
            "selectivity": float(sel.mean()), "selectivity_min": float(sel.min()),
            "pr_auc": float(np.mean(pr)) if pr else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained", default="/raid/checkpoints/aegir-artifacts/ablation_v1_ckpts/full/runs/20260605T032856Z_952013c6ca_tiny_pretrain/best_model.pt")
    ap.add_argument("--arm", default="full")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--max-chapters", type=int, default=800)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/raid/checkpoints/aegir-artifacts/evidence/e2/edge_probe.json")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    chapters = []
    for run in args.runs:
        p = Path(run) / "chapters.parquet"
        if p.exists():
            chapters.extend(pq.read_table(p, columns=["response_text"]).to_pylist())
    cols, pairs, ch_used = extract(chapters, args.max_chapters)
    roles = {}
    for c in cols:
        roles[c["role"]] = roles.get(c["role"], 0) + 1
    n_pos = sum(p[2] for p in pairs)
    print(f"chapters_used={ch_used}  columns={len(cols)} {roles}  "
          f"fk_pairs={len(pairs)} (pos={n_pos}, neg={len(pairs)-n_pos})")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    tok = ByteTokenizer()
    vocab_size = ByteTokenizer.vocab_size
    if args.pretrained:
        peek = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        if "embeddings.weight" in peek:
            vocab_size = peek["embeddings.weight"].shape[0]
        del peek
    torch.manual_seed(4649)
    model = AegirForColumnAnnotation(tiny_config(vocab_size, 8)).to(device=device, dtype=dtype)
    if args.pretrained:
        print(f"backbone[{args.arm}]: {load_backbone(model, args.pretrained)}")
    else:
        print("RANDOM-INIT floor")
    model.eval()

    print("featurizing columns (frozen backbone, cells-only)...")
    X = featurize_columns(model, cols, tok, device, args.max_length)

    # chapter-level split (no leakage)
    rng = np.random.default_rng(0)
    ch_ids = sorted({c["ch"] for c in cols})
    te_ch = set(rng.choice(ch_ids, size=max(1, int(len(ch_ids) * args.test_frac)), replace=False).tolist())

    def split_mask(idxs_ch):
        tr = np.array([i for i, ch in enumerate(idxs_ch) if ch not in te_ch])
        te = np.array([i for i, ch in enumerate(idxs_ch) if ch in te_ch])
        return tr, te

    out = {"arm": args.arm, "pretrained": args.pretrained, "n_cols": len(cols),
           "n_pairs": len(pairs), "roles": roles}

    # E2(a) column → role/SKOS-code
    role2id = {"pk": 0, "label": 1, "fk": 2}
    a_idx = [i for i, c in enumerate(cols) if c["role"] in role2id]
    ya = np.array([role2id[cols[i]["role"]] for i in a_idx])
    cha = [cols[i]["ch"] for i in a_idx]
    Xa = X[a_idx]
    tr, te = split_mask(cha)
    out["E2a_role"] = _probe(Xa[tr], ya[tr], Xa[te], ya[te], args.seeds, binary=False)
    print(f"E2(a) role/SKOS: acc={out['E2a_role']['acc']:.3f} ctrl={out['E2a_role']['control_acc']:.3f} "
          f"sel={out['E2a_role']['selectivity']:.3f} (min {out['E2a_role']['selectivity_min']:.3f})")

    # E2(b) FK-pair validity — pair features from column vectors
    pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])
    yb = np.array([p[2] for p in pairs])
    ra, rb = X[pa], X[pb]
    Xb = np.concatenate([ra, rb, ra * rb, np.abs(ra - rb)], axis=1)
    chb = [cols[p[0]]["ch"] for p in pairs]
    trb, teb = split_mask(chb)
    out["E2b_fk"] = _probe(Xb[trb], yb[trb], Xb[teb], yb[teb], args.seeds, binary=True)
    r = out["E2b_fk"]
    print(f"E2(b) FK-validity: acc={r['acc']:.3f} ctrl={r['control_acc']:.3f} "
          f"sel={r['selectivity']:.3f} (min {r['selectivity_min']:.3f}) PR-AUC={r['pr_auc']:.3f}")

    valid = (out["E2a_role"]["selectivity_min"] > 0) and (out["E2b_fk"]["selectivity_min"] > 0)
    out["instrument_valid"] = bool(valid)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nINSTRUMENT VALID (selectivity>0 both heads): {valid}  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

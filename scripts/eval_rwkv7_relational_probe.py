#!/usr/bin/env python
"""C1 — RWKV-7 held-out relational CTA probe (Path A (c): the binding generalization eval).

Frozen RWKV-7 backbone → per-column MEAN-POOLED hidden over **cells-only** text (World-tokenized; no header
shortcut) → linear probe predicts the column's ontology CONCEPT (top-K). Reports Hewitt&Liang control-task
SELECTIVITY (acc − permuted-label acc), sample-efficiency shots, and bootstrap CIs — reusing
``eval_cells_cta.probe_once`` / ``bootstrap_ci`` (the proven probing methodology). The probe predicts a column's
concept from its VALUES, so it is sensitive *by construction* to corpus value-richness — this is the metric the
C2 value-refinement is meant to move. At today's trivial values the baseline is expected ~weak (the floor);
that's the point. Compares arms (baseline / control / treatment …) on the SAME columns + split.

    LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs uv run --no-sync python scripts/eval_rwkv7_relational_probe.py \
        --arms baseline=/raid/datasets/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
               control=/raid/build/aegir/path-a/out/control/final.pth \
               treatment=/raid/build/aegir/path-a/out/a02/final.pth \
        --realize --topk 60 --out /raid/build/aegir/path-a/relprobe.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("RWKV_HEAD_SIZE", "64")
os.environ.setdefault("RWKV_MY_TESTING", "x070")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import eval_rwkv7_baseline as E  # noqa: E402 — RWKV-7 model + tokenizer + detect_config (fla forward)
from eval_cells_cta import bootstrap_ci  # noqa: E402 — reuse the CI (probe_once's max_iter=2000 too slow here)

_STOP = {"id", "pk", "fk", "code", "num", "no", "key", "ref", "uid", "guid", "name", "label", "value", "val"}


def _concept(colname: str) -> str:
    toks = [t for t in re.split(r"[^a-z0-9]+", colname.lower()) if t and t not in _STOP and not t.isdigit()]
    return "_".join(toks) or "misc"


def _load_hypernyms() -> dict:
    """norm(concept) → norm(parent) from the SKOS vocab's skos:broader (the ontology hypernym graph).
    Lets C1 label columns by their PARENT concept — a hypernym-CTA (infer the general type from a
    heterogeneous value population), which is the generalizable target and the de-leaked label for
    subtree-mixing (C2.5). One level up; missing → leaf concept (identity)."""
    try:
        from rdflib import Graph
        from rdflib.namespace import SKOS
    except Exception:  # noqa: BLE001
        return {}
    ttl = REPO / "corpora" / "vocabulary" / "vocabulary.ttl"
    if not ttl.exists():
        return {}
    g = Graph()
    g.parse(str(ttl), format="turtle")
    pref = {s: str(o) for s, _, o in g.triples((None, SKOS.prefLabel, None))}
    out: dict[str, str] = {}
    for s, _, o in g.triples((None, SKOS.broader, None)):
        if s in pref and o in pref:
            out[_concept(pref[s])] = _concept(pref[o])
    return out


def _load_domain_taxonomy(path) -> dict:
    """member-concept → hypernym from the HermiT-admitted domain taxonomy (a domain child→parent broader map
    that replaces the BFO-structural SKOS broaders for hypernym-labeling + subtree-mixing)."""
    from aegir.ontology.subtree_mix import concept_of
    d = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    for t in d.get("admitted", []):
        for m in t["members"]:
            out[concept_of(m)] = t["hypernym"]
    return out


def build_dataset(per_family: int, realize: bool, topk: int, max_cells: int, min_cells: int,
                  hypernym_map=None, mix=False):
    """In-process spine → RI-safe materialized rows → (cells-only text, concept) per non-PK column; top-K.
    With ``hypernym_map``, the label is the column concept's PARENT (SKOS broader) — a hypernym-CTA. With
    ``mix`` (C2.5 L1), parent-typed columns draw from the union of their children's pools (subtree mixing)."""
    from build_ddl_spine import catalog_files, load_spine
    from aegir.ontology.chapter_tables import definitions_for_spine, entity_pools_for_spine
    from aegir.ontology.rows import materialize_rows
    files = catalog_files(REPO / "src/aegir/ontology/catalog")
    spine, fks, *_ = load_spine(files, per_family if per_family > 0 else None, realize=realize)
    pools = entity_pools_for_spine(spine)
    if mix and hypernym_map:
        from aegir.ontology.subtree_mix import subtree_mixed_pools
        pools, n_mixed = subtree_mixed_pools(pools, hypernym_map)
        print(f"subtree-mix: {n_mixed} parent-typed columns mixed with children's pools", flush=True)
    materialize_rows(spine, fks, seed=123, definitions=definitions_for_spine(spine), entity_pools=pools)
    recs: list[tuple[str, str]] = []
    for st in spine:
        cols, rows = st.table.columns, st.table.rows
        if not rows:
            continue
        for ci, c in enumerate(cols):
            if getattr(c, "pk", False) or c.slot_ref == "__pk__":
                continue
            cells = [str(r[ci]).strip() for r in rows if ci < len(r) and str(r[ci]).strip()]
            if len(cells) < min_cells:
                continue
            concept = _concept(c.name)
            if hypernym_map:
                concept = hypernym_map.get(concept, concept)  # label by parent concept (hypernym CTA)
            recs.append((" | ".join(cells[:max_cells]), concept))  # cells-only (no header)
    keep = {c for c, _ in collections.Counter(c for _, c in recs).most_common(topk)}
    recs = [(t, c) for t, c in recs if c in keep]
    labels = sorted({c for _, c in recs})
    lab2i = {c: i for i, c in enumerate(labels)}
    texts = [t for t, _ in recs]
    y = np.array([lab2i[c] for _, c in recs], dtype=np.int64)
    return texts, y, labels


@torch.no_grad()
def featurize(model, texts, tok, device, max_len, batch):
    """Frozen RWKV-7 backbone (emb→blocks→ln_out, no head) → mean-pooled hidden per text.

    Pads to ONE global length so the fla/triton kernel compiles once (variable per-batch length forces a
    recompile every shape — pathologically slow)."""
    all_ids = [(tok.encodeBytes(t.encode("utf-8"))[:max_len] or [0]) for t in texts]
    L = ((max(len(s) for s in all_ids) + 63) // 64) * 64  # chunk-align: a non-(T%64) length hangs the fla kernel
    feats = []
    for i in range(0, len(all_ids), batch):
        ids = all_ids[i:i + batch]
        x = torch.zeros(len(ids), L, dtype=torch.long, device=device)
        m = torch.zeros(len(ids), L, 1, device=device)
        for j, s in enumerate(ids):
            x[j, :len(s)] = torch.tensor(s, device=device)
            m[j, :len(s), 0] = 1.0
        h = model.emb(x)
        v_first = torch.empty_like(h)
        for blk in model.blocks:
            h, v_first = blk(h, v_first)
        h = model.ln_out(h).float()
        feats.append(((h * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu().numpy())
    return np.concatenate(feats, 0)


def featurize_arm(ckpt, texts, tok, device, max_len, batch):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = E.detect_config(state)
    model = E.RWKV_x070(cfg).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(state, strict=False)
    model.eval()
    X = featurize(model, texts, tok, device, max_len, batch)
    del model
    torch.cuda.empty_cache()
    return X


def _probe(Xtr, ytr, Xte, yte, seed, control=False):
    """Fast capped-iter linear probe (acc + macro-f1). Hewitt&Liang control = permuted train labels.
    Scaled features (see probe_arm) converge well under 300 iters — vs the shared probe_once's max_iter=2000."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    y = np.random.default_rng(seed).permutation(ytr) if control else ytr
    # saga (not lbfgs): multinomial lbfgs on tens of classes is pathologically slow here (~100s/fit);
    # saga on scaled features matches its accuracy at ~2s.
    clf = LogisticRegression(solver="saga", max_iter=150, C=1.0, n_jobs=-1).fit(Xtr, y)
    pred = clf.classes_[clf.predict_proba(Xte).argmax(1)]
    return {"acc": float((pred == yte).mean()),
            "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0))}, (pred == yte)


def probe_arm(X, y, n_labels, shots, seeds, test_frac, split_seed):
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(y))
    n_te = max(1, int(test_frac * len(y)))
    te, trpool = idx[:n_te], idx[n_te:]
    X = StandardScaler().fit(X[trpool]).transform(X)  # scale (fit on train pool) → fast lbfgs convergence
    Xte, yte = X[te], y[te]
    out = {}
    for sh in shots:
        accs, sels, f1s, last_correct = [], [], [], None
        for sd in seeds:
            r = np.random.default_rng(1000 + sd)
            tr = trpool if sh == 0 else r.choice(trpool, min(sh, len(trpool)), replace=False)
            real, correct = _probe(X[tr], y[tr], Xte, yte, sd, control=False)
            ctrl, _ = _probe(X[tr], y[tr], Xte, yte, sd, control=True)
            accs.append(real["acc"])
            f1s.append(real["macro_f1"])
            sels.append(real["acc"] - ctrl["acc"])
            last_correct = correct
        lo, hi = bootstrap_ci(last_correct)
        out["full" if sh == 0 else str(sh)] = {"acc": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)),
                                               "selectivity": float(np.mean(sels)), "acc_ci95": [lo, hi]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=ckpt.pth ...")
    ap.add_argument("--tokenizer", default="/home/rch/local/src/oss/rwkv-lm/RWKV-v7/rwkv_vocab_v20230424.txt")
    ap.add_argument("--per-family", type=int, default=0, help="cap templates/family (0=all)")
    ap.add_argument("--realize", action="store_true", help="stochastic schema realization (richer tables)")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--max-cells", type=int, default=24)
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shots", type=int, nargs="+", default=[64, 256, 1024, 0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--hypernym", action="store_true", help="label by PARENT concept (SKOS broader) — hypernym CTA")
    ap.add_argument("--mix", action="store_true", help="C2.5 subtree value-mixing (parent cols ← children's pools)")
    ap.add_argument("--domain-taxonomy", default="",
                    help="admitted domain-taxonomy json → domain broader map (replaces SKOS broader for --hypernym/--mix)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"

    if a.domain_taxonomy:
        hmap = _load_domain_taxonomy(a.domain_taxonomy)
        print(f"domain taxonomy: {len(hmap)} member→hypernym edges", flush=True)
    elif a.hypernym or a.mix:
        hmap = _load_hypernyms()
        print(f"hypernym map: {len(hmap)} concept→parent edges (SKOS broader)", flush=True)
    else:
        hmap = None
    texts, y, labels = build_dataset(a.per_family, a.realize, a.topk, a.max_cells, a.min_cells,
                                     (hmap if a.hypernym else None), mix=a.mix)
    print(f"dataset: {len(texts)} columns · {len(labels)} concept-labels (top-{a.topk}) · "
          f"majority={collections.Counter(y.tolist()).most_common(1)[0][1] / max(1, len(y)):.3f}", flush=True)
    tok = E.RWKV_TOKENIZER(a.tokenizer)
    results = {"n_columns": len(texts), "n_labels": len(labels), "shots": a.shots, "hypernym": bool(a.hypernym),
               "majority_frac": collections.Counter(y.tolist()).most_common(1)[0][1] / max(1, len(y)), "arms": {}}
    for spec in a.arms:
        name, ckpt = spec.split("=", 1)
        X = featurize_arm(ckpt, texts, tok, dev, a.max_len, a.batch)
        results["arms"][name] = probe_arm(X, y, len(labels), a.shots, a.seeds, a.test_frac, 0)
        f = results["arms"][name].get("full", {})
        print(f"  {name}: acc={f.get('acc', 0):.3f} sel={f.get('selectivity', 0):.3f} "
              f"macro_f1={f.get('macro_f1', 0):.3f} (full-shot)", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"DONE → {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

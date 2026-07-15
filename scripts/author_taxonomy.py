#!/usr/bin/env python
"""author_taxonomy — fire the differentia-authoring CAS loop across all induced genera.

Genus induction (calibrated to the discriminating-potential frontier) → for each genus: name it + author each
member's OBSERVABLE, value-enriched differentia (per-species structural membranes) → the HermiT membrane over
the assembled genus (consistency + distinctness) → re-prompt the species it flags. Streams each authenticated
genus to <out>/genera/<i>.json as it completes (resumable: a crash at genus N keeps N-1). The methods are the
durable component strategy/components/methods/taxonomy_authentication.json — this is its runner, ready to lift
into a Metaflow foreach step.

    env $(cat build/cuda-driver-libs/.env|xargs) uv run --no-sync python scripts/author_taxonomy.py \
        --entities <run>/entities --members-per-genus 15 --out build/taxonomy/<tag>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.differentia_authoring import (author_genus, load_enum_values,  # noqa: E402
                                                  name_genus, to_manchester)
from aegir.ontology.genus_induction import embed, induce, load_specs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--k", type=int, default=0, help="genus count (0 = frontier_k from the induction)")
    ap.add_argument("--members-per-genus", type=int, default=15, help="most-central members authored per genus")
    ap.add_argument("--out", default=str(REPO / "build" / "taxonomy" / "run"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "genera").mkdir(parents=True, exist_ok=True)

    specs = load_specs(Path(a.entities))
    names = sorted(specs)
    enums = load_enum_values(Path(a.entities))
    X = embed([specs[n] for n in names])
    from aegir.ontology.genus_induction import discriminating_potential
    ks = [k for k in (8, 16, 32, 64, 128, 256, 512) if k < len(names)]
    k = a.k or discriminating_potential(X, ks)["frontier_k"]
    labels, centers = induce(X, k)
    print(f"lexicon {len(names)} · {k} genera · value-enriched species {sum(1 for n in names if enums.get(n))}",
          flush=True)

    n_authored = n_species = n_hermit_ok = 0
    for g in range(k):
        gp = out / "genera" / f"{g:03d}.json"
        if gp.exists():                                   # resume
            rec = json.loads(gp.read_text())
            n_authored += rec["accepted"]; n_species += rec["n_members"]; n_hermit_ok += int(rec["hermit"]["consistent"])
            continue
        idx = np.where(labels == g)[0]
        core = idx[np.argsort(-(X[idx] @ centers[g]))][: a.members_per_genus]
        members = [(names[i], specs[names[i]]) for i in core]
        glabel = (name_genus([m[0] for m in members]) or {}).get("label") or names[core[0]]
        diffs = author_genus(glabel, members, enum_values=enums, hermit=True, genus_rounds=2)
        h = diffs.get("__hermit__")
        axioms = to_manchester(glabel, diffs)
        acc = sum(1 for s, d in diffs.items() if d.accepted and s != "__hermit__")
        rec = {"genus": glabel, "n_members": len(members), "accepted": acc,
               "hermit": {"consistent": bool(h.accepted) if h else None, "reason": h.reason if h else ""},
               "differentiae": {s: {"property": d.property, "kind": d.kind, "restriction": d.restriction,
                                    "why": d.why, "rounds": d.rounds, "accepted": d.accepted, "reason": d.reason}
                                for s, d in diffs.items() if s != "__hermit__"},
               "axioms": axioms}
        gp.write_text(json.dumps(rec, indent=1))
        n_authored += acc; n_species += len(members); n_hermit_ok += int(bool(h.accepted) if h else 0)
        print(f"  [{g + 1}/{k}] {glabel}: {acc}/{len(members)} differentiated · "
              f"HermiT {'OK' if h and h.accepted else 'flag'} — {(h.reason if h else '')[:60]}", flush=True)

    summary = {"lexicon": len(names), "genera": k, "members_authored": n_species,
               "differentiae_accepted": n_authored, "genera_hermit_consistent": n_hermit_ok,
               "differentia_yield": round(n_authored / max(1, n_species), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE · {n_authored}/{n_species} differentiae across {k} genera · "
          f"{n_hermit_ok}/{k} genera HermiT-consistent · → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

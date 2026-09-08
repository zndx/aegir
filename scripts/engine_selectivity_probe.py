"""engine_selectivity_probe.py — engine-first de-risk: does the ITERATIVE loop close selectivity?

Probe-1 (single-shot append) confirmed Qwen3.8-27B authors excellent positive-blind features, BUT naive append
raised MaxSim for true AND false alike ("token surface is mass") so held-out separation barely moved. This
is the loop version: bounded REPLACEMENT (not unbounded append), sibling framing, a net-positive keep (only
adopt an edit that improves held-out separation), and metrics-only specificity feedback across rounds. Still
positive-blind: the proposer sees def + parent + siblings + the 8 false positives + SCALAR feedback — never a
positive target. If the loop lifts held-out separation meaningfully, the METHOD is GREEN to materialize.

  LD_LIBRARY_PATH=$(pwd)/build/cuda-driver-libs CUDA_VISIBLE_DEVICES=4 \
    uv run --no-sync python scripts/engine_selectivity_probe.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aegir.engine.client import complete_detailed  # noqa: E402
from aegir.ontology import domain_index as DI  # noqa: E402
from aegir.ontology.colbert_encoder import get_encoder  # noqa: E402

STORE = REPO / "build" / "domain_harvest"
SYS = ("You refine an ontology concept's retrieval features for a late-interaction (ColBERT MaxSim) index. "
       "You see ONLY the false positives to repel — never the documents to attract. You author concise, "
       "highly specific feature text; generic words that appear in any technical document are worse than "
       "useless because they make the concept attract everything.")


def windows(text, w=80, stride=60, cap=4):
    words = text.split()
    return [" ".join(words[i:i + w]) for i in range(0, max(1, len(words) - 20), stride)][:cap]


def maxsim(qv, dv):
    return float((qv @ dv.T).max(axis=1).mean())


def propose(label, definition, parent, siblings, fp_block, feedback):
    fb = f"\n\nFEEDBACK on your last attempt (metrics only): {feedback}" if feedback else ""
    usr = f"""Concept: {label}
Definition: {definition}
Parent: {parent}
Sibling concepts you must ALSO be distinguishable from: {siblings}

This concept is FALSELY attracting these off-topic passages (repel them; you may NOT see what to attract):
{fp_block}{fb}

Author a DISCRIMINATOR (<=300 characters, positive-anchor prose; lead with what is UNIQUELY this concept;
use only terms a true document would contain; DROP generic process/system/record/data words) and name_hints.
Output ONLY JSON: {{"discriminator": "<=300 chars", "name_hints": ["verbatim domain term", ...]}}"""
    r = complete_detailed(usr, system_prompt=SYS, max_tokens=4000, temperature=0.4)
    return _parse_json(r["text"]), r


def _parse_json(txt: str) -> dict:
    """Tolerant: strip fences, grab the outer braces, retry with literal newlines flattened; never raise."""
    s = re.sub(r"```(?:json)?", "", (txt or "")).strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return {}
    for v in (m.group(0), m.group(0).replace("\n", " ").replace("\r", " ")):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            continue
    return {}


def main():
    enc = get_encoder()
    man = [json.loads(l) for l in (STORE / "manifest.jsonl").read_text().splitlines() if l.strip()]
    by_dom: dict[str, list[str]] = {}
    for m in man:
        by_dom.setdefault(m["domain"], []).append(m["hash"])

    def read_docs(dom, n):
        return [(STORE / "docs" / f"{h}.txt").read_text()[:4000]
                for h in by_dom.get(dom, [])[:n] if (STORE / "docs" / f"{h}.txt").exists()]

    lims_win = [w for d in read_docs("9", 20) for w in windows(d)]
    far_win = [w for d in (read_docs("12", 20) + read_docs("10", 10)) for w in windows(d)]
    print(f"windows: LIMS {len(lims_win)}  cross-domain {len(far_win)}")

    skos = DI.load_skos()
    def root(c): return (c.ancestor_codes() or [c.code])[0].split(".")[0]
    kids: dict[str, list[str]] = {}
    for c in skos.values():
        if getattr(c, "broader", None):
            kids.setdefault(c.broader, []).append(c.pref_label)
    cand = [c for c in skos.values() if root(c) == "9" and (c.text() or "").strip()]
    C = sorted(cand, key=lambda c: -len(kids.get(c.code, [])))[0]
    parent = next((c.pref_label for c in skos.values() if c.code == getattr(C, "broader", None)), "(root)")
    siblings = [c.pref_label for c in skos.values() if root(c) == "9" and c.code != C.code][:8]
    definition = getattr(C, "definition", "") or C.text()
    print(f"\nCONCEPT  {C.pref_label} ({C.code})  siblings: {siblings[:5]}")

    lv = enc.encode(lims_win)
    fv = enc.encode(far_win)
    cv0 = enc.encode([f"{C.pref_label}. {definition}"])[0]
    ms_l0 = np.array([maxsim(w, cv0) for w in lv])
    ms_f0 = np.array([maxsim(w, cv0) for w in fv])
    shown = np.argsort(-ms_f0)[:8]
    held_far = np.array([i for i in range(len(far_win)) if i not in set(shown.tolist())])
    fp_block = "\n".join(f"- {far_win[i][:280]}" for i in shown)
    sep0 = ms_l0.mean() - ms_f0[held_far].mean()
    print(f"\nROUND 0 (def only)   held-LIMS={ms_l0.mean():.4f}  held-cross={ms_f0[held_far].mean():.4f}  SEP={sep0:+.4f}")

    best_sep, best = sep0, None
    feedback = ""
    for rnd in range(1, 4):
        feat, r = propose(C.pref_label, definition, parent, ", ".join(siblings), fp_block, feedback)
        disc = (feat.get("discriminator", "") or "")[:300]
        hints = feat.get("name_hints", []) or []
        cv = enc.encode([f"{C.pref_label}. {definition} {disc} {' '.join(hints)}"])[0]
        ms_l = np.array([maxsim(w, cv) for w in lv])
        ms_f = np.array([maxsim(w, cv) for w in fv])
        sep = ms_l.mean() - ms_f[held_far].mean()
        keep = sep > best_sep
        cross_rose = ms_f[held_far].mean() > ms_f0[held_far].mean()
        print(f"ROUND {rnd}  SEP={sep:+.4f} (LIMS {ms_l.mean():.4f}, cross {ms_f[held_far].mean():.4f})  "
              f"{'🟢KEEP' if keep else 'revert'}  [{r['completion_tokens']}tok, disc {len(disc)}ch]")
        print(f"   disc: {disc[:150]}")
        if keep:
            best_sep, best = sep, feat
        feedback = (f"held-out separation was {sep:+.4f} (best so far {best_sep:+.4f}); cross-domain attraction "
                    f"{'ROSE — your text still has generic words matching unrelated technical docs' if cross_rose else 'fell'}. "
                    f"Be SHORTER and use only terms unique to {C.pref_label}.")

    d = best_sep - sep0
    print(f"\n=== best held-out separation  {sep0:+.4f} → {best_sep:+.4f}   Δ={d:+.4f} ===")
    print(f"VERDICT: {'🟢 GREEN — the loop closes selectivity; materialize the design' if d > 0.005 else ('🟡 MARGINAL — loop helps but weakly; composition needs work' if d > 0 else '🔴 RED — the mass effect dominates; rethink scoring')}")
    import os
    os._exit(0)


if __name__ == "__main__":
    main()

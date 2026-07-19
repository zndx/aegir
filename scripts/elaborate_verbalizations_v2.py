#!/usr/bin/env python
"""elaborate_verbalizations_v2 — MEMBRANE-INLINE elaboration: propose/dispose at generation time (#28).

RH's redesign after the pruning findings: the v1 batch generated pre-membrane output (slot-faithful but
not CLAIM-faithful — invented claims surfaced by the judge), so pruning was a whole separate stage. v2
avoids-and-stages pruning by disposing DURING generation, the differentia_authoring CAS pattern:

  propose (engine, no-new-claims prompt + few-shot)
    → membrane 1: slot/parse (deterministic — v1's filter, kept)
    → membrane 2: CLT feature-overlap floor (τ from build/clt_faithfulness_calibration.json — cheap,
                  judge-free; cuda:4, physically isolated from the engine)
    → membrane 3: LLM-judge, MARGIN BAND ONLY (τ ≤ J < τ_clear — the judge spends only where CLT is
                  uncertain; faithful ≥ 4 required)
    → reject ⇒ RE-PROMPT with the membrane's reason (closed loop, never one-shot)
  accepted ⇒ merge into the catalog's verbal_templates (dedup) — nothing pre-membrane ever lands.

Perplexity is NOT gated per-candidate (the fluency floor is a corpus-level regression guard in
semantic_layer_gate); traces retained as in v1. Cache keyed (template_id, base-hash, PROMPT_VERSION).

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/elaborate_verbalizations_v2.py --limit 5   # smoke
    ... --n 3 --write-catalog                                                               # full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.schema import load_catalog, save_catalog  # noqa: E402

PROMPT_VERSION = "elab-v2-membrane-2026-07-19"
_SLOT_RE = re.compile(r"\{[^}]*\}")
_CACHE = REPO / "build" / "verbalization_elaborations_v2.json"
_TRACES = REPO / "build" / "verbalization_elaboration_traces_v2.jsonl"
_CAL = REPO / "build" / "clt_faithfulness_calibration.json"

_SYSTEM = (
    "You rephrase formal ontology axioms as clear sentences for a technical textbook on relational data "
    "modeling and domain ontologies. Hard rules: (1) Preserve EVERY placeholder token in curly braces "
    "exactly as written — identical spelling, identical braces, each appearing once. (2) NO NEW CLAIMS: "
    "every statement in a rephrasing must be entailed by the axiom sentence alone — do not add purposes, "
    "benefits, contexts, examples, or properties the axiom does not assert; do not weaken 'only' to "
    "'some', invent relations, or change quantifiers or cardinalities ('exactly one' stays exactly one). "
    "A rephrasing that says LESS is acceptable only if it drops nothing the axiom asserts. (3) Each "
    "rephrasing is a DISTINCT syntactic style (definitional, procedural, normative, relational-fronted). "
    "(4) OUTPUT CONTRACT: each final rephrasing on its own line beginning exactly 'REPHRASING: '; "
    "thinking stays OUTSIDE these lines.\n\n"
    "Example — axiom: 'A {PatientRecord} is a record that is borne by exactly one {Patient}.'\n"
    "GOOD: REPHRASING: Every {PatientRecord} belongs to exactly one {Patient}, as a record of that patient.\n"
    "BAD (adds claims): 'ensuring regulatory compliance and continuity of care' — the axiom says no such thing."
)

_TAG = re.compile(r"^\s*REPHRASING:\s*(.+?)\s*$")
_META = re.compile(r"\b(draft|let'?s|wait|check|i'?ll|i will|note:|axiom:|placeholder|constraint \d|"
                   r"verify|step \d|first,|okay|here'?s)\b", re.I)


def slots_of(text: str) -> "set[str]":
    return set(_SLOT_RE.findall(text))


# ── membranes (each returns (ok, reason)) ─────────────────────────────────────

def m_parse(cand: str, base_slots: "set[str]") -> "tuple[bool, str]":
    if _META.search(cand):
        return False, "reads as reasoning/meta text, not a rephrasing"
    if slots_of(cand) != base_slots:
        return False, (f"placeholder set mismatch: expected exactly {sorted(base_slots)}")
    if not (30 <= len(cand) <= 400):
        return False, "length out of bounds for a single-sentence rephrasing"
    return True, ""


_OPENERS = re.compile(r"^(?:a|an|the|each|every|any|all)\s+", re.I)


def subject_drift(base: str, cand: str) -> bool:
    """True when the candidate does not OPEN with the base's head slot — the signature of a
    relation-direction flip or head demotion (the judged failure mode: CLT feature overlap is
    direction-blind, so 'A requested-by B' and 'B requests A' both clear the floor). Drifted
    candidates are not rejected — they are ROUTED TO THE JUDGE regardless of CLT margin."""
    heads = _SLOT_RE.findall(base)
    if not heads:
        return False
    head = heads[0]
    c = _OPENERS.sub("", cand.strip())
    return not c.startswith(head)


class CltMembrane:
    """Feature-overlap floor (judge-free). Loads lazily on cuda (CUDA_VISIBLE_DEVICES pins the GPU)."""

    def __init__(self):
        cal = json.loads(_CAL.read_text()) if _CAL.exists() else {}
        self.tau = float(cal.get("tau_clt") or 0.10)
        self.clear = max(self.tau * 2, 0.20)      # above this, no judge needed
        self.clt = None
        self._cache: dict = {}

    def _feats(self, text: str):
        if self.clt is None:
            import torch
            from circuit_tracer import ReplacementModel
            self.clt = ReplacementModel.from_pretrained(
                "Qwen/Qwen3-1.7B-Base", "bluelightai/clt-qwen3-1.7b-base-20k", dtype=torch.bfloat16)
        if text not in self._cache:
            toks = self.clt.tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=192).input_ids.to(self.clt.cfg.device)
            f = self.clt.get_activations(toks)[1]
            self._cache[text] = set(map(tuple, (f != 0).any(dim=1).nonzero().tolist()))
        return self._cache[text]

    def overlap(self, base: str, cand: str) -> float:
        fb, fc = self._feats(base), self._feats(cand)
        return len(fb & fc) / max(1, len(fb | fc))

    def check(self, base: str, cand: str) -> "tuple[str, float]":
        """→ ('pass'|'margin'|'fail', J)."""
        j = self.overlap(base, cand)
        if j < self.tau:
            return "fail", j
        return ("pass" if j >= self.clear else "margin"), j


def m_judge(base: str, cand: str) -> "tuple[bool, str]":
    from aegir.ontology.verbalization_quality import judge
    v = judge(base, cand, base, cand_is_a=True)
    if not v:
        return False, "judge unavailable — margin candidate held back (fail-closed)"
    if (v.get("faithful") or 0) >= 4:
        return True, ""
    return False, f"judge: unfaithful ({v.get('faithful')}) — {str(v.get('why'))[:140]}"


# ── the loop ──────────────────────────────────────────────────────────────────

def elaborate(template_id: str, base: str, n: int, rounds: int, clt: CltMembrane) -> "tuple[list[str], list[dict]]":
    from aegir.engine.client import complete_detailed
    base_slots = slots_of(base)
    accepted: "list[str]" = []
    log: "list[dict]" = []
    feedback = ""
    for r in range(rounds):
        prompt = (f"Axiom sentence:\n{base}\n\nProduce {n} distinct rephrasings per the rules."
                  + (f"\n\nYour previous attempt was REJECTED: {feedback}\nFix exactly that." if feedback else ""))
        try:
            resp = complete_detailed(prompt, capability="instruct", system_prompt=_SYSTEM,
                                     max_tokens=1600, temperature=0.7)
        except Exception as e:  # noqa: BLE001
            log.append({"round": r, "error": str(e)[:120]})
            break
        text = resp.get("text") or ""
        _TRACES.parent.mkdir(exist_ok=True)
        with _TRACES.open("a") as fh:
            fh.write(json.dumps({"template_id": template_id, "round": r, "prompt_version": PROMPT_VERSION,
                                 "reasoning": (resp.get("reasoning_content") or "")[:4000],
                                 "text": text[:4000]}) + "\n")
        reasons = []
        for line in text.splitlines():
            m = _TAG.match(line)
            if not m:
                continue
            cand = m.group(1).strip()
            ok, why = m_parse(cand, base_slots)
            if not ok:
                reasons.append(why)
                continue
            verdict, j = clt.check(base, cand)
            if verdict == "fail":
                reasons.append(f"too far from the axiom in feature space (J={j:.2f} < τ={clt.tau}) — "
                               f"likely added/omitted claims")
                continue
            if verdict == "pass" and subject_drift(base, cand):
                verdict = "margin"                # direction-blindness guard: the judge disposes
            if verdict == "margin":
                ok, why = m_judge(base, cand)
                if not ok:
                    reasons.append(why)
                    continue
            if cand not in accepted and cand != base:
                accepted.append(cand)
                log.append({"round": r, "accepted": cand, "J": round(j, 3), "via": verdict})
        if len(accepted) >= n:
            break
        feedback = "; ".join(dict.fromkeys(reasons))[:400] or "not enough distinct valid rephrasings"
    if not accepted and feedback:
        # RH (agent-mediated pattern): membranes INFORM AND REFINE — exhaustion is never a silent
        # drop. The template escalates to the ACP-harnessed proposer (aegir.refine — the multi-turn
        # channel where the agent sees the full membrane dialogue) via the worklist artifact.
        log.append({"escalate": "acp", "reasons": feedback})
    return accepted[:n], log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(REPO / "src/aegir/ontology/catalog/catalog.json"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write-catalog", action="store_true")
    a = ap.parse_args()

    cache = json.loads(_CACHE.read_text()) if _CACHE.exists() else {}
    cat = load_catalog(Path(a.catalog))
    clt = CltMembrane()
    print(f"membranes: parse → CLT floor τ={clt.tau} (clear ≥{clt.clear}) → judge on margin · "
          f"prompt {PROMPT_VERSION}", flush=True)
    todo = [t for t in cat.templates if (t.verbal_template or "").strip()]
    if a.limit:
        todo = todo[:a.limit]
    n_acc = n_tpl = 0
    for i, t in enumerate(todo):
        base = t.verbal_template.strip()
        key = f"{t.template_id}::{hashlib.sha1(base.encode()).hexdigest()[:8]}::{PROMPT_VERSION}"
        if key in cache:
            continue
        acc, log = elaborate(t.template_id, base, a.n, a.rounds, clt)
        cache[key] = {"template_id": t.template_id, "base": base, "accepted": acc, "log": log}
        n_acc += len(acc)
        n_tpl += 1
        if n_tpl % 5 == 0:
            _CACHE.write_text(json.dumps(cache, indent=1))
        print(f"[{i + 1}/{len(todo)}] {t.template_id}: {len(acc)}/{a.n} accepted", flush=True)
    _CACHE.write_text(json.dumps(cache, indent=1))
    esc = {k: rec for k, rec in cache.items()
           if any(x.get("escalate") for x in rec.get("log", []))}
    if esc:
        wl = REPO / "build" / "elaboration_acp_worklist.json"
        wl.write_text(json.dumps({k: {"template_id": r["template_id"], "base": r["base"],
                                      "reasons": next(x["reasons"] for x in r["log"] if x.get("escalate"))}
                                  for k, r in esc.items()}, indent=1))
        print(f"escalated to ACP refinement (never dropped): {len(esc)} templates → {wl}", flush=True)
    print(f"\n{n_tpl} templates processed · {n_acc} elaborations ACCEPTED (post-membrane)", flush=True)

    if a.write_catalog:
        by_tid: dict = {}
        for rec in cache.values():
            by_tid.setdefault(rec["template_id"], []).extend(rec.get("accepted") or [])
        n_m = 0
        for t in cat.templates:
            add = [x for x in by_tid.get(t.template_id, []) if x not in (t.verbal_templates or [])]
            if add:
                t.verbal_templates = (t.verbal_templates or []) + add
                n_m += len(add)
        save_catalog(cat, Path(a.catalog))
        print(f"merged {n_m} accepted elaborations into {a.catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

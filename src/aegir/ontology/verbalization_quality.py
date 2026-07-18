"""verbalization_quality — naturalness + faithfulness instrumentation (RH roadmap step 4).

Entropy (audit_verbalization_entropy) is NECESSARY but not SUFFICIENT: diverse frames can be diverse
garbage. This module adds the sufficiency half, three instruments per RH's spec:

  * **Local-LM perplexity** — Qwen3-1.7B-Base (spare GPU 4, bf16; the CLT bluelightai/clt-qwen3-1.7b-base-20k rides this same base → one scorer, future feature-space instrument) scores each verbalization's
    fluency; reported RELATIVE to the legacy DeepOnto single-string (ratio ≤ 1 = at least as natural).
    Slots are filled deterministically (de-camelCased slot names) before scoring — perplexity on
    ``{Marker}`` text would be meaningless.
  * **LLM-as-judge** — the local engine (same membrane discipline as everywhere else: schema-constrained,
    no free-text trust) scores faithfulness-to-axiom (claims ⊆ Manchester, nothing invented) and
    naturalness 1–5, plus a pairwise preference vs the legacy baseline.
  * **Human preference samples** — a shuffled, blinded sample sheet emitted for RH to grade; the gate
    records the sheet, humans stay in the loop without blocking automation.

Pre-registered floors live in semantic_layer_gate (quality dimension) — registered BEFORE first
measurement, per the gate's instrument-before-improvement discipline.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent

_SLOT = re.compile(r"\{(\w+)(?::\w+)*\}")


def fill_slots(text: str) -> str:
    """``{ServiceProvider}`` → ``service provider`` — deterministic, so fluency scoring sees English."""
    return _SLOT.sub(lambda m: re.sub(r"(?<!^)(?=[A-Z])", " ", m.group(1)).lower(), text)


# ── local-LM perplexity (pythia-160m, offline, lazy) ─────────────────────────────────────────

_PPL = {"tok": None, "model": None, "device": None}


def _ppl_load():
    if _PPL["model"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = os.environ.get("AEGIR_PPL_DEVICE", "cuda:4" if torch.cuda.is_available() else "cpu")
    name = os.environ.get("AEGIR_PPL_MODEL", "Qwen/Qwen3-1.7B-Base")
    try:                                   # cache-first (air-gap friendly); one-time fetch fallback
        _PPL["tok"] = AutoTokenizer.from_pretrained(name, local_files_only=True)
        _PPL["model"] = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype="bfloat16")
    except OSError:
        _PPL["tok"] = AutoTokenizer.from_pretrained(name)
        _PPL["model"] = AutoModelForCausalLM.from_pretrained(name, dtype="bfloat16")
    _PPL["model"] = _PPL["model"].to(dev).eval()
    _PPL["device"] = dev


def perplexity(texts: "list[str]") -> "list[float]":
    """Mean per-token perplexity per text (slots pre-filled by the caller)."""
    _ppl_load()
    import torch
    tok, model, dev = _PPL["tok"], _PPL["model"], _PPL["device"]
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
            if ids.shape[1] < 2:
                out.append(float("nan"))
                continue
            loss = model(ids, labels=ids).loss
            out.append(float(torch.exp(loss)))
    return out


# ── LLM-as-judge (engine, schema-constrained) ────────────────────────────────────────────────

_JUDGE_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "faithful": {"type": "integer", "minimum": 1, "maximum": 5},
        "natural": {"type": "integer", "minimum": 1, "maximum": 5},
        "prefer": {"type": "string", "enum": ["a", "b", "tie"]},
        "why": {"type": "string"},
    },
    "required": ["faithful", "natural", "prefer", "why"],
})

# Blind pairwise protocol: neutral A/B labels, caller randomizes the assignment (the first run
# labelled CANDIDATE/BASELINE and the judge showed a strong literalness/label bias toward BASELINE).
_JUDGE_PROMPT = """You are judging two VERBALIZATIONS (A and B) of an OWL axiom (Manchester syntax).

AXIOM:
{manchester}

A: {a}

B: {b}

Score verbalization A:
- faithful (1-5): 5 = every claim in A is entailed by the axiom and no axiom conjunct is missing;
  deduct for invented claims, dropped restrictions, or wrong quantifiers/cardinalities. A PARAPHRASE
  that preserves the axiom's meaning is fully faithful — do not reward surface similarity to the
  axiom's wording.
- natural (1-5): 5 = fluent, idiomatic technical English a textbook would print.
- prefer: which verbalization is better OVERALL (faithfulness first, then naturalness) — "a", "b",
  or "tie".
Treat {{CamelCase}} placeholders as class names. Judge only what is written."""


def judge(manchester: str, candidate: str, baseline: str, *, cand_is_a: bool = True,
          timeout: float = 300.0) -> "dict | None":
    """Blind pairwise judgment. ``cand_is_a`` places the candidate as A or B (caller randomizes);
    the returned dict is DE-BIASED back to candidate/baseline terms (faithful/natural always score
    the CANDIDATE; prefer ∈ candidate|baseline|tie)."""
    from aegir.engine.client import complete_detailed
    a, b = (candidate, baseline) if cand_is_a else (baseline, candidate)
    try:
        r = complete_detailed(
            _JUDGE_PROMPT.format(manchester=manchester, a=a, b=b),
            capability="instruct", max_tokens=400, temperature=0.1,
            json_schema=_JUDGE_SCHEMA, timeout=timeout)
        v = json.loads(r.get("text") or r.get("content") or "")
    except Exception:  # noqa: BLE001 — a judge failure is a skipped sample, not a crash
        return None
    if not cand_is_a:
        # A-scores describe the BASELINE in this orientation → only the preference maps back
        v["faithful"], v["natural"] = None, None
        v["prefer"] = {"a": "baseline", "b": "candidate", "tie": "tie"}.get(v.get("prefer"), "tie")
    else:
        v["prefer"] = {"a": "candidate", "b": "baseline", "tie": "tie"}.get(v.get("prefer"), "tie")
    v["cand_is_a"] = cand_is_a
    return v


# ── human preference sheet ───────────────────────────────────────────────────────────────────

def human_sample(rows: "list[dict]", out_md: Path, k: int = 20, seed: int = 13) -> Path:
    """Blinded A/B sheet: axiom + two verbalizations in random order; the key is written separately
    so grading stays blind (RH grades the .md; the .key.json de-blinds afterwards)."""
    rng = random.Random(seed)
    pick = rng.sample(rows, min(k, len(rows)))
    lines, key = ["# Verbalization preference sample (blind)\n"], {}
    for i, r in enumerate(pick, 1):
        a_first = rng.random() < 0.5
        first, second = (r["candidate"], r["baseline"]) if a_first else (r["baseline"], r["candidate"])
        key[str(i)] = {"A": "candidate" if a_first else "baseline",
                       "B": "baseline" if a_first else "candidate", "template": r["template"]}
        lines += [f"## {i}\n", f"Axiom: `{r['manchester'][:200]}`\n",
                  f"- **A**: {first}", f"- **B**: {second}", "", "Preferred (A/B/tie): ____\n"]
    out_md.write_text("\n".join(lines))
    out_md.with_suffix(".key.json").write_text(json.dumps(key, indent=1))
    return out_md

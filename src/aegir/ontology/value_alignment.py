"""value_alignment — the agent-mediated round-trip: constrain WHICH real GitTables values are plausible for a
column, SUBTRACTIVELY (the agent never authors a value — that would break provenance; it decides which
provenance-retained real values are admissible). Holland CAS / Signals & Boundaries: a BOUNDARY detects an
implausible fit (numeric magnitude spread, semantic mismatch), SIGNALS the column context + candidate real
values, the AGENT (engine) REASONS and returns a plausibility CONSTRAINT, a MEMBRANE disposes (the constraint
must leave a non-empty admissible pool), and the model + prompt version + emitted constraint are LINEAGE.
Cached + persisted (auditable, bounded cost). [[gittables_value_realism]] [[signal_boundary_machinery]]

RH 2026-07-13: fixes `base_salary = 2.97M` by admitting only the real values a domain expert finds plausible
for THIS column, not by inventing a number.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PROMPT_VERSION = "align-v1-2026-07-13"
_CACHE = Path(__file__).resolve().parent.parent.parent.parent / "build" / "gittables" / "alignment_cache.json"
_NUMERIC = {"monetary", "measurement", "quantity"}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class Constraint:
    """A subtractive admissibility constraint for a column's real-value pool (never a generated value)."""
    num_min: "float | None" = None
    num_max: "float | None" = None
    keep_terms: "list[str] | None" = None      # for string columns: admit only values containing one of these
    notes: str = ""
    model: str = ""                            # LINEAGE: which model reasoned it
    prompt_version: str = PROMPT_VERSION        # LINEAGE: which prompt


def _num(v: str) -> "float | None":
    m = _NUM_RE.search(str(v))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def apply(values: "list[str]", c: "Constraint | None") -> "list[str]":
    """Subtractively filter a real-value pool to the constraint — the values stay real + provenance-retained."""
    if not c:
        return values
    out = values
    if c.num_min is not None or c.num_max is not None:
        lo, hi = (c.num_min if c.num_min is not None else -1e30), (c.num_max if c.num_max is not None else 1e30)
        out = [v for v in out if (n := _num(v)) is not None and lo <= n <= hi]
    if c.keep_terms:
        low = [t.lower() for t in c.keep_terms]
        out = [v for v in out if any(t in v.lower() for t in low)]
    return out


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(d: dict) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(d, indent=1, sort_keys=True))


def _key(col_name: str, semantic_type: str, domain: str) -> str:
    return f"{PROMPT_VERSION}|{semantic_type}|{re.sub(r'[^a-z0-9]+', '_', col_name.lower())}|{domain.lower()}"


_SYS = (
    "You align GENERATED relational values to plausibility WITHOUT inventing values. You are given a column, its "
    "ontology meaning, and CANDIDATE REAL values sampled from public tables. Decide which real values a domain "
    "expert would accept for THIS column. For a numeric/amount column return the plausible [num_min, num_max] "
    "(e.g. an annual base salary is ~30000..300000, a unit price ~0.5..5000). For a string column, return "
    "keep_terms ONLY if the pool is semantically off (else leave null — most string pools are fine). NEVER "
    "return a specific value. Output one json block "
    "{\"num_min\":<number|null>,\"num_max\":<number|null>,\"keep_terms\":<[str]|null>,\"notes\":\"<why>\"}.")


def constrain(col_name: str, definition: str, semantic_type: str, samples: "list[str]",
              domain: str = "", *, engine=True) -> Constraint:
    """Agent-mediated + cached. Returns a plausibility Constraint for the column (the SIGNAL→reason→dispose loop).
    ``engine=False`` returns an empty constraint (tests / offline) that is NOT cached — an offline placeholder must
    never poison the key so a later engine-on call can still resolve it authoritatively."""
    cache = _load_cache()
    k = _key(col_name, semantic_type, domain)
    if k in cache:
        return Constraint(**cache[k])
    if not engine:                                   # offline/test: empty placeholder, NEVER cached (see docstring)
        return Constraint(notes="offline (engine=False) — no constraint applied", model="")
    if semantic_type not in _NUMERIC and semantic_type not in ("organization", "product", "job_title"):
        c = Constraint(notes="no alignment needed (clean string pool or non-injected)", model="")
        cache[k] = asdict(c); _save_cache(cache); return c
    from aegir.engine.client import complete_detailed  # noqa: PLC0415 — late import (no engine dep at import)
    ex = ", ".join(str(s) for s in samples[:12])
    prompt = (f"Column: {col_name}\nOntology meaning: {definition or '(unspecified)'}\n"
              f"Semantic type: {semantic_type}\nTable domain: {domain or '(general)'}\n"
              f"Candidate real values: {ex}\n\nReturn the plausibility constraint.")
    schema = json.dumps({"type": "object", "properties": {
        "num_min": {"type": ["number", "null"]}, "num_max": {"type": ["number", "null"]},
        "keep_terms": {"type": ["array", "null"], "items": {"type": "string"}},
        "notes": {"type": "string"}}, "required": ["notes"]})
    try:
        out = complete_detailed(prompt, capability="instruct", system_prompt=_SYS,
                                max_tokens=2000, temperature=0.2, json_schema=schema)
        m = re.search(r"\{.*\}", out["text"], re.S)
        d = json.loads(m.group(0)) if m else {}
        c = Constraint(num_min=d.get("num_min"), num_max=d.get("num_max"), keep_terms=d.get("keep_terms"),
                       notes=str(d.get("notes", ""))[:200], model=out.get("model", ""))
    except Exception as e:  # noqa: BLE001 — an engine hiccup must NOT silently pass an unaligned column
        raise RuntimeError(f"value-alignment boundary: engine failed for column '{col_name}': {e}") from e
    cache[k] = asdict(c)
    _save_cache(cache)
    return c

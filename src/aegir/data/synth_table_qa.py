"""TAPAS-style synthetic SQL example generator.

Generates binary-classification examples over a structured table:
random natural-language statements that compare two SQL-like
expressions, paired with the truth label evaluated against the table.

Three statement patterns:

    A. aggregate-expr <op> constant
        e.g., "the sum of Earnings when Country is Australia is
              greater than 2,000,000"

    B. single-cell-expr <op> constant
        e.g., "the wins when Player is Lee Janzen is less than 5"

    C. aggregate-expr-1 <op> aggregate-expr-2
        e.g., "the sum of Earnings when Country is Australia is
              greater than the sum of Earnings when Country is
              United States"

Aggregations: SUM, MIN, MAX, AVG, COUNT.
Comparisons:  <, >, =, <=, >=, !=  (each with multiple verbalizations).

The generator returns ``(statement_text, label)`` tuples where label
is bool. Label-balanced via rejection-sampling: each generated example
flips toward the minority class until a 50/50 cumulative balance is
reached, then runs free.

This is the Aegir Phase 0.5 intermediate-pretrain primitive,
following Eisenschlos et al. (EMNLP Findings 2020), "Understanding
Tables with Intermediate Pre-training" — same recipe, written from
scratch for our byte-level + H-Net stack.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────
# Column type detection + numeric parsing
# ─────────────────────────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(r"[+-]?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def parse_numeric(s: str) -> float | None:
    """Parse a cell string to float, stripping common formatting.

    Handles thousands-separators (commas), surrounding $/%/percent signs,
    leading/trailing whitespace. Returns None if no numeric content found.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    m = _NUMERIC_RE.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def is_numeric_column(values: list[str], min_fraction: float = 0.8) -> bool:
    """True if at least ``min_fraction`` of values parse as numeric."""
    if not values:
        return False
    parsed = sum(1 for v in values if parse_numeric(v) is not None)
    return parsed / len(values) >= min_fraction


@dataclass
class ParsedColumn:
    name: str
    raw_values: list[str]
    is_numeric: bool
    numeric_values: list[float | None]  # None where parse failed


def parse_table(column_names: list[str], rows: list[list[str]]) -> list[ParsedColumn]:
    """Transpose rows into per-column views with type detection.

    Args:
        column_names: List of column names.
        rows: List of rows, each a list of cell strings (length ==
            len(column_names)). Ragged rows are tolerated; missing cells
            become "".

    Returns:
        One ParsedColumn per name, in input order.
    """
    n_cols = len(column_names)
    cols: list[list[str]] = [[] for _ in range(n_cols)]
    for row in rows:
        for i in range(n_cols):
            cols[i].append(str(row[i]) if i < len(row) else "")
    out: list[ParsedColumn] = []
    for name, vals in zip(column_names, cols):
        is_num = is_numeric_column(vals)
        nums: list[float | None]
        if is_num:
            nums = [parse_numeric(v) for v in vals]
        else:
            nums = [None] * len(vals)
        out.append(ParsedColumn(name=name, raw_values=vals,
                                is_numeric=is_num, numeric_values=nums))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Aggregations
# ─────────────────────────────────────────────────────────────────────────

def _matched_indices(filter_col: ParsedColumn, filter_val: str) -> list[int]:
    """Row indices where filter_col equals filter_val (string match)."""
    return [i for i, v in enumerate(filter_col.raw_values) if v == filter_val]


def _aggregate(target_col: ParsedColumn, indices: list[int], op: str) -> float | None:
    """Compute SUM / MIN / MAX / AVG / COUNT over ``indices``.

    Returns None if the operation is undefined for the subset (e.g.,
    SUM on a non-numeric column, or AVG over an empty set).
    """
    if op == "COUNT":
        return float(len(indices))
    if not target_col.is_numeric:
        return None
    vals: list[float] = [
        v for v in (target_col.numeric_values[i] for i in indices)
        if v is not None
    ]
    if not vals:
        return None
    if op == "SUM":
        return sum(vals)
    if op == "MIN":
        return min(vals)
    if op == "MAX":
        return max(vals)
    if op == "AVG":
        return sum(vals) / len(vals)
    raise ValueError(f"unknown agg: {op}")


def _single_cell(target_col: ParsedColumn, indices: list[int]) -> float | None:
    """Return the target column value at the *first* matching row, as float.

    Returns None if no rows match or the cell isn't numeric.
    """
    if not indices:
        return None
    if not target_col.is_numeric:
        return None
    v = target_col.numeric_values[indices[0]]
    return v


# ─────────────────────────────────────────────────────────────────────────
# Comparison + verbalization
# ─────────────────────────────────────────────────────────────────────────

_OPS = {
    "<":  ("is less than",     lambda a, b: a < b),
    ">":  ("is greater than",  lambda a, b: a > b),
    "=":  ("is equal to",      lambda a, b: math.isclose(a, b, rel_tol=1e-6)),
    "<=": ("is at most",       lambda a, b: a <= b or math.isclose(a, b, rel_tol=1e-6)),
    ">=": ("is at least",      lambda a, b: a >= b or math.isclose(a, b, rel_tol=1e-6)),
    "!=": ("differs from",     lambda a, b: not math.isclose(a, b, rel_tol=1e-6)),
}

_AGG_PHRASE = {
    "SUM":   "the sum of",
    "MIN":   "the minimum of",
    "MAX":   "the maximum of",
    "AVG":   "the average of",
    "COUNT": "the count of rows where",
}


def _verbalize_aggregate(agg: str, target: ParsedColumn,
                         filter_col: ParsedColumn, filter_val: str) -> str:
    if agg == "COUNT":
        # COUNT phrasing is "the count of rows where Country is Australia"
        return f"{_AGG_PHRASE['COUNT']} {filter_col.name} is {filter_val}"
    return (f"{_AGG_PHRASE[agg]} {target.name} when "
            f"{filter_col.name} is {filter_val}")


def _verbalize_single_cell(target: ParsedColumn,
                           filter_col: ParsedColumn,
                           filter_val: str) -> str:
    return f"{target.name} when {filter_col.name} is {filter_val}"


def _format_constant(x: float) -> str:
    if x.is_integer():
        return f"{int(x):,}"
    return f"{x:,.2f}"


# ─────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SynthExample:
    statement: str
    label: bool


def _pick_filter(cols: list[ParsedColumn], rng: random.Random) -> tuple[ParsedColumn, str] | None:
    """Pick a (filter_col, filter_val) pair whose value occurs in the table.

    Prefers categorical (non-numeric) filter columns when available.
    Returns None if no suitable filter can be constructed.
    """
    candidates = [c for c in cols if not c.is_numeric and any(c.raw_values)]
    if not candidates:
        candidates = [c for c in cols if any(c.raw_values)]
    if not candidates:
        return None
    fc = rng.choice(candidates)
    non_empty = [v for v in fc.raw_values if v]
    if not non_empty:
        return None
    return fc, rng.choice(non_empty)


def _pick_aggregate_target(cols: list[ParsedColumn], rng: random.Random,
                           allow_count: bool = True) -> tuple[str, ParsedColumn] | None:
    """Pick an (agg, target_col) pair.

    Numeric columns get SUM/MIN/MAX/AVG. COUNT works on any column.
    """
    numeric = [c for c in cols if c.is_numeric]
    options: list[tuple[str, ParsedColumn]] = []
    for c in numeric:
        for agg in ("SUM", "MIN", "MAX", "AVG"):
            options.append((agg, c))
    if allow_count:
        # COUNT(*) — target is conventional; we'll use the first column.
        # The phrase doesn't reference the target, just the filter.
        if cols:
            options.append(("COUNT", cols[0]))
    if not options:
        return None
    return rng.choice(options)


def _gen_pattern_A(cols: list[ParsedColumn], rng: random.Random) -> SynthExample | None:
    """aggregate-expr <op> numeric-constant."""
    agg_target = _pick_aggregate_target(cols, rng)
    if not agg_target:
        return None
    agg, target = agg_target
    filt = _pick_filter(cols, rng)
    if not filt:
        return None
    fc, fv = filt
    indices = _matched_indices(fc, fv)
    if not indices:
        return None
    lhs_val = _aggregate(target, indices, agg)
    if lhs_val is None:
        return None
    # Generate a constant — sometimes near lhs (to test boundary cases),
    # sometimes far. Half-and-half nudges label balance toward 50/50.
    if rng.random() < 0.5:
        rhs_val = lhs_val * rng.uniform(0.3, 1.7)
    else:
        rhs_val = lhs_val + rng.choice([-1, 1]) * rng.uniform(0.1, 5.0) * max(abs(lhs_val), 1.0)
    op = rng.choice(list(_OPS.keys()))
    op_phrase, op_fn = _OPS[op]
    lhs_text = _verbalize_aggregate(agg, target, fc, fv)
    rhs_text = _format_constant(rhs_val)
    return SynthExample(
        statement=f"{lhs_text} {op_phrase} {rhs_text}",
        label=op_fn(lhs_val, rhs_val),
    )


def _gen_pattern_B(cols: list[ParsedColumn], rng: random.Random) -> SynthExample | None:
    """single-cell <op> numeric-constant."""
    numeric_targets = [c for c in cols if c.is_numeric]
    if not numeric_targets:
        return None
    target = rng.choice(numeric_targets)
    filt = _pick_filter([c for c in cols if c is not target], rng)
    if not filt:
        return None
    fc, fv = filt
    indices = _matched_indices(fc, fv)
    if not indices:
        return None
    lhs_val = _single_cell(target, indices)
    if lhs_val is None:
        return None
    if rng.random() < 0.5:
        rhs_val = lhs_val * rng.uniform(0.3, 1.7)
    else:
        rhs_val = lhs_val + rng.choice([-1, 1]) * rng.uniform(0.1, 5.0) * max(abs(lhs_val), 1.0)
    op = rng.choice(list(_OPS.keys()))
    op_phrase, op_fn = _OPS[op]
    lhs_text = _verbalize_single_cell(target, fc, fv)
    rhs_text = _format_constant(rhs_val)
    return SynthExample(
        statement=f"{lhs_text} {op_phrase} {rhs_text}",
        label=op_fn(lhs_val, rhs_val),
    )


def _gen_pattern_C(cols: list[ParsedColumn], rng: random.Random) -> SynthExample | None:
    """aggregate-expr-1 <op> aggregate-expr-2."""
    a1 = _pick_aggregate_target(cols, rng, allow_count=False)
    a2 = _pick_aggregate_target(cols, rng, allow_count=False)
    if not a1 or not a2:
        return None
    agg1, t1 = a1
    agg2, t2 = a2
    f1 = _pick_filter(cols, rng)
    f2 = _pick_filter(cols, rng)
    if not f1 or not f2:
        return None
    fc1, fv1 = f1
    fc2, fv2 = f2
    i1 = _matched_indices(fc1, fv1)
    i2 = _matched_indices(fc2, fv2)
    if not i1 or not i2:
        return None
    lhs_val = _aggregate(t1, i1, agg1)
    rhs_val = _aggregate(t2, i2, agg2)
    if lhs_val is None or rhs_val is None:
        return None
    op = rng.choice(list(_OPS.keys()))
    op_phrase, op_fn = _OPS[op]
    lhs_text = _verbalize_aggregate(agg1, t1, fc1, fv1)
    rhs_text = _verbalize_aggregate(agg2, t2, fc2, fv2)
    return SynthExample(
        statement=f"{lhs_text} {op_phrase} {rhs_text}",
        label=op_fn(lhs_val, rhs_val),
    )


_PATTERNS: list[Callable[[list[ParsedColumn], random.Random], SynthExample | None]] = [
    _gen_pattern_A,
    _gen_pattern_A,  # weight A twice — most common
    _gen_pattern_B,
    _gen_pattern_C,
]


def generate_examples(
    column_names: list[str],
    rows: list[list[str]],
    n_max: int,
    rng: random.Random,
    max_attempts_per_example: int = 5,
    label_balance: bool = True,
) -> list[SynthExample]:
    """Generate up to ``n_max`` examples from a single table.

    Args:
        column_names: Column header strings.
        rows: Body rows (each a list of cell strings).
        n_max: Max examples to generate.
        rng: Random source.
        max_attempts_per_example: Tries before giving up on each slot.
        label_balance: If True, bias toward the minority class so cumulative
            T/F split stays near 50/50 within a table.

    Returns:
        List of SynthExamples (possibly < n_max if rejection often fires).
    """
    cols = parse_table(column_names, rows)
    if len(cols) < 2 or not rows:
        return []
    out: list[SynthExample] = []
    n_true = 0
    n_false = 0

    for _ in range(n_max):
        target_label: bool | None = None
        if label_balance and (n_true + n_false) >= 4:
            # Bias toward whichever side is under-represented.
            target_label = n_false > n_true

        for _attempt in range(max_attempts_per_example):
            pattern = rng.choice(_PATTERNS)
            ex = pattern(cols, rng)
            if ex is None:
                continue
            if target_label is not None and ex.label != target_label:
                continue
            out.append(ex)
            if ex.label:
                n_true += 1
            else:
                n_false += 1
            break

    return out

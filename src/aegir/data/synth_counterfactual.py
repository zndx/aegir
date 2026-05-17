"""TAPAS-style counterfactual example generator.

Generates binary-classification examples from a Wikipedia-style table
with surrounding text:

- **Negative example** (label=False, "not corrupted"): the original
  text as it appears alongside the table.
- **Positive example** (label=True, "corrupted"): the original text
  with one cell-grounded entity swapped for a different value from
  the same column of the same table.

This is the second of TAPAS's two intermediate-pretrain objectives
(Eisenschlos et al., EMNLP Findings 2020, §3.1). The synthetic-SQL
objective teaches *aggregation reasoning*; this objective teaches
*entity grounding* — the model must check whether a named entity
in surrounding prose is consistent with what the table actually
contains, rather than learning to ignore the table.

Generation policy:

1. Find all cell values in the table that occur as substrings of the
   surrounding text (longer-than-3-char values only, to avoid common
   filler like "1" or "is").
2. For each candidate (col_idx, cell_value) match, sample a different
   value from the same column as the swap candidate.
3. Produce a pair: (original_text, label=False), (perturbed_text, label=True).

Match scoring is greedy and case-insensitive; first occurrence wins.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass
class CounterfactualExample:
    text: str
    label: bool  # True = corrupted, False = original
    swap_info: tuple[str, str] | None = None  # (original_substring, replacement)


_MIN_ENTITY_LEN = 4   # short cells get filtered to avoid noise like "1", "no"
_WORD_BOUNDARY = re.compile(r"[A-Za-z0-9]")


def _find_match(text: str, needle: str) -> tuple[int, int] | None:
    """Case-insensitive first-match search with word-boundary check.

    Returns ``(start, end)`` indices into ``text`` or None. Uses regex
    with IGNORECASE so we never rely on ``str.lower()`` returning a
    string of the same length as the original — important because some
    Unicode characters (e.g., Turkish dotted-I) change length under
    lowercase, which would invalidate index arithmetic against ``text``.
    """
    if len(needle) < _MIN_ENTITY_LEN:
        return None
    try:
        pattern = re.compile(re.escape(needle), re.IGNORECASE)
    except re.error:
        return None
    for m in pattern.finditer(text):
        i, end = m.start(), m.end()
        if end > len(text):  # defensive — shouldn't happen, but safe
            continue
        before_ok = i == 0 or not _WORD_BOUNDARY.match(text[i - 1])
        after_ok = end == len(text) or not _WORD_BOUNDARY.match(text[end])
        if before_ok and after_ok:
            return i, end
    return None


def _column_alternatives(column_values: list[str], used_value: str,
                         min_len: int = _MIN_ENTITY_LEN) -> list[str]:
    """Distinct values in the column, excluding ``used_value`` and short cells."""
    seen: set[str] = set()
    out: list[str] = []
    used_norm = used_value.lower()
    for v in column_values:
        if not v or len(v) < min_len:
            continue
        if v.lower() == used_norm:
            continue
        if v.lower() in seen:
            continue
        seen.add(v.lower())
        out.append(v)
    return out


def generate_counterfactuals(
    column_names: list[str],
    rows: list[list[str]],
    surrounding_text: str,
    rng: random.Random,
    max_pairs: int = 1,
) -> list[CounterfactualExample]:
    """Generate up to ``max_pairs`` (original, perturbed) example pairs.

    Each pair contributes two CounterfactualExamples: one with the
    original text (label=False) and one with the corrupted text
    (label=True). Returns a flat list of examples.

    Args:
        column_names: Names of the table columns.
        rows: Body rows (each a list of cell strings).
        surrounding_text: Free-form text near the table (e.g.,
            TITLE + DESCRIPTION + SEGMENT_TEXT for TAPAS interactions).
        rng: Random source.
        max_pairs: Max (original, perturbed) pairs to emit. Each pair
            uses a different (column, cell) anchor where possible.

    Returns:
        List of CounterfactualExample. Length ≤ 2 * max_pairs.
    """
    if not surrounding_text.strip() or not rows or not column_names:
        return []

    # Transpose to per-column views.
    n_cols = len(column_names)
    cols: list[list[str]] = [[] for _ in range(n_cols)]
    for row in rows:
        for i in range(n_cols):
            cols[i].append(str(row[i]) if i < len(row) else "")

    # Score candidates: every cell value that appears in surrounding_text.
    candidates: list[tuple[int, str, tuple[int, int]]] = []
    seen_anchors: set[tuple[int, str]] = set()
    for col_idx in range(n_cols):
        for v in cols[col_idx]:
            if (col_idx, v.lower()) in seen_anchors:
                continue
            m = _find_match(surrounding_text, v)
            if m:
                candidates.append((col_idx, v, m))
                seen_anchors.add((col_idx, v.lower()))

    if not candidates:
        return []

    rng.shuffle(candidates)

    examples: list[CounterfactualExample] = []
    used_columns: set[int] = set()
    for col_idx, value, (start, end) in candidates:
        if col_idx in used_columns:
            continue  # Diversify across columns.
        alts = _column_alternatives(cols[col_idx], value)
        if not alts:
            continue
        replacement = rng.choice(alts)
        # Preserve original case style as much as possible: just swap
        # the matched substring with the replacement verbatim.
        perturbed = surrounding_text[:start] + replacement + surrounding_text[end:]

        examples.append(CounterfactualExample(
            text=surrounding_text, label=False, swap_info=None,
        ))
        examples.append(CounterfactualExample(
            text=perturbed, label=True,
            swap_info=(surrounding_text[start:end], replacement),
        ))
        used_columns.add(col_idx)
        if len(examples) >= 2 * max_pairs:
            break

    return examples

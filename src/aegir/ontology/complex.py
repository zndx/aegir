"""Family complex — empirically-derived allowable simplices over the
ontology's family axis.

A family is a vertex; a chapter's set of cited template-families is a
simplex; R_axiom (and friends) is a per-simplex efficacy weight.

The complex is *data-driven*: each generation run produces evidence for
or against specific simplices. The artifact in ``family_complex.json``
encodes the current best estimate:

  - ``maximal_simplices``: simplices measured with R_axiom ≥ floor and
    n ≥ min_count. Larger combinations whose 1-faces all pass aren't
    automatically promoted; they need their own measurement to enter.

  - ``measured_below_floor``: simplices we've directly observed to fail
    (mean + std < floor). These are local punctures — they do NOT
    propagate to their supersets, since data shows larger simplices
    can rescue a bad pair (e.g. {ebpf,lt} fails at R_ax=0.29 but
    {foundation,obs,ebpf,lt} passes at 0.45 because foundation+obs
    provide a noun-form column spine that the bad pair lacks).

Allowability for generation::

    S is allowed iff:
       1. S ⊆ M for some M in maximal_simplices  (in the closure)
       2. S is not in measured_below_floor       (not directly broken)

If a target topic's filtered templates produce a family-set that's
not allowed, ``best_face`` returns the largest allowed subset and the
sampler drops templates to match.

This module is intentionally narrow: no numpy, no pandas, no I/O
beyond reading the JSON artifact. Designed to be importable by the
chapter generator without inflating its hot-path dependency surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SimplexEvidence:
    """A measured simplex with its statistics."""
    vertices: frozenset[str]
    r_axiom_mean: float
    r_axiom_std: float
    r_iri_mean: float
    r_density_mean: float
    accepted_frac: float
    n: int


@dataclass
class FamilyComplex:
    """Empirical sub-complex over the family axis with allowability queries."""

    maximal_simplices: list[SimplexEvidence] = field(default_factory=list)
    measured_below_floor: list[SimplexEvidence] = field(default_factory=list)
    r_axiom_floor: float = 0.45
    min_count: int = 2
    source_runs: list[str] = field(default_factory=list)
    computed_at: str = ""

    # Indexed views built once at load
    _max_sets: list[frozenset[str]] = field(default_factory=list, repr=False)
    _puncture_sets: set[frozenset[str]] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._max_sets = [m.vertices for m in self.maximal_simplices]
        self._puncture_sets = {p.vertices for p in self.measured_below_floor}

    @property
    def vertices(self) -> frozenset[str]:
        out: set[str] = set()
        for m in self._max_sets:
            out.update(m)
        for p in self._puncture_sets:
            out.update(p)
        return frozenset(out)

    def is_in_closure(self, simplex: frozenset[str]) -> bool:
        """True if simplex is a subset of any maximal simplex."""
        if not simplex:
            return True
        return any(simplex.issubset(m) for m in self._max_sets)

    def is_punctured(self, simplex: frozenset[str]) -> bool:
        """True if this exact simplex was measured below floor."""
        return simplex in self._puncture_sets

    def is_allowed(self, simplex: frozenset[str]) -> bool:
        """True iff simplex is in the closure AND not directly punctured."""
        return self.is_in_closure(simplex) and not self.is_punctured(simplex)

    def best_face(self, candidate: frozenset[str]) -> frozenset[str]:
        """Largest subset of ``candidate`` that is allowed.

        If ``candidate`` itself is allowed, returns it unchanged. Otherwise
        searches subsets in descending size order, returning the first
        allowed one. Falls back to empty set if nothing is allowed (rare
        — would mean every vertex of candidate is in a puncture, which
        the complex doesn't currently support since 0-simplex punctures
        aren't enabled).
        """
        if self.is_allowed(candidate):
            return candidate
        verts = sorted(candidate)
        for k in range(len(verts) - 1, 0, -1):
            best: frozenset[str] | None = None
            for combo in combinations(verts, k):
                cand = frozenset(combo)
                if self.is_allowed(cand):
                    # Among same-size allowed faces, prefer the one whose
                    # measured R_axiom is highest if we have evidence.
                    if best is None or self._score_for_preference(cand) > self._score_for_preference(best):
                        best = cand
            if best is not None:
                return best
        return frozenset()

    def _score_for_preference(self, simplex: frozenset[str]) -> float:
        """Lookup R_axiom_mean for this simplex if it's a measured maximal.
        Returns 0.5 (above floor by default) for un-measured allowed
        simplices — they're permitted by closure but not preferred over
        directly-measured good ones."""
        for m in self.maximal_simplices:
            if m.vertices == simplex:
                return m.r_axiom_mean
        return self.r_axiom_floor + 0.01

    @classmethod
    def from_json(cls, path: str | Path) -> "FamilyComplex":
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            maximal_simplices=[
                SimplexEvidence(
                    vertices=frozenset(s["vertices"]),
                    r_axiom_mean=float(s["r_axiom_mean"]),
                    r_axiom_std=float(s.get("r_axiom_std", 0.0)),
                    r_iri_mean=float(s.get("r_iri_mean", 0.0)),
                    r_density_mean=float(s.get("r_density_mean", 0.0)),
                    accepted_frac=float(s.get("accepted_frac", 0.0)),
                    n=int(s["n"]),
                )
                for s in data.get("maximal_simplices", [])
            ],
            measured_below_floor=[
                SimplexEvidence(
                    vertices=frozenset(s["vertices"]),
                    r_axiom_mean=float(s["r_axiom_mean"]),
                    r_axiom_std=float(s.get("r_axiom_std", 0.0)),
                    r_iri_mean=float(s.get("r_iri_mean", 0.0)),
                    r_density_mean=float(s.get("r_density_mean", 0.0)),
                    accepted_frac=float(s.get("accepted_frac", 0.0)),
                    n=int(s["n"]),
                )
                for s in data.get("measured_below_floor", [])
            ],
            r_axiom_floor=float(data.get("r_axiom_floor", 0.45)),
            min_count=int(data.get("min_count", 2)),
            source_runs=list(data.get("source_runs", [])),
            computed_at=str(data.get("computed_at", "")),
        )

    def to_json(self, path: str | Path) -> None:
        path = Path(path)

        def _emit(es: Iterable[SimplexEvidence]) -> list[dict]:
            return [
                {
                    "vertices": sorted(e.vertices),
                    "r_axiom_mean": round(e.r_axiom_mean, 4),
                    "r_axiom_std": round(e.r_axiom_std, 4),
                    "r_iri_mean": round(e.r_iri_mean, 4),
                    "r_density_mean": round(e.r_density_mean, 4),
                    "accepted_frac": round(e.accepted_frac, 4),
                    "n": e.n,
                }
                for e in es
            ]

        path.write_text(json.dumps({
            "r_axiom_floor": self.r_axiom_floor,
            "min_count": self.min_count,
            "computed_at": self.computed_at,
            "source_runs": self.source_runs,
            "maximal_simplices": _emit(self.maximal_simplices),
            "measured_below_floor": _emit(self.measured_below_floor),
        }, indent=2) + "\n")

"""scan_coined_external — the boundary that keeps coined external names from re-surfacing.

Real BFO/CCO are OPAQUE IRIs (bfo:0000031, cco:ont00000965). Human-readable CamelCase tokens
(cco:DesignativeICE, cco:InformationContentEntity, cco:DirectiveICE, cco:BusinessEntity …) do NOT
exist in the authoritative ontologies — they are hallucinations that re-enter from stale docs, notes,
and old code and re-inject fictions into the ontology. This gate extracts every cco:/bfo: reference
from the LIVE surface (code + current docs) and rejects any the authority does not contain — the same
existence boundary the ontology's `verify_external_refs` enforces, applied to the source tree so the
tokens cannot come back. [[bfo_cco_grounding_mandate]] [[external-namespace-integrity]]

Scope: scripts/*.py, src/**/*.py, docs/current/**/*.md, EVIDENCE.md, the strategy objective cards.
Out of scope (archival / handled elsewhere): docs/scratch/ (dated work logs), *.json catalogs (the
Remediation Sweep + `validate_detailed` gate them), ref/, build/, outputs/. A line may opt out with a
trailing `coined-ok` marker (for the few places that NAME a token in order to reject it); files whose
job is to reason about the contamination are allowlisted below.

    uv run --no-sync python scripts/scan_coined_external.py            # exit 1 if any coined ref
    just scan-coined                                                  # CI wrapper
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.external_index import get_index  # noqa: E402

REF = re.compile(r"(?<![\w./-])(cco|bfo):([A-Za-z0-9_]+)")

# Named hallucinations: readable CamelCase forms that are NOT real IRIs in any namespace and
# keep re-surfacing from stale docs/notes/old code. Banned wherever they appear (bare, cco:, sdg:).
# Real CCO ICE children are opaque: Designative ont00000686 / Descriptive ont00000853 /
# Prescriptive ont00000965 / genus ont00000958. "DirectiveICE" never existed. [[bfo_cco_grounding_mandate]]
BANNED = re.compile(r"\b(?:(?:Designative|Descriptive|Prescriptive|Directive|Name)ICE|InformationContentEntity)\b")
# config JSON scanned for BANNED tokens only (catalog.json stays excluded — the sweep gates it)
CONFIG = ["src/aegir/ontology/axiom_patterns.json"]

# Files whose PURPOSE is to name/reject the coined tokens — allowlisted wholesale.
ALLOW = {
    "scripts/scan_coined_external.py",
    "scripts/resolve_contamination.py",
    "scripts/reauthor_contamination.py",
    "src/aegir/ontology/external_index.py",
}

SCOPES = [
    ("scripts", "*.py"),
    ("src", "*.py"),
    ("docs/current", "*.md"),
    ("strategy/objective", "*.md"),
]
NAMED_FILES = ["EVIDENCE.md"]


def _iter_files():
    for root, glob in SCOPES:
        base = REPO / root
        if base.exists():
            yield from base.rglob(glob)
    for name in NAMED_FILES:
        p = REPO / name
        if p.exists():
            yield p


def main() -> int:
    idx = get_index()
    violations: list[tuple[str, int, str]] = []
    for path in _iter_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOW:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "coined-ok" in line:
                continue
            for m in REF.finditer(line):
                ref = f"{m.group(1)}:{m.group(2)}"
                if not idx.exists(ref):
                    violations.append((rel, lineno, ref))

    # BANNED pass: the named hallucinations, anywhere (incl. bare + config JSON)
    for path in list(_iter_files()) + [REPO / c for c in CONFIG if (REPO / c).exists()]:
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOW:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "coined-ok" in line:
                continue
            for m in BANNED.finditer(line):
                violations.append((rel, lineno, f"{m.group(0)} (banned hallucination)"))

    if not violations:
        print(f"scan-coined: clean — every cco:/bfo: ref on the live surface exists in the authority "
              f"(CCO {idx.version('cco').split('/')[-1]}, BFO {idx.version('bfo').split('/')[-1]}).")
        return 0

    by_ref: dict[str, int] = {}
    for _, _, ref in violations:
        by_ref[ref] = by_ref.get(ref, 0) + 1
    print(f"scan-coined: {len(violations)} coined/nonexistent external ref(s) on the live surface "
          f"({len(by_ref)} distinct) — real CCO/BFO are opaque IRIs; these do not exist:\n")
    for ref, n in sorted(by_ref.items(), key=lambda kv: -kv[1]):
        print(f"  ✗ {ref:36s} ×{n}")
    print("\nlocations:")
    for rel, lineno, ref in violations:
        print(f"  {rel}:{lineno}  {ref}")
    print("\nfix: replace with the authoritative opaque IRI (gloss with the real CCO label), or mark a "
          "deliberate rejection-context mention with a trailing 'coined-ok'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

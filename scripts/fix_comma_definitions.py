"""fix_comma_definitions — repair malformed genus+differentia definitions in the catalog.

Templates authored `EquivalentTo: <genus>, <differentia>` (comma). In Manchester a comma is a LIST of
SEPARATE equivalences, so `X ≡ A, B` asserts X ≡ A AND X ≡ B — over-strongly entailing A ≡ B, and NOT
a genus+differentia definition. It also serializes as standalone `owl:equivalentClass` axioms that the
grounding walk can't read (natural experiment 2026-07-10: comma-form grounds 0/35, `and`-form 341/343).

The fix is `to_equivalent`'s transform applied to the EquivalentTo body: split top-level conjuncts on
commas (restrictions carry no top-level commas — verified in evolve_rigor.to_equivalent), rejoin with
`and` → the proper `X ≡ (A ⊓ B)`. This RELAXES the theory (drops the A ≡ B entailment), so it can only
reduce unsat. `--apply` writes the catalog; default is a dry run.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.schema import load_catalog, save_catalog, CATALOG_FILE  # noqa: E402

_EQUIV = re.compile(r"(Class:\s*\{[^}]+\}\s+EquivalentTo:\s*)(.+)", re.S)


def _split_top_level(body: str) -> "list[str]":
    """Split on commas at paren/bracket depth 0 (defensive vs any nested restriction commas)."""
    parts, buf, depth = [], [], 0
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def convert(manchester: str) -> "str | None":
    m = _EQUIV.match(manchester.strip())
    if not m:
        return None
    body = m.group(2)
    if " and " in body or "," not in body:  # already an intersection, or single genus — leave it
        return None
    conjuncts = _split_top_level(body)
    if len(conjuncts) < 2:
        return None
    return m.group(1) + " and ".join(conjuncts)


def main() -> int:
    apply = "--apply" in sys.argv
    cat = load_catalog(CATALOG_FILE)
    n = 0
    for t in cat.templates:
        new = convert(t.manchester_template or "")
        if new and new != t.manchester_template:
            n += 1
            print(f"[{n}] {t.template_id}")
            print(f"    - {t.manchester_template.strip()[:150]}")
            print(f"    + {new.strip()[:150]}")
            if apply:
                t.manchester_template = new
    print(f"\n{'APPLIED' if apply else 'DRY RUN'}: {n} comma-form definitions → intersection")
    if apply and n:
        save_catalog(cat, CATALOG_FILE)
        print(f"   wrote {CATALOG_FILE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

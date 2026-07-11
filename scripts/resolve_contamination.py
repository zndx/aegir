"""resolve_contamination — one-time cleanup of fictional external references in the catalog.

The retired CCO_READABLE_ALIASES dict masked fictional external refs (cco:DirectiveICE, cco:BusinessEntity,
cco:has_part). Walking it back (with prejudice) exposes them. This resolves ONLY the unambiguous ones —
through the AUTHORITATIVE index (verified to exist), not a hardcoded alias:

  1. `cco:BFO_\\d+` (a BFO IRI mis-namespaced under cco:) → `bfo:\\d+`, verified in BFO.
  2. exact label match (abbrev-expanded) to a REAL external entity that EXISTS → its real IRI.
  3. a reference that is actually one of OUR sdg: classes (LLM wrote our concept under cco:) → sdg:.

Everything else — genuine judgments (DirectiveICE ≟ Prescriptive) and invalid relations (has_part, has_quality
are RO/absent) — is LEFT for the agent to re-author under the now-active verification gate. No silent guessing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.external_index import get_index  # noqa: E402
from aegir.ontology.schema import load_catalog, save_catalog, CATALOG_FILE  # noqa: E402

_REF = re.compile(r"\b(cco|bfo):([A-Za-z0-9_]+)")
_ABBREV = {"ice": "information content entity"}


def _label_query(local: str) -> str:
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local).replace("_", " ").lower().split()
    return " ".join(_ABBREV.get(w, w) for w in words)


def main() -> int:
    apply = "--apply" in sys.argv
    idx = get_index()
    cat = load_catalog(CATALOG_FILE)
    sdg_classes = {re.search(r"Class:\s*\{(\w+)", t.manchester_template or "").group(1)
                   for t in cat.templates if re.search(r"Class:\s*\{(\w+)", t.manchester_template or "")}

    resolved: dict[str, str] = {}   # fictional ref → real ref
    flagged: dict[str, int] = {}
    for t in cat.templates:
        for m in _REF.finditer(t.manchester_template or ""):
            ref, ns, local = m.group(0), m.group(1), m.group(2)
            if idx.exists(ref) or ref in resolved:
                continue
            # 1. BFO mis-namespaced under cco:
            bm = re.fullmatch(r"BFO_(\d+)", local)
            if ns == "cco" and bm and idx.exists(f"bfo:{bm.group(1)}"):
                resolved[ref] = f"bfo:{bm.group(1)}"
                continue
            # 2. exact abbrev-expanded label match to a real external entity — CROSS-NAMESPACE, because the LLM
            # routinely writes a BFO property under cco: (cco:has_participant) or vice-versa. The label resolves
            # uniquely to the real IRI regardless of the (wrong) prefix; take it only when unambiguous.
            hits = idx.search(_label_query(local))
            if len(hits) == 1:
                resolved[ref] = hits[0]
                continue
            # 3. actually one of OUR sdg: classes
            if local in sdg_classes:
                resolved[ref] = f"sdg:{local}"
                continue
            flagged[ref] = flagged.get(ref, 0) + 1

    print(f"=== contamination resolution ({len(resolved)} resolved, {len(flagged)} flagged for agent) ===")
    print("RESOLVED (via authoritative index — verified to exist):")
    for f, r in sorted(resolved.items()):
        print(f"    {f:36s} -> {r}  [{idx.kind(r) or 'sdg'}]")
    print("FLAGGED (genuine judgment / invalid relation → agent re-authors under the gate):")
    for f, n in sorted(flagged.items(), key=lambda x: -x[1]):
        print(f"    {f:36s} ×{n}")

    if apply and resolved:
        n = 0
        for t in cat.templates:
            man = t.manchester_template or ""
            new = man
            for f, r in resolved.items():
                new = re.sub(re.escape(f) + r"\b", r, new)
            if new != man:
                t.manchester_template = new
                n += 1
        save_catalog(cat, CATALOG_FILE)
        print(f"\nAPPLIED: rewrote {n} templates → catalog.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""validate_value_provenance — the in-flight lineage gate (#45, RISE-arc).

Extracts every GitTables-backed ``sdg:valueProvenance`` claim from the
integration overlays and validates each against the kvasir value index
(``kvasir find <idx> validate --file``): a claim that does not resolve is a
PROVENANCE-VALIDATION SIGNAL (kin of the census verdict and the
underspecification complaint — never silent). Census-cited and
AUTHORED-flagged annotations are counted and skipped (their authorities are
the census artifact and the flag itself).

    uv run python scripts/validate_value_provenance.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

KV = REPO / "components/kvasir/target/release/kvasir"
IDX = REPO / "build/foreign/gittables/values.kci"


def main() -> int:
    import rdflib
    from aegir.ontology.domain_index import DEFAULT_OVERLAY, integration_overlay_files

    SDG = rdflib.Namespace("https://signals.zndx.org/sdg#")
    g = rdflib.Graph()
    for f in [DEFAULT_OVERLAY, *integration_overlay_files()]:
        g.parse(str(f))
    claims: "list[tuple[str, str, str]]" = []   # (subject-frag, value-form, lineage)
    n_census = n_authored = 0
    for s, o in g.subject_objects(SDG.valueProvenance):
        text = str(o)
        frag = str(s).rsplit("#", 1)[-1]
        if text.startswith("census"):
            n_census += 1
            continue
        if text.upper().startswith("AUTHORED"):
            n_authored += 1
            continue
        # gtvs segment: "… · <form>@<lineage> (…)? · …"
        m = re.search(r"·\s*([^·]+?)@([^\s·(]+)", text)
        if m:
            claims.append((frag, m.group(1).strip(), m.group(2).strip()))
        else:
            print(f"⚑ MALFORMED provenance on {frag}: no form@lineage segment")
            return 1
    if not claims:
        print(f"no GitTables claims (census {n_census} · authored {n_authored})")
        return 0
    tsv = REPO / "build/value_provenance_claims.tsv"
    tsv.write_text("".join(f"{v}\t{lin}\n" for _, v, lin in claims))
    r = subprocess.run([str(KV), "find", str(IDX), "validate", "--file", str(tsv)],
                       capture_output=True, text=True)
    ok = unresolved = 0
    for line, (frag, _, _) in zip(r.stdout.splitlines(), claims):
        parts = line.split("\t")
        if parts[0] == "OK":
            ok += 1
            print(f"  ✓ {frag:<28} {parts[1]}@{parts[2]} · freq {parts[3]}")
        else:
            unresolved += 1
            print(f"  ⚑ UNRESOLVED {frag:<28} {parts[1]}@{parts[2]} — remediate the claim")
    print(f"value-provenance gate: {ok} OK · {unresolved} UNRESOLVED · "
          f"{n_census} census-cited · {n_authored} authored-flagged")
    return 1 if (unresolved or r.returncode not in (0, 1)) else 0


if __name__ == "__main__":
    raise SystemExit(main())

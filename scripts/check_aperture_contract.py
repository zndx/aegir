#!/usr/bin/env python
"""check_aperture_contract — the fiat SKOS contract on the admission surface (RH 2026-07-21).

Every point in the aperture qdrant collection SHALL be a skos:Concept carrying:
  1) skos:broader   — anchors are logically distinguishable, generally not top-concepts;
  2) skos:narrower  — sufficient conceptual gravity implies subordinate structure;
  3) skos:altLabel  — the chord's display form, equal to the IRI FRAGMENT (display is
     identity in the orienting instrument — the label-drift class is closed by form).

Mechanical floors, reported per anchor; --fix-mechanical authors the derivable parts into
the overlay TTL (altLabel := fragment; explicit skos:narrower from broader-inverses).
Missing broader on a top anchor is an AUTHORING decision — surfaced, never invented.

    uv run python scripts/check_aperture_contract.py [--fix-mechanical]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix-mechanical", action="store_true")
    a = ap.parse_args()
    from aegir.ontology import domain_index as DI
    vocab = DI.load_skos()
    client = DI._client(DI.DEFAULT_QDRANT_URL)
    pts, _ = client.scroll(DI.DEFAULT_APERTURE, limit=256, with_payload=True)
    narrower: "dict[str, list]" = {}
    for k, c in vocab.items():
        b = getattr(c, "broader", "") or ""
        if b:
            narrower.setdefault(b, []).append(str(k))
    rows = []
    for p in pts:
        iri = (p.payload or {}).get("iri") or ""
        frag = iri.rsplit("#", 1)[-1]
        c = vocab.get(iri)
        has_broader = bool(getattr(c, "broader", "")) if c else False
        has_narrower = bool(narrower.get(iri))
        alt = (getattr(c, "alt_label", "") or "") if c else ""
        alt_ok = alt == frag
        rows.append({"point": p.id, "iri": iri, "fragment": frag,
                     "in_vocab": c is not None, "broader": has_broader,
                     "narrower": has_narrower, "altLabel_eq_fragment": alt_ok,
                     "altLabel": alt})
    n = len(rows)
    ok_b = sum(r["broader"] for r in rows)
    ok_n = sum(r["narrower"] for r in rows)
    ok_a = sum(r["altLabel_eq_fragment"] for r in rows)
    print(f"aperture contract: {n} points · broader {ok_b}/{n} · narrower {ok_n}/{n} · "
          f"altLabel==fragment {ok_a}/{n}")
    for r in rows:
        flags = [k for k in ("broader", "narrower", "altLabel_eq_fragment") if not r[k]]
        if flags or not r["in_vocab"]:
            print(f"  ✘ {r['fragment'][:44]:<44} missing: "
                  f"{', '.join(flags) or 'NOT IN VOCAB'}")
    (REPO / "build/aperture_contract.json").write_text(json.dumps(
        {"n": n, "broader_ok": ok_b, "narrower_ok": ok_n, "altlabel_ok": ok_a,
         "rows": rows}, indent=1))
    print("→ build/aperture_contract.json")

    if a.fix_mechanical:
        ttl_p = DI.DEFAULT_OVERLAY
        t = ttl_p.read_text()
        fixed = 0
        for r in rows:
            if not r["in_vocab"]:
                continue
            blk_key = f"<{r['iri']}> a skos:Concept"
            if blk_key not in t:
                continue
            if not r["altLabel_eq_fragment"]:
                t = t.replace(blk_key + " ;",
                              blk_key + f' ;\n    skos:altLabel "{r["fragment"]}" ;', 1)
                fixed += 1
        # explicit narrower inverses for anchors that have children but no explicit property
        for r in rows:
            kids = narrower.get(r["iri"], [])
            if kids and f"<{r['iri']}> a skos:Concept" in t and "skos:narrower" not in \
                    t.split(f"<{r['iri']}> a skos:Concept", 1)[1][:600]:
                nar = " ,\n        ".join(f"<{k}>" for k in kids[:12])
                t = t.replace(f"<{r['iri']}> a skos:Concept ;",
                              f"<{r['iri']}> a skos:Concept ;\n    skos:narrower {nar} ;", 1)
                fixed += 1
        ttl_p.write_text(t)
        print(f"--fix-mechanical: {fixed} overlay edits (altLabel:=fragment; explicit narrower)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

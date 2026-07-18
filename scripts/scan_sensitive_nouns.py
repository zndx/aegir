#!/usr/bin/env python
"""Sensitive-noun scan — REAL UNIVERSALS, FICTIONAL PARTICULARS, enforced (RH 2026-07-03).

The corpus's attribution-clean claim extends from text to VALUES: real standards/protocols/units/kinds
are nomenclature and welcome; real-world PARTICULARS — organizations, brands, commercial products,
people, named facilities — must never be asserted (found live: 'Apple Inc' as an OWL individual typed
CorporateReputationInfluenceSituation in the published founding artifact). This scans every artifact
class against the deterministic denylist tier (`individuals.real_entity_hits` — the head of the brand
distribution; the engine-screened audit is the judgment tier and rides the registry re-seed):

  - src/aegir/ontology/individual_registry.json   (the ABox source)
  - src/aegir/ontology/entity_value_pools.json    (frozen legacy pool)
  - corpora/ontology/sdg-ontology.omn             (published Individual labels)
  - corpora/corpus/collections/**/*.md            (published chapter prose)
  - any --spine run dir's base_rows.parquet       (materialized cells)

Exit 0 = clean, 1 = hits (the P5 release gate + the standing regression gate once cleaned).
Deterministic. No LLM / GPU / network.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.individuals import _AMBIG_CTX, _BRANDS_AMBIG, real_entity_hits  # noqa: E402

# ── PROVENANCE is the deciding factor (RH 2026-07-14) ──────────────────────────────────────────────────
# A real particular (org / person / product) that is traceable to the EXPLICIT GitTables lineage is
# PERMITTED everywhere — tables AND prose — because its provenance is retained (the
# `provenance/gittables_value_sampling` strategy component + per-value source in the pools). Only a
# NON-provenanced real particular (an LLM naming a brand that never came from GitTables) stays a violation.
# This does NOT relax the fictional-particulars policy; it recognises lineage-backed values as accounted-for.
_GT_PROFILES = REPO / "build/gittables/value_profiles.json"


def gittables_provenance_tokens() -> "set[str]":
    """Lowercased tokens (full values + component words) present in the GitTables value pools — the explicit
    lineage set. A brand/particular whose token is in here is provenance-backed → not a violation (RH
    2026-07-18: 'lineage is critical when it comes to admitting nouns, and provable sourcing from gittables
    is a-ok'). Covers BOTH pool artifacts: the semantic-type pools (value_profiles.json) and the
    differentia-type pools (differentia_profiles.json — every value carries <sha>:<col> lineage)."""
    toks: "set[str]" = set()

    def _add(values) -> None:
        for pair in values or []:
            v = str(pair[0] if isinstance(pair, (list, tuple)) else pair).lower()
            toks.add(v)
            toks.update(w for w in re.findall(r"[a-z0-9&]+", v) if len(w) >= 2)

    try:
        for _st, rec in json.loads(_GT_PROFILES.read_text()).items():
            _add(rec.get("values"))
    except (OSError, ValueError):
        pass
    try:
        dp = json.loads((_GT_PROFILES.parent / "differentia_profiles.json").read_text())
        for _col, rec in (dp.get("pools") or {}).items():
            _add(rec.get("values"))
    except (OSError, ValueError):
        pass
    return toks


_PROV = gittables_provenance_tokens()


def _provenanced(brand: str) -> bool:
    """True when the matched brand token is accounted for by the explicit GitTables lineage."""
    return bool(_PROV) and brand.strip().lower() in _PROV


def scan_registry(path: Path) -> "list[tuple[str, str, str]]":
    if not path.exists():
        return []
    reg = json.loads(path.read_text())
    out = []
    for tid, rec in (reg.get("templates") or reg).items():
        cols = rec.get("columns") if isinstance(rec, dict) else None
        for col, vals in (cols or {}).items() if cols else []:
            for v, brand in real_entity_hits(list(vals)):
                if not _provenanced(brand):
                    out.append((f"{tid}.{col}", v, brand))
    return out


def scan_pools(path: Path) -> "list[tuple[str, str, str]]":
    if not path.exists():
        return []
    pools = json.loads(path.read_text())
    out = []
    for tid, cols in pools.items():
        if not isinstance(cols, dict):
            continue
        for col, vals in cols.items():
            if isinstance(vals, list):
                for v, brand in real_entity_hits([str(x) for x in vals]):
                    if not _provenanced(brand):
                        out.append((f"{tid}.{col}", v, brand))
    return out


def scan_omn(path: Path) -> "list[tuple[str, str, str]]":
    if not path.exists():
        return []
    labels = re.findall(r'rdfs:label "([^"]+)"', path.read_text())
    return [("omn:rdfs:label", v, b) for v, b in real_entity_hits(labels) if not _provenanced(b)]


def scan_chapters(root: Path) -> "tuple[list, list]":
    """(generated_hits, quoted_source_hits). Quoted FinePDFs excerpts (topic gists / style anchors)
    are LICENSED SOURCE REFERENCES — grounding, not our assertions; reported separately, not gated."""
    if not root.exists():
        return [], []
    ours, quoted = [], []
    for md in root.rglob("*.md"):
        text = md.read_text(errors="replace")
        low = text.lower()

        def _domain_initialism(brand: str) -> bool:
            """A 2-5-letter all-caps hit is DOMAIN USAGE (not the company) when the document
            itself contains a phrase whose initials spell it — e.g. 'satisfactory academic
            progress' → SAP is the chapter's own acronym (the measured false-positive: a
            financial-aid corpus flagged for the software company)."""
            b = brand.strip()
            if not (2 <= len(b) <= 5 and b.isupper() and b.isalpha()):
                return False
            words = re.findall(r"[a-z]+", low)
            init = "".join(w[0] for w in words)
            return b.lower() in init and any(
                "".join(w[0] for w in words[i:i + len(b)]) == b.lower()
                and sum(len(w) for w in words[i:i + len(b)]) >= len(b) * 3
                for i in range(len(words) - len(b) + 1))

        allowed: "set[str]" = set()
        # scan line-wise so the hit report carries usable context. Match on the FULL line — truncating
        # BEFORE the match manufactures a word boundary at the cut ('…record of intel|ligence' → \bintel\b
        # matches at end-of-string; the 2026-07-17 corpus-taxonomy false positives). Truncate only the report.
        for line in text.splitlines():
            for v, brand in real_entity_hits([line.strip()]):
                if _provenanced(brand):        # traceable to explicit GitTables lineage → permitted (RH 2026-07-14)
                    continue
                if brand in allowed or _domain_initialism(brand):
                    allowed.add(brand)
                    continue
                # heading case: every word capitalized ⇒ neither TitleCase NOR the
                # brand+CapitalizedWord context branch carries signal ('### … Shell Region …'
                # is exchanger anatomy, not the oil major). In headings, ambiguous-tier
                # brands count only with an explicit corporate suffix.
                if line.lstrip().startswith("#") and brand in _BRANDS_AMBIG and not re.search(
                        rf"(?i)\b{re.escape(brand)}\b\s+(?:inc|corp|llc|ltd|plc|group|"
                        rf"systems|health(?:care)?|cloud|labs?)\b", line):
                    continue
                (quoted if "FinePDFs" in line else ours).append(
                    (str(md.relative_to(root)), v[:120], brand))
    return ours, quoted


def scan_spine(run_dir: Path) -> "list[tuple[str, str, str]]":
    p = run_dir / "base_rows.parquet"
    if not p.exists():
        return []
    import pyarrow.parquet as pq
    out = []
    for r in pq.read_table(p, columns=["table_name", "col_name", "value"]).to_pylist():
        v = r.get("value")
        if v:
            for hit, brand in real_entity_hits([str(v)]):
                if not _provenanced(brand):
                    out.append((f"{r['table_name']}.{r['col_name']}", hit, brand))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine", default=None, help="optional spine run dir (scans base_rows.parquet)")
    ap.add_argument("--json-out", default=None, help="write the full hit list as JSON")
    ap.add_argument("--root", default=None,
                    help="scan THIS chapters tree only (flow-run candidates), not the committed corpora")
    a = ap.parse_args()

    ch_ours, ch_quoted = scan_chapters(Path(a.root) if a.root else REPO / "corpora/corpus/collections")
    sections = {"chapters": ch_ours} if a.root else {
        "individual_registry": scan_registry(REPO / "src/aegir/ontology/individual_registry.json"),
        "entity_value_pools": scan_pools(REPO / "src/aegir/ontology/entity_value_pools.json"),
        "corpora omn labels": scan_omn(REPO / "corpora/ontology/sdg-ontology.omn"),
        "corpora chapters (generated)": ch_ours,
    }
    if a.spine:
        sections[f"spine {a.spine}"] = scan_spine(Path(a.spine))

    total = 0
    for name, hits in sections.items():
        brands = sorted({b for _, _, b in hits})
        print(f"{name:30} {len(hits):5} hit(s)" + (f"  brands: {', '.join(brands[:10])}" if hits else ""))
        for loc, v, b in hits[:5]:
            print(f"    {b:12} {v!r:42} @ {loc}")
        if len(hits) > 5:
            print(f"    … {len(hits) - 5} more")
        total += len(hits)
    print(f"{'quoted FinePDFs excerpts':30} {len(ch_quoted):5} hit(s)  (licensed source references — "
          "grounding, not gated)")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {k: [{"where": w, "value": v, "brand": b} for w, v, b in hs] for k, hs in sections.items()},
            indent=1))
        print(f"→ {a.json_out}")
    print(f"\nTOTAL: {total} real-particular hit(s) — {'CLEAN' if total == 0 else 'the ABox must be fictional'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

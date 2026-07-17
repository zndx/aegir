#!/usr/bin/env python
"""emit_differentia_specs — derive the per-differentia-type HARVEST SPECS from the authored taxonomy.

One spec per distinct differentia property (the "differentia type"): its kind/xsd, the projected corpus
column names, the authored seed exemplars (maxsim query seeds), the CAS loop's semantic glosses, the
incumbent rows-classifier verdict, and the full species/genus usage map. The output is a BYTE-DETERMINISTIC
pure function of the taxonomy artifact (sorted keys/lists, no timestamps) so it lands in sdg-strategy as a
content-hashed component (targets/differentia_value_specs.json via strategy.manifest collect_targets) —
verifiable by rehash from Atelier with no aegir context, and consumable there DIRECTLY as the what-to-harvest
contract for the GitTables value-harvest capability (maxsim/NHSVM/CatBoost). gRPC carries only results.

    uv run --no-sync python scripts/emit_differentia_specs.py --taxonomy build/taxonomy/v05
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.entities import prop_name, _snake  # noqa: E402  (spec ≡ actual column projection)

_VERSION = "dvs-v1-2026-07-17"
_GLOSS_CAP = 6                     # deduped glosses kept per spec (sorted; enough context, bounded bytes)


def _norm_restriction_fn():
    """emit_taxonomy._norm_restriction — imported, not replicated, so spec kind/xsd ≡ the overlay's."""
    spec = importlib.util.spec_from_file_location("emit_taxonomy", REPO / "scripts" / "emit_taxonomy.py")
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m._norm_restriction


def _xsd_of(rr: str) -> str:
    xm = re.search(r"xsd:(\w+)", rr)
    return xm.group(1) if xm else "string"


def _regex_semantic_type(col: str, xsd: str) -> "str | None":
    """The INCUMBENT rows._semantic_type_of verdict for this column — null means the mechanical
    fallback today, i.e. exactly what the harvest displaces. Informational (prioritization)."""
    try:
        from aegir.ontology.rows import _semantic_type_of
        return _semantic_type_of(col, f"xsd:{xsd}", "")
    except Exception:  # noqa: BLE001
        return None


def build_specs(genera_dir: Path) -> "tuple[dict, dict]":
    norm = _norm_restriction_fn()
    prop_kind: "dict[str, str]" = {}                       # first-wins, mirroring emit_taxonomy props.setdefault
    raw: "dict[str, dict]" = {}
    n_acc = n_skip = 0
    for gf in sorted(glob.glob(str(genera_dir / "genera" / "*.json"))):
        r = json.loads(Path(gf).read_text())
        g = r["genus"]
        for s, d in r["differentiae"].items():
            if not d.get("accepted") or s == g:            # same filter as the overlay
                continue
            rr, kind, fillers = norm(d["restriction"])
            if prop_kind.setdefault(d["property"], kind) != kind:
                n_skip += 1                                # mixed-kind record: excluded from the overlay too
                continue
            n_acc += 1
            p = raw.setdefault(d["property"], {"kind": kind, "xsds": Counter(), "values": set(),
                                               "targets": set(), "glosses": set(), "usage": []})
            p["xsds"][_xsd_of(rr) if kind == "data" else "anyURI"] += 1
            p["values"].update(re.findall(r'value\s+"([^"]+)"', rr))
            p["targets"].update(fillers)
            if d.get("why"):
                p["glosses"].add(" ".join(d["why"].split())[:300])
            p["usage"].append({"species": s, "genus": g, "restriction": rr})

    specs: "dict[str, dict]" = {}
    for prop, p in sorted(raw.items()):
        pn = prop_name(prop)
        col = _snake(pn)
        xsd = p["xsds"].most_common(1)[0][0]
        seeds = sorted(p["values"])
        spec = {
            "kind": p["kind"],
            "xsd": xsd,
            "column": {"snake": col, "camel": pn},
            "seed_exemplars": seeds,
            "object_targets": sorted(f"sdg:{t}" for t in p["targets"]),
            "gloss": sorted(p["glosses"])[:_GLOSS_CAP],
            "regex_semantic_type": _regex_semantic_type(col, xsd) if p["kind"] == "data" else None,
            "usage": sorted(p["usage"], key=lambda u: (u["genus"], u["species"])),
        }
        if p["kind"] == "data" and len(seeds) <= 1:
            spec["pool_risk"] = "constant_column"          # the known watch-item; harvest exists to widen these
        elif p["kind"] == "data" and len(seeds) <= 3:
            spec["pool_risk"] = "narrow"
        specs[prop] = spec

    stats = {"n_properties": len(specs), "n_accepted_differentiae": n_acc, "n_mixed_kind_skipped": n_skip,
             "n_object": sum(1 for s in specs.values() if s["kind"] == "object"),
             "n_uncovered_by_regex": sum(1 for s in specs.values()
                                         if s["kind"] == "data" and s["regex_semantic_type"] is None),
             "n_constant_column_risk": sum(1 for s in specs.values()
                                           if s.get("pool_risk") == "constant_column")}
    return specs, stats


def component(specs: dict, stats: dict, taxonomy_tag: str) -> dict:
    return {
        "component": "targets/differentia_value_specs",
        "purpose": ("One HARVEST SPEC per distinct differentia type (property) of the authored genus+"
                    "differentia taxonomy: the what-to-harvest contract for grounding differentia-column "
                    "VALUES in real GitTables columns. Captured in sdg-strategy so (a) the harvest is "
                    "REPRODUCIBLE — a pure function of (this component @sha, GitTables snapshot, harvester "
                    "version) — and (b) Atelier consumes it DIRECTLY by strategy ref (verifiable by rehash, "
                    "no aegir context; gRPC carries only results, never the contract)."),
        "version": _VERSION,
        "derivation": {
            "generator": "scripts/emit_differentia_specs.py (aegir)",
            "inputs": {"taxonomy": f"build/taxonomy/{taxonomy_tag}/genera/*.json",
                       "authoring_prompt_version": "differentia-v2-2026-07-15"},
            "consistency": ("kind/xsd via emit_taxonomy._norm_restriction (spec ≡ overlay semantics); "
                            "column names via entities.prop_name/_snake (spec ≡ corpus projection)"),
            "determinism": ("byte-deterministic: sorted keys/lists, no timestamps; re-run on the same "
                            "taxonomy artifact reproduces the component hash"),
            "stats": stats,
        },
        "harvest_contract": {
            "consumer": ("Atelier's column-type ensemble (maxsim over value embeddings + NHSVM + CatBoost) "
                         "run over the GitTables corpus — NOT forked into aegir; federated capability"),
            "transport": {
                "face": "zndx.engine.v1 (signals-protocol submodule — the shared federation face)",
                "staging": ["increment-1: aegir-local mock harvester proves contract+seam (no proto change)",
                            "increment-2: first-class Harvest RPC added to signals-protocol (additive-only), "
                            "Atelier servicer on :50251"],
            },
            "request": {"property": "spec key", "column": "spec.column", "xsd": "spec.xsd",
                        "kind": "spec.kind", "genera": "from spec.usage", "siblings": "co-species context",
                        "seed_exemplars": "spec.seed_exemplars — the maxsim query seeds",
                        "gloss": "spec.gloss — embedding-side semantic description"},
            "response": {"values": "[{value, source: '<table_sha1>:<column>', confidence}]",
                         "pool_confidence": "ensemble confidence for the pool as a whole",
                         "method": "which ensemble member(s) matched"},
            "acceptance": {
                "pool_floor": "min_distinct >= 4 admitted values (gtvs filter_heuristics)",
                "confidence_floor": "below floor -> column stays FLAGGED, never fabricated (the same "
                                    "below-frontier discipline as the 5 residual differentiae)",
                "pii": "person-name-like pools inherit the gtvs sensitivity-membrane gate before injection",
            },
            "lineage": ("per-value <table_sha1>:<column> retained; policy = provenance/"
                        "gittables_value_sampling (provenance succeeds exclusion, RH 2026-07-13); harvested "
                        "pools land content-addressed in build/gittables/differentia_profiles.json and are "
                        "consumed by the SAME unified core (rows._gittables_value) as the name-rule pools"),
            "closure": ("NHSVM is the SAME discriminator whose resolution set the taxonomy's rate-distortion "
                        "frontier — so a spec that harvests nothing above floor is evidence the differentia "
                        "sits BELOW discriminator resolution (a taxonomy signal), while a harvested pool "
                        "fills the value layer at exactly the resolution the type layer was authored against"),
        },
        "spec_schema": {
            "kind": "data | object (object -> the differentia projects as an FK column, not a value pool)",
            "xsd": "majority normalized xsd of the authored restrictions",
            "column": "the projected corpus column names (snake for the dominant register, camel otherwise)",
            "seed_exemplars": "authored enum values — maxsim seeds AND the current in-corpus values",
            "object_targets": "sdg: filler classes (object kind)",
            "gloss": "deduped authored 'why' sentences from the differentia-authoring CAS loop",
            "regex_semantic_type": ("the incumbent rows._semantic_type_of verdict; null == mechanical "
                                    "fallback today == exactly what the harvest displaces"),
            "pool_risk": "constant_column (1 seed) | narrow (<=3) — the pools harvest must widen first",
            "usage": "[{species, genus, restriction}] — the full grounding map (audit + sibling context)",
        },
        "specs": specs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", default=str(REPO / "build" / "taxonomy" / "v05"))
    ap.add_argument("--out", default=str(REPO / "build" / "taxonomy" / "differentia_specs.json"))
    a = ap.parse_args()
    tdir = Path(a.taxonomy)
    specs, stats = build_specs(tdir)
    doc = component(specs, stats, tdir.name)
    out = Path(a.out)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{stats['n_properties']} differentia-type specs ({stats['n_object']} object · "
          f"{stats['n_uncovered_by_regex']} uncovered-by-regex · {stats['n_constant_column_risk']} "
          f"constant-column risk) from {stats['n_accepted_differentiae']} accepted differentiae → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

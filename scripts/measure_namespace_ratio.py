"""measure_namespace_ratio — the over-rotation instrument.

The sdg: namespace is the ontology's DOMAIN VALUE-ADD: BFO/CCO are upper/structural plumbing we reuse, but
the domain concepts (and most relations) are ours to coin — GROUNDED into BFO/CCO, never fabricated inside it
([[coined_alias_ban]], [[bfo_cco_grounding_mandate]]). A pipeline that stops creating sdg: terms while still
ingesting novel domain content has OVER-ROTATED — corrected the earlier "coin blindly" gap too far the other way.

This measures two things as we process inputs:
  1. FOOTPRINT — the sdg: vs real-BFO/CCO share of a catalog / derive run, split class vs relation.
  2. TREND — across a series of batches, the sdg-creation rate r_i, its first derivative Δr (velocity) and
     SECOND derivative Δ²r (acceleration). Healthy saturation decelerates GENTLY toward a positive floor
     (later passages reuse existing sdg: terms); over-rotation is r collapsing toward ~0 (novel content stops
     minting sdg:). We alarm on: r below --floor, OR a sharp negative Δ²r driving r toward the floor.

    uv run --no-sync python scripts/measure_namespace_ratio.py catalog                       # footprint of the live catalog
    uv run --no-sync python scripts/measure_namespace_ratio.py run <derive_run_dir>          # footprint of a derive run's entities
    uv run --no-sync python scripts/measure_namespace_ratio.py track <label> <derive_run_dir>  # append to the series + print the trend
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
SERIES = REPO / "build" / "namespace_ratio_series.jsonl"
REF = re.compile(r"\b(sdg|cco|bfo|fhir):([A-Za-z0-9_]+)")
_REL_KW = r"(?:some|only|min|max|exactly|value)"


def catalog_footprint(templates) -> dict:
    """Distinct refs by namespace, split into class/filler slots vs relation (object-property) slots."""
    cls = {"sdg": set(), "cco": set(), "bfo": set(), "fhir": set()}
    rel = {"sdg": set(), "cco": set(), "bfo": set(), "fhir": set()}
    for t in templates:
        man = t.manchester_template or ""
        for pfx, local in REF.findall(man):
            ref = f"{pfx}:{local}"
            is_rel = re.search(rf"{re.escape(ref)}\s+{_REL_KW}\b", man)
            (rel if is_rel else cls)[pfx].add(ref)
    return _summ(cls, rel)


def run_footprint(run_dir: Path) -> dict:
    """A derive run's newly-authored entities (entities/*.json: name/label/genus). The GENUS is the grounding;
    an sdg: head grounded to a bfo:/cco: genus is a healthy domain coinage — that is what we count."""
    cls = {"sdg": set(), "cco": set(), "bfo": set(), "fhir": set()}
    rel = {"sdg": set(), "cco": set(), "bfo": set(), "fhir": set()}
    ent_dir = run_dir / "entities"
    heads, grounded = 0, 0
    for p in sorted(ent_dir.glob("*.json")) if ent_dir.exists() else []:
        for e in _load_entities(p):
            head = e.get("name") or e.get("label") or ""
            pfx = head.split(":", 1)[0] if ":" in head else "sdg"  # bare heads are our own
            if pfx in cls:
                cls[pfx].add(head)
                heads += 1
                genus = str(e.get("genus") or "")
                if pfx == "sdg" and re.match(r"(bfo|cco):", genus):
                    grounded += 1
    out = _summ(cls, rel)
    out["sdg_heads"], out["sdg_heads_grounded"] = heads, grounded
    return out


def _load_entities(p: Path) -> list:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    if isinstance(d, dict):
        d = d.get("entities", d.get("classes", [d]))
    return [e for e in d if isinstance(e, dict)]


def _summ(cls: dict, rel: dict) -> dict:
    ct = sum(len(v) for v in cls.values()) or 1
    rt = sum(len(v) for v in rel.values()) or 1
    return {
        "class": {k: len(v) for k, v in cls.items()},
        "relation": {k: len(v) for k, v in rel.items()},
        "sdg_class_share": len(cls["sdg"]) / ct,
        "sdg_relation_share": len(rel["sdg"]) / rt,
        # r = sdg share of ALL distinct terms — the creation ratio the trend tracks
        "sdg_share": (len(cls["sdg"]) + len(rel["sdg"])) / (ct + rt),
    }


def trend(rs: list, floor: float) -> dict:
    """First (velocity) + second (acceleration) derivative of the creation-ratio series; over-rotation verdict."""
    dr = [rs[i] - rs[i - 1] for i in range(1, len(rs))]
    d2r = [dr[i] - dr[i - 1] for i in range(1, len(dr))]
    r = rs[-1]
    v = dr[-1] if dr else 0.0
    a = d2r[-1] if d2r else 0.0
    # over-rotation: at/below the floor, OR decelerating (v<0, a<0) AND projected to cross the floor within 3 steps
    proj = r + 3 * v
    over = r <= floor or (v < 0 and a < 0 and proj <= floor)
    return {"r": r, "velocity": v, "acceleration": a, "projected_3": proj,
            "verdict": "OVER-ROTATION" if over else "healthy", "series": rs}


def _print_fp(fp: dict, title: str) -> None:
    print(f"=== {title} ===")
    print("  class    :", fp["class"], f"→ sdg {fp['sdg_class_share']:.1%}")
    print("  relation :", fp["relation"], f"→ sdg {fp['sdg_relation_share']:.1%}")
    if "sdg_heads" in fp:
        g = fp["sdg_heads_grounded"]
        print(f"  derived sdg heads: {fp['sdg_heads']}  grounded-to-bfo/cco genus: {g} "
              f"({g / max(fp['sdg_heads'], 1):.0%})")
    print(f"  sdg creation-ratio r = {fp['sdg_share']:.3f}")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    mode = args[0]
    if mode == "catalog":
        from aegir.ontology.schema import load_catalog, CATALOG_FILE
        _print_fp(catalog_footprint(load_catalog(CATALOG_FILE).templates), "catalog footprint")
    elif mode == "run" and len(args) >= 2:
        _print_fp(run_footprint(Path(args[1])), f"derive run {args[1]}")
    elif mode == "track" and len(args) >= 3:
        label, run_dir = args[1], Path(args[2])
        fp = run_footprint(run_dir)
        rec = {"label": label, "run": str(run_dir), "r": fp["sdg_share"],
               "sdg_class_share": fp["sdg_class_share"], "sdg_relation_share": fp["sdg_relation_share"]}
        with SERIES.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        rs = [json.loads(x)["r"] for x in SERIES.read_text().splitlines() if x.strip()]
        _print_fp(fp, f"tracked: {label}")
        t = trend(rs, floor=0.15)
        print(f"\n  TREND over {len(rs)} batches: r={t['r']:.3f}  velocity={t['velocity']:+.3f}  "
              f"acceleration={t['acceleration']:+.3f}  proj(+3)={t['projected_3']:.3f}  → {t['verdict']}")
        if t["verdict"] != "healthy":
            print("  ⚠  sdg creation collapsing — the pipeline may be over-rotated toward reusing BFO/CCO; "
                  "inspect the derive prompt / genus-selection, not just the gate.")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

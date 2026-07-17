#!/usr/bin/env python
"""harvest_differentia_values — run the differentia value-harvest pre-pass (see aegir.ontology.differentia_harvest).

Consumes the sdg-strategy spec component BY REF, harvests real GitTables pools for the differentia
columns (local baseline now; --harvester federated once the Atelier servicer lands), writes
build/gittables/differentia_profiles.json with the reproducibility triple + per-value lineage.

    uv run --no-sync python scripts/harvest_differentia_values.py --sample 8000 --seed 13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology import differentia_harvest as dh  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harvester", choices=("local", "federated"), default="local")
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ref", default=None, help="strategy ref (shadow runs); default = checked-out CURRENT")
    a = ap.parse_args()

    component = dh.load_component(a.ref)
    specs = dh.harvestable(component)
    print(f"spec component {component['version']} → {len(specs)} harvestable differentia types "
          f"(of {component['derivation']['stats']['n_properties']})", flush=True)
    if a.harvester == "federated":
        pools, meta = dh.harvest_federated(specs)
    else:
        pools, meta = dh.harvest_local(specs, a.sample, a.seed)
    out = dh.write_profiles(pools, meta, component)
    n_vals = sum(p["n"] for p in pools.values())
    covered = sorted(pools)
    print(f"harvested {len(pools)}/{len(specs)} differentia types above floor · {n_vals} values "
          f"({meta['name_exact']} exact-name cols + {meta['token_subset']} seed-corroborated) → {out}")
    print(f"covered: {', '.join(covered[:12])}{' …' if len(covered) > 12 else ''}")
    print(f"below-floor (stay flagged, never fabricated): {len(specs) - len(pools)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

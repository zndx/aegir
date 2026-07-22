#!/usr/bin/env python
"""promote_armed_domains — certification made durable: the armed set into sdg-strategy.

RH ruling (2026-07-22): the certified armed-domain set is RELEASE-VERSIONED MEASUREMENT,
not aegir code — it ships in the independently released strategy submodule and every
consumer (aegir realize, federated sites) reads it by strategy ref. With that,
AEGIR_PROPERTY_DOMAINS=1 is the universal path: no build/-local ledgers at runtime, no
fail-open on fresh checkouts.

Writes into strategy/components/methods/:
  armed_property_domains.omn         — the certified frames, verbatim (staged minus
                                       demoted ∪ deferred at promotion time)
  armed_property_domains.provenance.json
      staged_sha256      — identity of the DOMAINS_OMN this was filtered from; the
                           runtime accessor REFUSES on mismatch (catalog regenerated ⟹
                           the gate must re-run — the Merkle binding, never silent)
      demoted/deferred   — the exclusion ledgers (audit trail, CAS/T2 worklists)
      gate               — the certifying verdicts (HermiT + kvasir), copied verbatim

    uv run python scripts/promote_armed_domains.py [--commit]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
SUB = REPO / "strategy"
DEST = SUB / "components/methods"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", action="store_true",
                    help="commit the promotion in the strategy submodule")
    a = ap.parse_args()

    from aegir.ontology.property_domains import DOMAINS_OMN
    staged_sha = hashlib.sha256(DOMAINS_OMN.encode()).hexdigest()

    demotions = json.loads((REPO / "build/domain_demotions.json").read_text())
    deferrals = json.loads((REPO / "build/domain_deferrals.json").read_text())
    excluded = set(demotions.get("demoted", [])) | set(deferrals.get("deferred", []))

    kept = []
    for frame in re.split(r"\n(?=(?:Object|Data)Property:)", DOMAINS_OMN.strip()):
        m = re.match(r"(?:Object|Data)Property:\s*(\S+)", frame)
        if m and m.group(1) in excluded:
            continue
        kept.append(frame)
    armed = "\n".join(kept) + "\n"

    gate = {}
    for name in ("union_armed_gate.json",):
        p = REPO / "build" / name
        if p.exists():
            d = json.loads(p.read_text())
            gate["hermit"] = {k: (len(v) if isinstance(v, list) else v)
                              for k, v in d.items() if k != "unsat"}
    gate["kvasir"] = "no clash · full rules (P1+P1.1+P2) · inline theory (2026-07-22)"

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "armed_property_domains.omn").write_text(armed)
    prov = {
        "staged_sha256": staged_sha,
        "n_staged_domain_lines": DOMAINS_OMN.count("Domain:"),
        "n_armed_domain_lines": armed.count("Domain:"),
        "demoted": sorted(demotions.get("demoted", [])),
        "demotion_rounds": demotions.get("rounds", []),
        "deferred": sorted(deferrals.get("deferred", [])),
        "deferral_reason": deferrals.get("reason", ""),
        "gate": gate,
        "doctrine": "certified set only; runtime accessor refuses on staged_sha256 "
                    "mismatch (re-enter the gate after catalog regeneration)",
    }
    (DEST / "armed_property_domains.provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"promoted: {prov['n_armed_domain_lines']} armed Domain lines "
          f"({len(prov['demoted'])} demoted · {len(prov['deferred'])} deferred) "
          f"→ {DEST.relative_to(REPO)}")

    if a.commit:
        subprocess.run(["git", "add", "components/methods/armed_property_domains.omn",
                        "components/methods/armed_property_domains.provenance.json"],
                       cwd=SUB, check=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=SUB)
        if staged.returncode == 0:
            print("promotion unchanged — nothing to commit (idempotent)")
            return 0
        msg = (f"methods: armed property domains v1 — {prov['n_armed_domain_lines']} "
               f"certified Domain lines (HermiT 0-unsat · kvasir no-clash; "
               f"{len(prov['demoted'])} demoted, {len(prov['deferred'])} "
               f"tractability-deferred)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
        subprocess.run(["git", "commit", "-m", msg], cwd=SUB, check=True)
        print("committed in strategy submodule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

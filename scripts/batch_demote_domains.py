#!/usr/bin/env python
"""batch_demote_domains — class-level batch demotion for the arming gate.

The per-witness demotion loop converges linearly in bad-frames-per-class (kvasir emits
ONE justification per unsat class, first-derivation-wins), which plateaus when classes
carry dozens of armed data properties. This pass closes the product space in one sweep:

    for every witnessed-unsat class C with an own-side anchoring s(C)
    (bare parents under the full inline theory), demote EVERY armed property
    p used in C's restrictions whose domain lifts to a side in conflict
    with s(C).

Sound direction: C's own anchoring is the identity (post-reconciliation, catalog-backed);
an armed usage-derived domain contradicting it is the accretion. Demotions are cumulative
in build/domain_demotions.json and stay CAS-review work, never deletions.

    uv run python scripts/batch_demote_domains.py \\
        [--witnesses build/armed_witnesses.json] [--omn build/unified_armed/sdg-ontology.omn]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "build/domain_demotions.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--witnesses", type=Path, default=REPO / "build/armed_witnesses.json")
    ap.add_argument("--omn", type=Path, default=REPO / "build/unified_armed/sdg-ontology.omn")
    a = ap.parse_args()

    omn = a.omn.read_text()
    rep = json.loads(a.witnesses.read_text())
    unsat = {w["name"] for w in rep.get("witnesses", []) if w["kind"] == "unsat_class"}
    if not unsat:
        print("no unsat witnesses — nothing to demote")
        return 0

    from derive_category_cuts import doc_taxonomy, module_taxonomy, BASE_DISJOINT, norm
    parents, _ = doc_taxonomy(omn)
    for c, ps in module_taxonomy().items():
        parents.setdefault(c, set()).update(ps)
    disjoint = {frozenset(p) for p in BASE_DISJOINT}
    for m in re.finditer(r"^DisjointClasses:\s*([^\n]+)$", omn, flags=re.M):
        atoms = [norm(x.strip()) for x in m.group(1).split(",")]
        if len(atoms) == 2 and all(re.match(r"^\w+:[\w.-]+$", x) for x in atoms):
            disjoint.add(frozenset(atoms))

    memo: dict = {}

    def anc(c, seen=None):
        if c in memo:
            return memo[c]
        seen = seen or set()
        if c in seen:
            return set()
        seen.add(c)
        out = {c}
        for p_ in parents.get(c, ()):
            out |= anc(p_, seen)
        memo[c] = out
        return out

    def conflicting(x, y):
        ax, ay = anc(x), anc(y)
        for pair in disjoint:
            u, v = tuple(pair)
            if (u in ax and v in ay) or (v in ax and u in ay):
                return True
        return False

    # armed table: property → domain (from the armed doc's OWN Domain lines, so demotions
    # already applied upstream are naturally absent)
    armed_dom: dict = {}
    for m in re.finditer(r"^(?:Object|Data)Property:\s*(sdg:[\w.-]+)\n"
                         r"((?:    [^\n]*\n)*)", omn, flags=re.M):
        dm = re.search(r"^\s*Domain:\s*(\S+?)\s*$", m.group(2), flags=re.M)
        if dm:
            armed_dom[m.group(1)] = norm(dm.group(1))

    # per-class: restriction properties (both frame shapes) + bare parents
    frame_rx = re.compile(r"^Class:\s*(<[^>\n]+>|(?:sdg|bfo|cco):[\w.-]+)([^\n]*)\n"
                          r"((?:    [^\n]*\n)*)", re.M)
    used_props: dict = {}
    bare_parents: dict = {}
    bare = re.compile(r"^(?:bfo:\d{7}|cco:\w+|sdg:[\w.-]+)$")
    for m in frame_rx.finditer(omn):
        cls = norm(m.group(1)).rsplit("#", 1)[-1]
        cls = cls if ":" in cls else cls
        cname = norm(m.group(1))
        local = cname.split(":", 1)[-1]
        body = (m.group(2) + "\n" + m.group(3))
        ps = used_props.setdefault(local, set())
        for pm in re.finditer(r"(sdg:[\w.-]+)\s+(?:some|only|exactly|min|max|value)\b", body):
            ps.add(pm.group(1))
        bp = bare_parents.setdefault(local, set())
        for line in body.splitlines():
            sm = re.match(r"^\s*(?:SubClassOf|EquivalentTo):\s*(.*)$", line)
            if not sm:
                continue
            for it in norm(sm.group(1)).split(","):
                it = it.strip()
                if bare.match(it):
                    bp.add(it)
                elif " and " in it:
                    # ≡-conjunct genera ARE the class's identity anchoring (the recurring
                    # conjunct-lift lesson) — without this, Course ≡ bfo:0000031 ⊓ … has
                    # no own-side and the batch pass skips it.
                    for c_ in re.split(r"\s+and\s+", it):
                        c_ = c_.strip().strip("()")
                        if bare.match(c_):
                            bp.add(c_)

    prior = json.loads(OUT.read_text()) if OUT.exists() else {"demoted": [], "rounds": []}
    demoted = set(prior.get("demoted", []))
    before = len(demoted)
    per_class = 0
    for cls in sorted(unsat):
        own = bare_parents.get(cls, set())
        if not own:
            continue
        hit = False
        for p in sorted(used_props.get(cls, set())):
            d = armed_dom.get(p)
            if not d:
                continue
            if any(conflicting(d, o) for o in own):
                demoted.add(p)
                hit = True
        per_class += hit
    new = len(demoted) - before
    prior["demoted"] = sorted(demoted)
    prior.setdefault("rounds", []).append(
        {"batch": True, "unsat_classes": len(unsat),
         "classes_with_conflicting_armed_props": per_class, "newly_demoted": new})
    OUT.write_text(json.dumps(prior, indent=1))
    print(f"batch: {new} new demotions (total {len(demoted)}) from {len(unsat)} unsat "
          f"classes ({per_class} carried conflicting armed props)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

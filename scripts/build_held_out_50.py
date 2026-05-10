#!/usr/bin/env python
"""Generate the held-out 50-ontology evaluation set for P5.

25 good ontologies + 25 bad ontologies. The good set composes from
the catalog with reasonable slot fillers and drawn from multiple
branches (so R_D engages). The bad set deliberately produces
malformed compositions to exercise the R_A=0 hard-gate path.

The set is **separate from the C1 30-ontology test set**: that one
trained the verifier weights; this one is for held-out P5 policy
evaluation, never seen during training. Authoring the set now —
before P5 begins — keeps it leakage-free.

Output: ``tests/p5_held_out_50/labels.json`` plus 50 prompt JSON
records readable by ``aegir.rl.eval.evaluate``.

Run::

    LD_LIBRARY_PATH=/tmp/jvm-libs uv run --no-sync python scripts/build_held_out_50.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aegir.ontology.schema import load_catalog  # noqa: E402

CATALOG = "src/aegir/ontology/catalog/combined.json"
OUT_DIR = Path("tests/p5_held_out_50")


# Branch-aligned slot fillers. The prompt-text describes the
# scenario the policy is asked to construct, biased so that good
# compositions can satisfy it.
SCENARIOS = [
    ("compliant_lab_run",
     "A compliant lab run that records a reading, attests to a SOC2 control, and emits provenance."),
    ("ebpf_observability",
     "An eBPF program that attaches to a kernel hook and captures a syscall under audit."),
    ("column_lineage_for_pii",
     "Column-level lineage from a PII source column to a derived dataset, governed by a GDPR directive."),
    ("dst_belief_combination",
     "A belief structure combining two evidence pieces via Dempster's rule for a privacy claim."),
    ("schema_evolution_event",
     "A schema migration that adds a column at version v3, governed by a backwards-compatibility directive."),
    ("telemetry_anomaly",
     "A telemetry span observing an anomaly, supported by evidence and triggering an alert."),
    ("medical_directive_hipaa",
     "A HIPAA technical safeguard governing the storage of PHI columns in a clinical-trial dataset."),
    ("attestation_audit",
     "An attestation activity producing audit evidence about an artifact's compliance to NIST 800-53."),
    ("schemaorg_person_dataset",
     "A schema.org-aligned dataset describing person columns: name, email, affiliation."),
    ("kernel_security_event",
     "A kernel security event observed by an XDP program, attributed to a suspicious syscall."),
]


def good_composition(catalog, _scenario_id: str, rng: random.Random) -> list[dict]:
    """Compose a 5-7 template ontology drawing from at least two
    branches, with realistic slot fillers."""
    by_branch: dict[str, list] = {}
    for t in catalog.templates:
        if not t.is_complex:
            continue
        b = t.provenance.get("branch", "?")
        by_branch.setdefault(b, []).append(t)

    branch_names = list(by_branch.keys())
    rng.shuffle(branch_names)
    chosen_branches = branch_names[:rng.randint(2, 4)]
    out: list[dict] = []
    target_n = rng.randint(5, 7)
    for _ in range(target_n):
        branch = rng.choice(chosen_branches)
        tmpl = rng.choice(by_branch[branch])
        fillers = {
            slot: rng.choice([
                "sdg:LabRun", "sdg:Sample", "sdg:Reading",
                "sdg:HipaaRule", "sdg:ColumnPolicy",
                "sdg:eBPFProgram", "sdg:KernelHook", "sdg:Syscall",
                "sdg:Transformation", "sdg:LineageEdge",
                "sdg:Evidence", "sdg:Claim", "sdg:BeliefInterval",
                "cco:Artifact", "bfo:0000015",
            ])
            for slot in tmpl.slot_types.keys()
        }
        out.append({"template_id": tmpl.template_id, "slot_fillers": fillers})
    return out


def bad_composition(catalog, _scenario_id: str, rng: random.Random) -> list[dict]:
    """Return a deliberately malformed composition. Three failure
    modes are sampled uniformly:

    - ``invalid_template_id``: emit a non-existent template_id (R_A=0).
    - ``missing_slots``: omit required slots (R_A=0).
    - ``trivial_basic``: pick only F1-subclass templates with
      identical stub fillers (R_A=1 but R_B/R_C/R_D collapse).
    """
    mode = rng.choice(["invalid_template_id", "missing_slots", "trivial_basic"])
    out: list[dict] = []
    if mode == "invalid_template_id":
        for _ in range(4):
            out.append({
                "template_id": f"completely_invented_template_{rng.randint(0, 9999)}",
                "slot_fillers": {"X": "sdg:Stub"},
            })
    elif mode == "missing_slots":
        for _ in range(4):
            tmpl = rng.choice([t for t in catalog.templates if len(t.slot_types) >= 2])
            out.append({
                "template_id": tmpl.template_id,
                "slot_fillers": {},
            })
    else:  # trivial_basic
        basics = [t for t in catalog.templates if t.template_id.endswith("_basic")]
        if not basics:
            basics = catalog.templates[:5]
        for _ in range(4):
            tmpl = rng.choice(basics)
            out.append({
                "template_id": tmpl.template_id,
                "slot_fillers": {s: "sdg:Stub" for s in tmpl.slot_types.keys()},
            })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(CATALOG)
    rng = random.Random(11)

    prompts: list[dict] = []
    n_good = 25
    n_bad = 25
    for i in range(n_good):
        scenario_id, scenario_text = SCENARIOS[i % len(SCENARIOS)]
        composition = good_composition(catalog, scenario_id, rng)
        prompts.append({
            "ontology_id": f"good_{i:02d}_{scenario_id}",
            "scenario_text": scenario_text,
            "label": 1,
            "expected_composition": composition,
        })
    for i in range(n_bad):
        scenario_id, scenario_text = SCENARIOS[i % len(SCENARIOS)]
        composition = bad_composition(catalog, scenario_id, rng)
        prompts.append({
            "ontology_id": f"bad_{i:02d}_{scenario_id}",
            "scenario_text": scenario_text,
            "label": 0,
            "expected_composition": composition,
        })

    (OUT_DIR / "labels.json").write_text(
        json.dumps([p for p in prompts], indent=2)
    )
    print(f"wrote {len(prompts)} prompts (25 good + 25 bad) to {OUT_DIR}/labels.json")
    print(f"catalog used: {len(catalog.templates)} templates ({catalog.version})")


if __name__ == "__main__":
    main()

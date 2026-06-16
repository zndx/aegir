"""ContractGate — the meta-harness `gate` effector (deterministic).

Runs the committed contract verifiers on ONE construct (a CatalogTemplate) and
returns the spine's signal vector. This is the membrane (Holland S&B): a
construct is "inside the ontology" iff the conjunctive contract passes. The agent
(mint) reasons; this gate judges — every signal here is computed from real
verifiers, never simulated.

signals = {deeponto_ok, deeponto_complex, polyglot_ok, novelty_ok, schema_ok,
           r1_on, r1_ci_low, n_cols}

Reuses (no reinvention): deeponto_harness.{ensure_jvm,probe_template},
ddl.{template_to_table,render_ddl,validate_ddl}, coverage_r1.{construct_terms,
topic_term_signatures,r1,boot_ci}, generate_ontology.{gate_schema_realism,
gate_novelty,candidate_embedding,_embed,template_embedding_text}.
"""
from __future__ import annotations

import glob
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402
from aegir.ontology.deeponto_harness import ensure_jvm, probe_template  # noqa: E402
from aegir.ontology.ddl import template_to_table, render_ddl, validate_ddl  # noqa: E402
import coverage_r1 as cr  # noqa: E402
import generate_ontology as go  # noqa: E402
from mediate_consistency import coherence  # noqa: E402  (HermiT ground-truth coherence)

log = logging.getLogger("contract_gate")
CATALOG_GLOB = "src/aegir/ontology/catalog/0[1-7]_*.json"


class ContractGate:
    """gate(construct, topic) -> signal dict. Build-once state: JVM, topic
    term-signatures (Vt, indexed by topic_id), seed-embedding novelty pool."""

    def __init__(self, coverage_run: str, tau_nov: float = 0.93,
                 n_shuffled: int = 40, seed: int = 20260616):
        self.tau_nov = tau_nov
        self.n_shuffled = n_shuffled
        self.rng = np.random.default_rng(seed)
        ensure_jvm()  # idempotent; needs LD_LIBRARY_PATH=build/jvm-libs

        import pyarrow.parquet as pq
        rows = pq.read_table(Path(coverage_run) / "topic_coverage.parquet").to_pylist()
        by_topic = {r["topic_id"]: r for r in rows}
        n_topics = len(rows)
        reprs = [(by_topic.get(i, {}).get("topic_repr_text") or "") for i in range(n_topics)]
        self.Vt = cr.topic_term_signatures(reprs)            # list[set], index == topic_id
        log.info("Vt: %d topics, mean |V_t|=%.1f", len(self.Vt),
                 float(np.mean([len(s) for s in self.Vt])))

        # Seed novelty pool (embed every seed template the audit way; once).
        seed_by_fam: list[tuple[str, CatalogTemplate]] = []
        for f in sorted(glob.glob(str(REPO / CATALOG_GLOB))):
            if "candidate" in f or "combined" in f:
                continue
            fam = Path(f).stem
            for t in load_catalog(f).templates:
                seed_by_fam.append((fam, t))
        seed_texts = [go.template_embedding_text({**asdict(t), "_family": fam})
                      for fam, t in seed_by_fam]
        self.seed_pool = go._embed(seed_texts)
        log.info("seed novelty pool: %d templates embedded", len(seed_texts))

    def _polyglot_ok(self, st) -> bool:
        if st is None:
            return False
        try:
            ddl = render_ddl(st, [])
            return all(dv.valid for dv in validate_ddl(ddl))
        except RuntimeError as e:           # polyglot_sql unavailable — honest fail-closed (it IS built here)
            log.warning("polyglot unavailable, cannot validate DDL: %s", e)
            return False
        except Exception as e:              # render/validate failure = invalid DDL
            log.debug("DDL render/validate failed: %s", e)
            return False

    def _r1(self, t: CatalogTemplate, topic_id: int) -> tuple[float, float]:
        """Per-construct register-fair coverage: on-topic R1, and the bootstrap-CI
        lower bound of (on − shuffled) — the specificity that drives the loop."""
        vc = cr.construct_terms(t)
        if not vc or topic_id >= len(self.Vt):
            return 0.0, -1.0
        on = cr.r1(vc, self.Vt[topic_id], use_embed=False)
        others = [j for j in range(len(self.Vt)) if j != topic_id and self.Vt[j]]
        if not others:
            return float(on), -1.0
        k = min(self.n_shuffled, len(others))
        sample = self.rng.choice(others, size=k, replace=False)
        deltas = on - np.array([cr.r1(vc, self.Vt[j], use_embed=False) for j in sample])
        _, lo, _ = cr.boot_ci(deltas, self.rng)
        return float(on), float(lo)

    def gate(self, construct: dict, topic: dict) -> dict:
        t: CatalogTemplate = construct["template"]
        fam = topic.get("top_family") or "07_long_tail"
        tid = int(topic["topic_id"])
        sig: dict = {}

        # DeepOnto FIRST — also populates verbal_template, which construct_terms() reads.
        r = probe_template(t)
        t.verbal_template = r.verbal_template
        t.is_complex = r.is_complex
        t.mean_verbal_length = r.mean_verbal_length
        sig["deeponto_ok"] = bool(r.verbalized)
        sig["deeponto_complex"] = bool(r.is_complex)
        sig["consistent"] = bool(coherence(t)["consistent"])   # HermiT ground-truth coherence (inc-2a)

        try:
            st = template_to_table(t, fam)
            sig["n_cols"] = len(st.table.columns)
        except Exception:
            st = None
            sig["n_cols"] = 0
        sig["schema_ok"] = bool(go.gate_schema_realism(t, fam)[0])   # n_cols>=3 + DDL lowers
        sig["polyglot_ok"] = self._polyglot_ok(st)

        sig["r1_on"], sig["r1_ci_low"] = self._r1(t, tid)

        try:
            cand_emb = go.candidate_embedding(t, fam)
            sig["novelty_ok"] = bool(go.gate_novelty(cand_emb, self.seed_pool, self.tau_nov)[0])
        except Exception as e:
            log.debug("novelty failed: %s", e)
            sig["novelty_ok"] = False
        return sig


if __name__ == "__main__":
    # Unit check: gate a known SEED construct + the e6 generated candidate; the
    # signals must match the standalone verifiers (deeponto/polyglot/r1/novelty).
    import json
    logging.basicConfig(level=logging.INFO)
    run = "/raid/checkpoints/aegir-artifacts/coverage_v1/043d7dcc185245c8"
    gate = ContractGate(run)
    cat = {t.template_id: t for t in load_catalog("src/aegir/ontology/catalog/combined.json").templates}
    # a complex seed (should verbalize + be complex + lower) on an arbitrary topic
    seed = cat["artifact_with_existential"]
    sig = gate.gate({"template": seed}, {"topic_id": 124, "top_family": "01_foundation"})
    print("SEED artifact_with_existential @ t124:", json.dumps(sig))
    gen = json.load(open("/raid/checkpoints/aegir-artifacts/evidence/e6/08_generated_fixed.candidate.json"))
    gt = CatalogTemplate(**gen["templates"][0])
    tid = int(gt.provenance.get("generated_from_topic", 124))
    sig2 = gate.gate({"template": gt}, {"topic_id": tid, "top_family": gt.provenance.get("family", "07_long_tail")})
    print(f"GENERATED {gt.template_id} @ t{tid}:", json.dumps(sig2))

#!/usr/bin/env python
"""Extract the Atelier-facing release from a DDL-spine run (v2 — natural-register, spine-consuming).

Atelier classifies columns BLIND into the SKOS vocabulary; we hold the column→code reference as the
scoring key. v1 parsed chapter JSON blocks (no longer emitted) and shipped obfuscated names; v2 reads
the spine footprint directly (``base_rows.parquet`` + ``base_table_index.parquet`` + ``naming_map.parquet``
from ``build_ddl_spine --realize --naming natural``) and ships the NATURAL-register surface — the DBA
names are the *deliberately weak* name evidence (Atelier's ``name_match`` channel matches names against
the very vocabulary we emit, so a semantic name would BE the answer key; natural names make the
benchmark measure value-driven comprehension). Emits:

  - corpus_columns.parquet : RELEASE — opaque table_id/column_id, column_name + register +
                             name_provenance stamps, sample values, FK topology by opaque id.
                             ALWAYS-NAMED surface (deployment realism: real warehouses present names
                             everywhere — SchemaPile-scale data + a wiki is the downstream reality;
                             ``col3`` is a regime no user will hand Atelier). The name-provenance
                             LADDER grades trust instead of hiding names: engine-derived > composed >
                             static-legacy > degraded-mechanical (deterministic cryptic-DBA
                             abbreviation of a name with no natural record — damps exact vocabulary-
                             label matching while keeping the evidence channel live). Only the
                             verbatim answer-key channel is structurally excluded: ontology-native
                             names never ship un-degraded. Leakage is MEASURED (provenance-sliced
                             ablation), not eliminated. ``--values-only`` masks to ``col<pos>`` as an
                             explicit diagnostic arm — never the default surface.
  - reference.parquet      : HELD-BACK key — column → SKOS code, plus the semantic register
                             (elucidation channel), template identity, and slot_ref lineage.
  - release_stats.json     : scale stats + the GENERATION MANIFEST {ontology_sha, vocab_generation,
                             ddl_run_id, corpus_run_id, naming} — the version-skew guard Atelier
                             fail-fasts on.

Atelier loader contract honored (their ``aegir_release.load_aegir_release_samples``): ``column_name``
is always a non-null snake_case string, unique within its table; ``sample_values`` a list[str];
``n_rows`` int. Deterministic. No LLM / GPU / network.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# the name-provenance ladder: membrane-grade > pre-membrane natural > mechanical degradation.
# NAMED = a table carries natural naming at all (its unchanged columns are register-invariant
# structural tokens); MEMBRANE_GRADE = what a leakage-minimal analysis slice filters to.
MEMBRANE_GRADE = {"engine-derived", "composed"}
NAMED_PROVENANCE = MEMBRANE_GRADE | {"static-legacy"}


def load_code_map() -> "tuple[dict, int]":
    import pyarrow.parquet as pq
    recs = pq.read_table(REPO / "corpora/vocabulary/annotations.parquet").to_pylist()
    return {r["abbrev"].lower(): r["code"] for r in recs if r["parent_code"]}, len(recs)


def build_reference_resolver(vocab_recs: list, code_map: dict, naming: dict, index: dict):
    """R1 (spec 225241, ACCEPTED 2026-07-03): per-column reference derivation. The RESOLUTION CHAIN,
    each rung stamped in ``ref_basis`` so coarseness stays auditable (coarse-by-design vs
    coarse-by-lineage-gap must never blur — the gate would soften silently):

      1. direct class→code (measured: 154/468 distinct entity classes) ............ ``leaf``
      2. realized-ontology ancestor walk (told Sub pairs from the certified omn via the kvasir
         lowering) to the first coded ancestor .......................... ``hypernym:lineage-gap``
      3. the template head's code (433/433 — the reliable anchor) ....... ``hypernym:lineage-gap``

    data-attribute columns key at hypernyms by construction (``hypernym:underdetermined`` — the
    rout_stop_address doctrine applied to KEY DERIVATION, not just scoring). fk columns resolve
    THROUGH their target table's entity class. Returns ``resolve(tname, cname, nm, fk_tgt) ->
    (code, basis)`` plus ``head_code(tname)`` for join-key/view inheritance."""
    import re as _re

    def norm(s: str) -> str:
        return _re.sub(r"[^a-z0-9]", "", (s or "").lower())

    lut: dict = {}
    for r in vocab_recs:
        if r.get("parent_code"):
            for k in (r.get("label"), r.get("abbrev"), r.get("notation")):
                if k:
                    lut.setdefault(norm(str(k)), r["code"])

    # told ancestors from the CERTIFIED artifact (the same lowering kvasir consumes)
    ancestors: dict = {}
    try:
        from aegir.ontology.kvasir_bridge import lower_manchester
        omn = (REPO / "corpora/ontology/sdg-ontology.omn").read_text()
        for line in lower_manchester(omn)[0].splitlines():
            toks = line.split()
            local = lambda x: x.strip("<>").rsplit("#", 1)[-1]  # noqa: E731
            if toks and toks[0] in ("SubClassOf", "EquivalentToIntersection") and len(toks) >= 3:
                ancestors.setdefault(local(toks[1]), []).extend(local(x) for x in toks[2:])
    except Exception as e:  # noqa: BLE001 — resolver degrades to rungs 1+3, loudly
        print(f"  (ancestor walk unavailable — {e}; resolving with rungs 1+3 only)", file=sys.stderr)

    def coded_ancestor(cls: str) -> "str | None":
        seen, frontier = set(), [cls]
        while frontier:
            nxt = []
            for c in frontier:
                for a in ancestors.get(c, []):
                    if a in seen:
                        continue
                    seen.add(a)
                    hit = lut.get(norm(a))
                    if hit:
                        return hit
                    nxt.append(a)
            frontier = nxt
        return None

    # BFO-anchor → vocabulary bridge: the GENERIC (class) and DOM (semantic-type) subtrees never
    # meet in the current taxonomy (GENERIC→SDG.GENERIC, DOM→SDG.ICE — measured 2026-07-03), so
    # cross-frame hierarchical credit is structurally impossible on single-coded refs. Until the P5
    # vocabulary regen connects the tree, entity references ship SET-VALUED ('leaf|bridge'): the
    # ontology-class leaf AND the deployment-frame hypernym its BFO grounding names. Both frames
    # earn credit; the scorer already takes max over '|'-sets.
    _BFO_BRIDGE = {
        "bfo:0000023": "SDG.DOM.AGENT_ROLE",      # role
        "bfo:0000015": "SDG.PROCESS",             # process
        "bfo:0000040": "SDG.ARTIFACT",            # material entity
        "bfo:0000030": "SDG.ARTIFACT",            # object
        "bfo:0000031": "SDG.ICE",                 # generically dependent continuant (ICE-ish)
    }

    def bridge_for(cls: str) -> "str | None":
        seen, frontier = set(), [cls]
        while frontier:
            nxt = []
            for c in frontier:
                for a in ancestors.get(c, []):
                    if a in seen:
                        continue
                    seen.add(a)
                    if a in _BFO_BRIDGE:
                        return _BFO_BRIDGE[a]
                    nxt.append(a)
            frontier = nxt
        return None

    def class_code(cls: str) -> "tuple[str, str]":
        direct = lut.get(norm(cls))
        bridge = bridge_for(cls)
        if direct and bridge and bridge != direct:
            return f"{direct}|{bridge}", "set"
        if direct:
            return direct, "leaf"
        via = coded_ancestor(cls)
        if via and bridge and bridge != via:
            return f"{via}|{bridge}", "set"
        if via or bridge:
            return (via or bridge or ""), "hypernym:lineage-gap"
        return "", "hypernym:lineage-gap"

    # per-table entity class (for fk resolution): the entity table's class-typed slot_ref, else head
    tbl_entity_cls: dict = {}
    for (tname, _c), nmrow in naming.items():
        sr = nmrow.get("slot_ref", "")
        if (nmrow.get("kind") == "entity" and sr and sr != "__pk__" and sr != "subject"
                and not sr.startswith(("data:", "fk:"))):
            tbl_entity_cls.setdefault(tname, sr)

    def head_code(tname: str) -> "tuple[str, str]":
        tid = (index.get(tname) or {}).get("template_id", "")
        return code_map.get(tid.lower(), ""), "leaf"  # the head class IS the table's own kind

    # the measured data-attributes: explicit deployment-frame mapping for the dominant realizer
    # attrs, lut where it matches, the descriptive-ICE hypernym elsewhere
    _DESC = "SDG.ICE.DESCRIPTIVE"
    _ATTR_MAP = {"role": "SDG.DOM.AGENT_ROLE", "event_count": "SDG.DOM.MEASUREMENT"}

    def data_code(attr: str) -> "tuple[str, str]":
        hit = _ATTR_MAP.get(attr) or lut.get(norm(attr))
        return (hit or _DESC), "hypernym:underdetermined"

    def resolve(tname: str, cname: str, nm: dict, fk_tgt: "str | None") -> "tuple[str, str]":
        sr = nm.get("slot_ref", "") or ""
        if sr.startswith("data:"):
            return data_code(sr[5:])
        if sr == "subject" or sr == "fk:subject":
            # the subject / its junction FK is an instance of the table's own class
            base = fk_tgt if (sr == "fk:subject" and fk_tgt) else tname
            return head_code(base)
        if sr.startswith("fk:"):
            # resolve THROUGH the target table's entity class; dim/star targets — and classes whose
            # ancestor chains exit into un-coded BFO/CCO — fall to the target's HEAD anchor (rung 3)
            if fk_tgt and tbl_entity_cls.get(fk_tgt):
                code, basis = class_code(tbl_entity_cls[fk_tgt])
                if code:
                    return code, basis
            if fk_tgt:
                c, _ = head_code(fk_tgt)
                return c, "hypernym:lineage-gap"
            c, _ = head_code(tname)
            return c, "hypernym:lineage-gap"
        if sr and sr != "__pk__":
            code, basis = class_code(sr)
            if code:
                return code, basis
            c, _ = head_code(tname)   # rung 3: the reliable anchor, stamped as the gap it is
            return c, "hypernym:lineage-gap"
        return head_code(tname)

    return resolve, head_code


def generation_manifest(spine_dir: Path, spine_manifest: dict, n_vocab: int) -> dict:
    """The version-skew guard (task #136.2): one place recording which generation of each Data Product
    this release was cut against, so a consumer can fail fast on cross-generation mixes."""
    try:
        ontology_sha = subprocess.run(["git", "-C", str(REPO / "corpora"), "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        ontology_sha = None
    return {
        "ontology_sha": ontology_sha,
        "vocab_generation": n_vocab,
        "ddl_run_id": spine_manifest.get("run_id") or spine_dir.name,
        "corpus_run_id": None,  # set when a release is cut against a generated text corpus
        "naming": spine_manifest.get("naming"),
        # spine-wide (naming_map) distribution — release-visible columns differ (see
        # column_name_provenance in release_stats); renamed for clarity at Atelier's request
        "spine_name_provenance_distribution": spine_manifest.get("name_provenance_distribution"),
        # corpora-level train/eval split (Atelier §5.1, the RWKV-ensemble eval-design ask): releases
        # meant for post-model-integration efficacy gates must be cut from held-out chapters/topics
        # never present in the training mix. P4 defines the partition; previews are unsplit.
        "holdout_partition": "preview-unsplit",
    }


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-spine", required=True,
                    help="a build_ddl_spine run dir (base_rows + base_table_index + naming_map parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-values", type=int, default=50,
                    help="sample values kept per column (Atelier reads 5 for embedding, ≤50 all_values)")
    ap.add_argument("--values-only", action="store_true",
                    help="DIAGNOSTIC ARM: mask every name to col<pos> (Atelier's generic-name guard "
                         "→ values-only classification). Never the default — deployment reality is "
                         "always-named")
    ap.add_argument("--no-degrade", action="store_true",
                    help="ship un-degraded semantic names for columns lacking a natural record "
                         "(diagnostic only — leaks the vocabulary-label channel)")
    args = ap.parse_args()
    spine_dir = Path(args.from_spine)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spine_manifest = json.loads((spine_dir / "manifest.json").read_text())
    code_map, n_vocab = load_code_map()
    base_rows = pq.read_table(spine_dir / "base_rows.parquet").to_pylist()
    index = {r["table_name"]: r for r in pq.read_table(spine_dir / "base_table_index.parquet").to_pylist()}
    nm_path = spine_dir / "naming_map.parquet"
    naming = {}
    if nm_path.exists():
        naming = {(r["table_name"], r["col_name"]): r
                  for r in pq.read_table(nm_path).to_pylist()}

    # group cells: (table, col) → ordered distinct values; (table) → n_rows, fk targets, col order
    cells: dict[tuple, list[str]] = {}
    pk_cells: dict[str, list[str]] = {}   # table → pk values (join_key view columns project these)
    col_pos: dict[str, list[str]] = {}
    fk_target: dict[tuple, str] = {}
    pk_cols: set[tuple] = set()
    for r in base_rows:
        key = (r["table_name"], r["col_name"])
        if r["is_pk"]:
            pk_cols.add(key)
            if r["value"] not in (None, ""):
                pk_cells.setdefault(r["table_name"], []).append(str(r["value"]))
            continue
        if r["col_name"] not in col_pos.setdefault(r["table_name"], []):
            col_pos[r["table_name"]].append(r["col_name"])
        if r["value"] is not None and r["value"] != "":
            cells.setdefault(key, []).append(str(r["value"]))
        if r["is_fk"] and r.get("fk_target_table"):
            fk_target[key] = r["fk_target_table"]

    tbl_ids = {t: f"tbl_{i:06d}" for i, t in enumerate(sorted(col_pos))}  # opaque, name-free
    col_rows: list[dict] = []
    ref_rows: list[dict] = []
    prov_counts: Counter = Counter()
    basis_counts: Counter = Counter()
    code_hits = code_misses = 0
    covered: set[str] = set()
    resolve_ref, head_code = build_reference_resolver(
        pq.read_table(REPO / "corpora/vocabulary/annotations.parquet").to_pylist(),
        code_map, naming, index)
    base_ref: dict = {}   # (table_name, base_col) → (code, basis) — view-lineage inheritance (R4)

    for tname, cols in sorted(col_pos.items()):
        idx = index.get(tname) or {}
        tid = idx.get("template_id", "")
        covered.add(tid)
        code = code_map.get(tid.lower(), "")
        if code:
            code_hits += 1
        else:
            code_misses += 1
        table_id = tbl_ids[tname]
        # table-level register/provenance from any column's naming row (uniform per table)
        t_nm = next((naming.get((tname, c)) for c in cols if naming.get((tname, c))), None)
        used_names: set[str] = set()
        for pos, cname in enumerate(cols):
            vals = cells.get((tname, cname)) or []
            if not vals:
                continue
            nm = naming.get((tname, cname)) or {}
            prov = nm.get("name_provenance", "unmapped")
            register = nm.get("register", "semantic")
            public_name = cname
            if args.values_only:
                public_name = f"col{pos}"
                prov = "values-only"
            elif prov in ("semantic-passthrough", "unmapped", "semantic") and not args.no_degrade:
                # structural DBA tokens (attr_name, entity_id, value …) are register-invariant: they
                # appear as passthrough when the table's OTHER columns carry natural naming. A concept-
                # bearing name with NO natural record ships DEGRADED (cryptic-DBA abbreviation) — the
                # deployment-realistic form: named, evidence-live, vocabulary-label matching damped.
                row_provs = {(naming.get((tname, c)) or {}).get("name_provenance") for c in cols}
                if not (row_provs & NAMED_PROVENANCE):
                    from aegir.ontology.natural_naming import degrade_name
                    public_name = degrade_name(cname)
                    prov = "degraded-mechanical"
            if public_name in used_names:   # degradation can collide distinct names; the join is
                public_name = f"{public_name}_{pos}"  # name-keyed within table — dedupe, never fail
            used_names.add(public_name)
            prov_counts[prov] += 1
            distinct = list(dict.fromkeys(vals))
            col_id = f"{table_id}_c{pos}"
            ref_code, basis = resolve_ref(tname, cname, nm, fk_target.get((tname, cname)))
            basis_counts[basis if ref_code else "unresolved"] += 1
            base_ref[(tname, cname)] = (ref_code, basis)
            col_rows.append({  # PUBLIC release: no table_name / template / semantic register
                "table_id": table_id, "column_id": col_id, "column_name": public_name,
                "register": register, "name_provenance": prov,
                "construct": "base", "derivation": "",
                "n_rows": len(vals), "sample_values": distinct[:args.max_values],
                "fk_to_table_id": tbl_ids.get(fk_target.get((tname, cname), ""), None),
            })
            ref_rows.append({  # HELD-BACK key: PER-COLUMN answer (R1) + auditable basis + CTA aux
                "table_id": table_id, "column_id": col_id, "reference_code": ref_code,
                "ref_basis": basis, "table_class_code": code,
                "template_id": tid, "source_table": tname, "column_name": public_name,
                "semantic_table": nm.get("semantic_table", ""), "semantic_col": nm.get("semantic_col", ""),
                "slot_ref": nm.get("slot_ref", ""), "kind": nm.get("kind", idx.get("family", "")),
                "construct": "base", "derivation": "",
            })
        base_ref[(tname, "id")] = head_code(tname)   # pk identity — join_key views inherit it
        _ = t_nm  # (table-level row retained for future table_label emission)

    # ── R4: views as a first-class scored stratum (spec 225241) ───────────────────
    # view columns join the blind surface (construct/derivation describe shape, not answers);
    # references derive THROUGH columns_json lineage (deterministic in the realize layer);
    # verbalizations ship as the DOCS CHANNEL only (they are the model's training-pair text —
    # never part of the names+values blind surface).
    views_path = spine_dir / "views.parquet"
    n_view_cols = 0
    docs_rows: list[dict] = []
    if views_path.exists() and not args.values_only:
        views = pq.read_table(views_path).to_pylist()
        for v in sorted(views, key=lambda x: x["view_name"]):
            vcols = json.loads(v["columns_json"])
            vrows = json.loads(v.get("rows_json") or "[]")
            if not vcols:
                continue
            table_id = tbl_ids.setdefault(v["view_name"], f"tbl_{len(tbl_ids):06d}")
            docs_rows.append({"table_id": table_id, "description": v.get("verbalization") or ""})
            bases = json.loads(v.get("base_tables_json") or "[]")
            for pos, c in enumerate(vcols):
                vc, bt, bc = c["view_col"], c["base_table"], c["base_col"]
                vals = [str(row[pos]) for row in vrows
                        if isinstance(row, list) and len(row) > pos and row[pos] not in (None, "")]
                if not vals:
                    # realizer views ship rows_json=[] (only flat-mode views materialize) — PROJECT
                    # the column's values through its lineage instead: a view column's value
                    # population IS its base column's (identity/rename) or the base pk (join_key);
                    # per-column sampling needs no joined-row alignment
                    vals = pk_cells.get(bt, []) if bc == "id" else \
                        list(cells.get((bt, bc)) or [])
                if not vals:
                    continue
                if bc == "id":
                    derivation = "join_key"
                elif vc == bc:
                    derivation = "identity"
                else:
                    derivation = "rename"
                ref_code, basis = base_ref.get((bt, bc)) or head_code(bt)
                basis_counts[basis if ref_code else "unresolved"] += 1
                col_id = f"{table_id}_c{pos}"
                distinct = list(dict.fromkeys(vals))
                col_rows.append({
                    "table_id": table_id, "column_id": col_id, "column_name": vc,
                    "register": "natural", "name_provenance": "composed",
                    "construct": "view", "derivation": derivation,
                    "n_rows": len(vals), "sample_values": distinct[:args.max_values],
                    "fk_to_table_id": tbl_ids.get(bases[0]) if bases else None,
                })
                ref_rows.append({
                    "table_id": table_id, "column_id": col_id, "reference_code": ref_code,
                    "ref_basis": basis, "table_class_code": "",
                    "template_id": "", "source_table": v["view_name"], "column_name": vc,
                    "semantic_table": bt, "semantic_col": bc,
                    "slot_ref": "", "kind": "view", "construct": "view", "derivation": derivation,
                })
                n_view_cols += 1

    pq.write_table(pa.Table.from_pylist(col_rows), out / "corpus_columns.parquet")
    # KEY SEPARATION (Atelier ask, 2026-07-03): the reference lives in a SIBLING dir, never beside
    # the blind surface — with filesystem-capable agent-mediated classification on their side, the
    # blind-integrity audit becomes structural (point the agent at <release>/ only) instead of
    # procedural. P5 sealed runs ship no key at all.
    key_dir = out.parent / (out.name + ".key")
    key_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ref_rows), key_dir / "reference.parquet")
    if docs_rows:
        docs_dir = out / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(docs_rows), docs_dir / "view_descriptions.parquet")
    stats = {
        "release_version": "v3-per-column-frame",
        "n_tables": len(col_pos),
        "n_view_tables": len(docs_rows),
        "n_view_columns": n_view_cols,
        "ref_basis": dict(basis_counts),
        "n_columns": len(col_rows),
        "n_cells": sum(r["n_rows"] for r in col_rows),
        "template_coverage": len(covered),
        "reference_code_hits": code_hits,
        "reference_code_misses": code_misses,
        "column_name_provenance": dict(prov_counts),
        "generation_manifest": generation_manifest(spine_dir, spine_manifest, n_vocab),
    }
    (out / "release_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Atelier release → {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  corpus_columns.parquet (release) + release_stats.json → {out}")
    print(f"  reference.parquet (HELD BACK) → {key_dir}  [never beside the blind surface]")
    # loud contract checks (never silent): uniqueness within table + non-null snake_case names
    by_tbl: dict[str, set] = {}
    for r in col_rows:
        s = by_tbl.setdefault(r["table_id"], set())
        if r["column_name"] in s:
            print(f"  !! DUPLICATE column_name '{r['column_name']}' within {r['table_id']} "
                  "(breaks Atelier's name-keyed prediction join)")
            return 1
        s.add(r["column_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

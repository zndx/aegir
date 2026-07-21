"""sdg-strategy manifest — the run's externalized determinants as a content-addressed tree (#148).

A STRATEGY is the complete answer to "why did the pipeline do that?": everything outcome-shaping
that is neither the input window nor the code commit, in four pillars —

- **lens**    — the qdrant aperture (collection snapshot: ids, labels, vector hashes)
- **voices**  — every agent-facing surface (prompts, schemas, feedback templates, tool
                docstrings, the vendored vibe profile, backend config)
- **knobs**   — the flow's tunable defaults (HOCON)
- **targets** — what it aims at (mined norms by content hash + source identity, gate floors)

Dual-key identity, Merkle-shaped: ``strategy_id = sha256(sorted component hashes)[:12]`` is the
IDENTITY (verifiable by rehash from any signals project); release tags / zettel ids are NAMES.
The tree lives in the ``strategy/`` submodule (git@github.com:zndx/sdg-strategy.git, branch
``trunk``): shadows are BRANCHES (promotion = cherry-pick/merge; only main lineage takes release
tags); the submodule pointer tracks main; shadow manifests resolve BY SHA from the object store
(`load_by_ref`) — pinned by construction, never the working tree.

RH rulings encoded: verbatim pass-through (components are the artifacts themselves); truth flows
repo → runtime (the live collection is materialized FROM the snapshot, never exported back except
by an explicit re-seed).
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SUB = REPO / "strategy"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()


# ── collectors: the LIVE state of each pillar, verbatim ──────────────────────────────

def _scroll_snapshot(cl, collection: str) -> dict:
    import numpy as _np
    rows, off = [], None
    while True:
        pts, off = cl.scroll(collection, limit=512, offset=off,
                             with_payload=True, with_vectors=True)
        for p in pts:
            vec = p.vector if not isinstance(p.vector, dict) else next(iter(p.vector.values()))
            vs = _sha(_np.asarray(vec, dtype="float32").tobytes()) if vec is not None else ""
            pl = p.payload or {}
            row = {"id": str(p.id),
                   "iri": pl.get("iri") or "",
                   "label": pl.get("pref_label") or pl.get("label") or "",
                   "path": pl.get("path") or "",
                   "vector_sha": vs[:16]}
            if pl.get("constituents"):        # Canonical Aperture anchors carry their lattice (#31)
                row["constituents"] = [{"iri": x.get("iri"), "label": x.get("label"),
                                        "code": x.get("code"), "rel": x.get("rel"),
                                        "kind": x.get("kind")}
                                       for x in pl["constituents"]]
            rows.append(row)
        if off is None:
            break
    rows.sort(key=lambda r: r["id"])
    return {"collection": collection, "n": len(rows), "points": rows}


def collect_lens() -> "dict[str, bytes]":
    """The lens is TWO collections: the AIMING collection (harvest selection over the raw
    stream — the idempotence-critical filter) and the full-vocab collection (classification
    + congruence scoring). Both snapshotted; lens/binding.json DECLARES the runtime targets
    (truth flows repo → runtime — the code follows the strategy, not vice versa)."""
    from aegir.ontology.domain_index import (DEFAULT_APERTURE, DEFAULT_COLLECTION,
                                             DEFAULT_QDRANT_URL, _client)
    try:
        cl = _client()
        out: "dict[str, bytes]" = {
            "lens/vocab.snapshot.json": _canon(_scroll_snapshot(cl, DEFAULT_COLLECTION)),
        }
        from aegir.ontology.domain_index import DEFAULT_OVERLAY, DEFAULT_VOCAB
        for src, dst in ((DEFAULT_VOCAB, "lens/vocab.skos.ttl"),
                         (DEFAULT_OVERLAY, "lens/aperture.skos.ttl")):
            sp = Path(src)
            if sp.exists():
                out[dst] = sp.read_bytes()
        out.update({
            "lens/binding.json": _canon({"vocab_collection": DEFAULT_COLLECTION,
                                         "aiming_collection": DEFAULT_APERTURE,
                                         "qdrant_url": DEFAULT_QDRANT_URL,
                                         "materialized_from": "live"}),
        })
        try:
            out["lens/aperture.snapshot.json"] = _canon(_scroll_snapshot(cl, DEFAULT_APERTURE))
        except Exception as e:  # noqa: BLE001 — aiming collection absent: captured explicitly
            out["lens/aiming.UNAVAILABLE"] = str(e)[:200].encode()
        return out
    except Exception as e:  # noqa: BLE001 — qdrant down: the pillar is explicitly UNCAPTURED
        return {"lens/aperture.UNAVAILABLE": f"qdrant unreachable: {e}".encode()}


def _tool_docs(path: Path) -> dict:
    """MCP tool names + docstrings via ast (tool descriptions ARE agent-visible prompts)."""
    out = {}
    try:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in node.decorator_list:
                    s = ast.unparse(d)
                    if "tool" in s:
                        out[node.name] = ast.get_docstring(node) or ""
    except Exception:  # noqa: BLE001
        pass
    return out


def collect_voices() -> "dict[str, bytes]":
    from aegir.generate import agent_stage, harness
    from aegir.ontology import derive_harness, derive_loop, entities
    comp: "dict[str, bytes]" = {
        "voices/derive.system.md": derive_harness._SYSTEM.encode(),
        "voices/derive.schema.json": _canon(entities.ENTITY_SCHEMA),
        "voices/derive.feedback.py": __import__("inspect").getsource(derive_loop._feedback).encode()
                                     + __import__("inspect").getsource(derive_loop._dormant_brief).encode(),
        "voices/derive.dormant_hints.json": _canon(getattr(derive_loop, "_DORMANT_HINTS", {})),
        "voices/register_voice.json": _canon(getattr(harness, "_REGISTER_VOICE", {})),
        "voices/turn_protocol.md": getattr(agent_stage, "_TURN_PROTOCOL", "").encode(),
        "voices/backend.local.toml": getattr(harness, "_LOCAL_TOML", "").encode(),
        "voices/mcp_kvasir.tools.json": _canon(_tool_docs(REPO / "scripts/mcp_kvasir.py")),
    }
    vibe = REPO / "components/oss-mistral-cli/vibe/core/prompts/aegir_writer.md"
    if vibe.exists():  # vendored: the profile is a strategy component, not a fork detail
        comp["voices/aegir_writer.md"] = vibe.read_bytes()
    return comp


def collect_knobs() -> "dict[str, bytes]":
    """Flow Parameter defaults, read programmatically (no drift-prone literals)."""
    knobs = {}
    try:
        from metaflow import Parameter
        from aegir.flows.sdg_corpora_flow import SdgCorporaFlow
        for name in dir(SdgCorporaFlow):
            attr = getattr(SdgCorporaFlow, name, None)
            if isinstance(attr, Parameter):
                knobs[attr.name] = attr.kwargs.get("default")
    except Exception as e:  # noqa: BLE001
        knobs["_error"] = str(e)[:120]
    return {"knobs/flow_defaults.json": _canon(knobs)}


def collect_targets() -> "dict[str, bytes]":
    comp: "dict[str, bytes]" = {}
    for f in ("schemapile_shape_norms.json", "schemapile_key_norms.json"):
        p = REPO / "build" / f
        if p.exists():
            comp[f"targets/{f}"] = p.read_bytes()
    try:
        from aegir.ontology import individuals
        comp["targets/brand_lexicon.json"] = _canon({
            "clear": getattr(individuals, "_BRANDS_CLEAR", None) and individuals._BRANDS_CLEAR.pattern,
            "ambiguous": sorted(getattr(individuals, "_BRANDS_AMBIG", ())),
        })
    except Exception:  # noqa: BLE001
        pass
    comp["targets/floors.json"] = _canon({
        "payload_min_chars": 6000,
        "verdict_accept": "rich",
        "congruence_top_k": 5,
        "note": "gate floors gathered at capture; unify into strategy.conf at increment 2",
    })
    # The COLLECTION UNIT definition (RH 2026-07-09) — an outcome-shaping determinant that
    # is neither input window nor code: what a "collection" of output documents IS
    # (connected DDL components) and the Barabási calibration that makes it domain-aligned.
    # Captured from the calibrated artifact; CONSUMED by the builder (truth flows repo →
    # runtime — scripts/relational_collections.py applies the pinned operating point;
    # --sweep re-derives and PROPOSES an update, never silently rebinds).
    cal = REPO / "build" / "relational_collections" / "collections.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        comp["targets/collections_unit.json"] = _canon({
            "definition": ("a collection = the output documents whose tables/views form a "
                           "connected DDL graph (FK + view-composition edges), after "
                           "infrastructure-hub removal"),
            "method": ("targeted-attack fragmentation × neighborhood-overlap pruning; "
                       "selection rule pre-declared: giant component ≤ 15%, docs touch "
                       "p50 one component, minimum within-collection topic entropy "
                       "(validated against the inverted topic layer's associations)"),
            "operating_point": c.get("operating_point"),
            "infrastructure_hubs": c.get("infrastructure_hubs"),
        })
    # The DIFFERENTIA-TYPE HARVEST SPECS (RH 2026-07-17): one spec per distinct differentia property
    # of the authored taxonomy — the what-to-harvest contract for grounding differentia-column values
    # in real GitTables columns via Atelier's federated ensemble (maxsim/NHSVM/CatBoost). Captured
    # here so the harvest is reproducible (pure function of the component @sha + GitTables snapshot +
    # harvester version) and Atelier consumes the contract BY STRATEGY REF, not over the wire.
    # Emitted deterministically by scripts/emit_differentia_specs.py from build/taxonomy/<tag>/.
    dvs = REPO / "build" / "taxonomy" / "differentia_specs.json"
    if dvs.exists():
        comp["targets/differentia_value_specs.json"] = dvs.read_bytes()
    # The RENDER REGISTER target (RH 2026-07-09, "presentation ≠ identifier"): the corpus commits
    # to prose tables that read as human documents, not raw schemas. Measured FinePDFs reference +
    # our baseline + floors; the metric (table_register.score_register) scores every chapter, the
    # transform (presentation.py) drives it. Captured from the reference artifact.
    reg = REPO / "build" / "table_register" / "reference.json"
    if reg.exists():
        comp["targets/table_register.json"] = reg.read_bytes()
    # SEMANTIC PLAUSIBILITY as a strategy aspect (RH 2026-07-09): a quality we hold the corpus to
    # even where the shape-mask metric CANNOT see it — whether a value/name is domain-plausible
    # (is "Chelsea" a city or a person?). Not fully mechanizable; declared with the proxies we DO
    # have and an honest boundary, so it stays a first-class target rather than an unstated hope.
    comp["targets/semantic_plausibility.json"] = _canon({
        "aspect": ("values, names, and headers must be plausible FOR THEIR DOMAIN — not merely "
                   "well-shaped. The render register (table_register) makes a table LOOK real; "
                   "semantic plausibility asks whether its CONTENT could be real."),
        "boundary": ("shape masks and header rules cannot judge meaning; this aspect is only "
                     "partially mechanizable. It is declared, tracked by proxy, and — where a "
                     "proxy cannot reach — held as an authored obligation, not silently dropped."),
        "proxies": [
            "entity_value_pools / individuals registry (domain-real instance values, membrane-gated)",
            "brand_lexicon (disallow real trademark leakage; ambiguous-brand guard)",
            "grounding-anchor retrieval (CCO/FHIR vocabulary as the plausibility field)",
            "naturalness_norms (placeholder/numeric/distinct-ratio character vs FinePDFs anchor)",
        ],
        "open": ("a learned or LLM-judge plausibility scorer over cell values is not yet built; "
                 "until then plausibility is proxied, and the gap is acknowledged, not hidden."),
    })
    return comp


# ── manifest: hash, write, load, drift ───────────────────────────────────────────────

def collect() -> "tuple[dict[str, bytes], dict]":
    comp: "dict[str, bytes]" = {}
    for c in (collect_lens(), collect_voices(), collect_knobs(), collect_targets()):
        comp.update(c)
    hashes = {path: _sha(data) for path, data in sorted(comp.items())}
    root = _sha(_canon(hashes))[:12]
    manifest = {"strategy_id": root, "components": hashes,
                "pillars": {p: [k for k in hashes if k.startswith(p + "/")]
                            for p in ("lens", "voices", "knobs", "targets")}}
    return comp, manifest


def write_to_submodule(comp: "dict[str, bytes]", manifest: dict) -> str:
    sid = manifest["strategy_id"]
    for rel, data in comp.items():
        p = SUB / "components" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (SUB / "manifests").mkdir(exist_ok=True)
    (SUB / "manifests" / f"{sid}.json").write_text(json.dumps(manifest, indent=1))
    (SUB / "CURRENT").write_text(sid + "\n")
    return sid


def declared() -> "dict | None":
    """The manifest in force: the ref named by AEGIR_STRATEGY_REF (shadow runs — resolved
    BY SHA from the object store) or the submodule's checked-out CURRENT (main)."""
    import os as _os
    ref = _os.environ.get("AEGIR_STRATEGY_REF")
    if ref:
        try:
            return load_by_ref(ref)
        except Exception:  # noqa: BLE001
            return None
    cur = SUB / "CURRENT"
    if not cur.exists():
        return None
    sid = cur.read_text().strip()
    p = SUB / "manifests" / f"{sid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_by_ref(ref: str) -> dict:
    """A manifest pinned BY SHA/branch from the submodule object store (shadows resolve here —
    never the working tree)."""
    sid = subprocess.run(["git", "show", f"{ref}:CURRENT"], cwd=SUB,
                         capture_output=True, text=True, check=True).stdout.strip()
    raw = subprocess.run(["git", "show", f"{ref}:manifests/{sid}.json"], cwd=SUB,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(raw)


def read_component(path: str, ref: "str | None" = None) -> bytes:
    """A component's BYTES — the working tree for main, `git show ref:...` for shadows
    (pinned by construction)."""
    if ref:
        r = subprocess.run(["git", "show", f"{ref}:components/{path}"], cwd=SUB,
                           capture_output=True, check=True)
        return r.stdout
    return (SUB / "components" / path).read_bytes()


def lens_binding(ref: "str | None" = None) -> dict:
    """The DECLARED runtime targets: {vocab_collection, aiming_collection, qdrant_url}.
    Resolved FROM the strategy (truth flows repo → runtime); AEGIR_STRATEGY_REF routes
    shadow runs; code defaults only when no strategy is declared."""
    import os as _os
    ref = ref or _os.environ.get("AEGIR_STRATEGY_REF") or None
    try:
        return json.loads(read_component("lens/binding.json", ref))
    except Exception:  # noqa: BLE001
        from aegir.ontology.domain_index import DEFAULT_APERTURE, DEFAULT_COLLECTION
        return {"vocab_collection": DEFAULT_COLLECTION, "aiming_collection": DEFAULT_APERTURE}


def submodule_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SUB,
                          capture_output=True, text=True).stdout.strip()


def drift(manifest: "dict | None" = None) -> "list[str]":
    """Live state vs the declared manifest — the idempotence-honesty check. Returns the
    changed component paths ('' = clean)."""
    manifest = manifest or declared()
    if not manifest:
        return ["<no declared manifest — run `python -m aegir.strategy.manifest seed`>"]
    comp, live = collect()
    want, have = manifest["components"], live["components"]
    out = [f"~{k}" for k in want if k in have and want[k] != have[k]]
    out += [f"-{k}" for k in want if k not in have]
    out += [f"+{k}" for k in have if k not in want]
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        comp, man = collect()
        sid = write_to_submodule(comp, man)
        print(f"seeded strategy {sid}: {len(comp)} components "
              f"({', '.join(f'{p}:{len(v)}' for p, v in man['pillars'].items())})")
    elif len(sys.argv) > 1 and sys.argv[1] == "drift":
        d = drift()
        print("CLEAN" if not d else "\n".join(d))
        sys.exit(1 if d else 0)
    else:
        print("usage: python -m aegir.strategy.manifest {seed|drift}")

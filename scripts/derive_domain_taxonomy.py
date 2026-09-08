#!/usr/bin/env python
"""Derive a DOMAIN-TAXONOMIC hypernym layer over the corpus's latent concepts (Path A domain-taxonomy
enrichment — unblocks C2.5 L1 subtree-mixing + deepens the ontology).

The SKOS ``broader`` graph is template→BFO-structural (every template attaches under one of 7 BFO/CCO
anchors); there is no domain mid-tier (``specimen ⊃ {blood, serum, plasma}``). This script derives that
mid-tier: it gathers the latent domain concepts (spine column-name concepts), embedding-clusters them into
coherent groups, and asks the local engine (Qwen3.8-27B) for **domain hypernyms** — each grounded under one BFO/CCO
anchor — that subsume ≥2 of the concepts. Output is candidate ``child subClassOf hypernym`` +
``hypernym subClassOf anchor`` edges, to be **HermiT-admitted** (next stage, reasoning_gates) before emission
into the SKOS vocab.

Sentinel-rigorous (the seed_entity_values lesson — Qwen3.8-27B leaks reasoning into content): the model emits
``TAXON <hypernym> <- <anchor>: m1 | m2 | ...`` lines; untagged lines are ignored; budget is generous so the
tagged lines land after the reasoning preamble. Cached + thinking-traces retained.

    just engine-serve   # engine up (Qwen3.8-27B)
    uv run --no-sync python scripts/derive_domain_taxonomy.py --k 18 --limit 2   # smoke (2 clusters)
    uv run --no-sync python scripts/derive_domain_taxonomy.py                    # full (cached, resumable)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

_CACHE = REPO / "build" / "domain_taxonomy_cache.json"
_TRACES = REPO / "build" / "domain_taxonomy_traces.jsonl"
_OUT = REPO / "build" / "domain_taxonomy_candidates.json"

# the 7 BFO/CCO anchors the domain hypernyms must ground under (key → human label + hint)
ANCHORS = {
    "process": "Process — unfolds in time (observation, execution, governance activity)",
    "independent_continuant": "Independent Continuant — a bearer of qualities persisting in time",
    "artifact": "Artifact — an object made to realize a function (instrument, system, dataset-as-object)",
    "information_content_entity": "Information Content Entity — carries information about something",
    "descriptive_ice": "Descriptive ICE — describes a state of affairs (measurement, record, observation)",
    "directive_ice": "Directive ICE — prescribes/directs (plan, rule, requirement, policy)",
    "designative_ice": "Designative ICE — designates/names (identifier, code, key)",
}
# drop these from the concept pool before clustering: template-pattern + obviously-generic tokens.
_STRUCT = re.compile(r"subclass|existential|min_one|with_|_via_|anchored|clause|universal|two_|"
                     r"generic_entity|cardinality|_part|^basic$|^equiv$|^related$|^applies_to$|^part_of$")
_GENERIC = {"attr", "misc", "entity", "exit", "column", "text", "note", "type", "value", "val", "about",
            "scope", "role", "category", "attribute", "attr_type", "attribute_set", "specifies", "status"}

_SYSTEM = (
    "You build a DOMAIN taxonomy for a technical relational-data corpus (data engineering, telemetry, "
    "lab/observation, governance, provenance). Given a cluster of column/concept names, propose DOMAIN "
    "HYPERNYM categories that subsume them — each hypernym a real domain noun (e.g. 'specimen' over blood|"
    "serum, 'assay' over pcr|elisa, 'telemetry_probe' over ebpf_program|kernel_hook). Each hypernym must be "
    "subClassOf ONE of these BFO/CCO anchors (use the anchor KEY):\n" +
    "\n".join(f"  - {k}: {v}" for k, v in ANCHORS.items()) +
    "\nHard rules: (1) each hypernym subsumes >=2 of the GIVEN concepts; (2) a hypernym is a domain category, "
    "NOT a structural/generic word; (3) IGNORE purely-structural/generic concepts you can't place; (4) OUTPUT "
    "CONTRACT: emit exactly one line per hypernym, beginning with the literal tag 'TAXON <hypernym> <- "
    "<anchor_key>: member1 | member2 | ...' (hypernym is snake_case; members are from the given list). Put "
    "ALL reasoning on OTHER lines — a line without the 'TAXON ' tag is ignored."
)

_TAXON = re.compile(r"^\s*TAXON\s+([A-Za-z][A-Za-z0-9_]*)\s*<-\s*([a-z_]+)\s*:\s*(.+?)\s*$")


def gather_concepts(per_family: int | None) -> "list[tuple[str, int]]":
    from build_ddl_spine import catalog_files, load_spine
    from aegir.ontology.subtree_mix import concept_of
    spine, *_ = load_spine(catalog_files(REPO / "src/aegir/ontology/catalog"), per_family, realize=True)
    cnt: collections.Counter = collections.Counter()
    for st in spine:
        for c in st.table.columns:
            if c.slot_ref == "__pk__" or getattr(c, "pk", False):
                continue
            k = concept_of(c.name)
            if _STRUCT.search(k) or k in _GENERIC or len(k) < 3:
                continue
            cnt[k] += 1
    return cnt.most_common()


def cluster(concepts: "list[str]", k: int) -> "dict[int, list[str]]":
    from sklearn.cluster import KMeans
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode([c.replace("_", " ") for c in concepts])
    labels = KMeans(n_clusters=min(k, len(concepts)), random_state=0, n_init=10).fit_predict(emb)
    groups: dict[int, list[str]] = {}
    for c, lab in zip(concepts, labels):
        groups.setdefault(int(lab), []).append(c)
    return groups


def parse_taxa(raw: str, valid_concepts: "set[str]") -> "list[dict]":
    out = []
    for line in raw.splitlines():
        m = _TAXON.match(line)
        if not m:
            continue
        hyp, anchor, members = m.group(1).strip().lower(), m.group(2).strip(), m.group(3)
        if anchor not in ANCHORS:
            continue
        mem = [x.strip() for x in members.split("|")]
        mem = [x for x in dict.fromkeys(mem) if x in valid_concepts and x != hyp]
        if len(mem) >= 2:
            out.append({"hypernym": hyp, "anchor": anchor, "members": mem})
    return out


def derive_cluster(concepts: "list[str]", *, capability: str, temperature: float) -> "tuple[list[dict], str, str]":
    from aegir.engine.client import complete_detailed
    prompt = "Concepts to organize:\n" + ", ".join(concepts)
    out = complete_detailed(prompt, capability=capability, system_prompt=_SYSTEM,
                            max_tokens=8192, temperature=temperature)
    return parse_taxa(out["text"], set(concepts)), out["text"], out.get("reasoning_content", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-family", type=int, default=None)
    ap.add_argument("--k", type=int, default=0, help="num clusters (0 = auto ≈ n/30)")
    ap.add_argument("--capability", default="instruct")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N clusters (smoke)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    concepts = [c for c, _ in gather_concepts(a.per_family)]
    k = a.k or max(8, len(concepts) // 30)
    groups = cluster(concepts, k)
    print(f"concepts: {len(concepts)} domain candidates → {len(groups)} clusters (k={k})", flush=True)

    cache = json.loads(_CACHE.read_text()) if (_CACHE.exists() and not a.no_cache) else {}
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    trace_fh = _TRACES.open("a")
    taxa: list[dict] = []
    items = sorted(groups.items())
    if a.limit:
        items = items[:a.limit]
    for cid, members in items:
        key = hashlib.md5((",".join(sorted(members))).encode()).hexdigest()[:10]
        if key in cache and not a.no_cache:
            cl_taxa = cache[key]["taxa"]
        else:
            try:
                cl_taxa, raw, reasoning = derive_cluster(members, capability=a.capability, temperature=a.temperature)
            except Exception as e:  # noqa: BLE001
                print(f"  cluster {cid} engine error: {e}", file=sys.stderr)
                continue
            cache[key] = {"cluster": cid, "n_in": len(members), "taxa": cl_taxa}
            trace_fh.write(json.dumps({"cluster": cid, "members_in": members, "taxa": cl_taxa,
                                       "raw": raw, "reasoning_content": reasoning}) + "\n")
            if not a.no_cache:
                _CACHE.write_text(json.dumps(cache, indent=1))
        for t in cl_taxa:
            print(f"  [{cid}] TAXON {t['hypernym']} <- {t['anchor']}: {' | '.join(t['members'][:6])}"
                  f"{' …' if len(t['members']) > 6 else ''}", flush=True)
        taxa.extend(cl_taxa)
    trace_fh.close()

    # merge by (hypernym, anchor): union members
    merged: dict = {}
    for t in taxa:
        mk = (t["hypernym"], t["anchor"])
        merged.setdefault(mk, set()).update(t["members"])
    final = [{"hypernym": h, "anchor": an, "members": sorted(m)} for (h, an), m in merged.items()]
    _OUT.write_text(json.dumps({"n_hypernyms": len(final), "n_edges": sum(len(t["members"]) for t in final),
                                "taxa": final}, indent=1))
    print(f"\nderived: {len(final)} domain hypernyms, {sum(len(t['members']) for t in final)} subClassOf edges "
          f"→ {_OUT.relative_to(REPO)}  (HermiT-admission next)", flush=True)
    return 0


if __name__ == "__main__":
    from aegir.utils.clean_exit import clean_exit
    clean_exit(main())

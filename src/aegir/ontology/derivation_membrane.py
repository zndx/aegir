"""Content-first derivation membrane — three POSITIVE richness gates that avoid collapse BY CONSTRUCTION.

RH construction (2026-06-20): when a strong LLM faithfully represents a rich domain it explores a wide array
of structural primitives **by necessity** — so structural diversity is an EMERGENT property of faithful
representation, not something to impose by rejecting repeats. The earlier structural-novelty gate did the
opposite and was wrong both ways: it threw away semantically-distinct primitives that shared a shape (it
collapsed the dominant ``subClassOf … some`` form across all domains to one admit) AND, if merely inverted to
"semantics-novel, structure-free", it would have admitted trivial ``X is a Y`` axioms ad nauseam. We delete
the novelty question entirely and verify **richness** three ways; diversity then falls out:

  * **G1 — parses with DeepOnto.** Instance-level structural validity (NOT the static pattern): the
    instantiated Manchester is well-formed, fully filled, clean-prefixed, and the real DeepOnto
    ``OntologySyntaxParser`` (``extract_parts``) accepts it. Catches the ``instantiate`` corruption class
    and class-as-property misuse that a pattern-only check could not see.
  * **G2 — evidences (multiple) complex asserted classes.** Each admitted primitive must be a genuinely
    complex class (≥2 constraints, or disjointness, or cardinality, or a boolean combination) — never a bare
    atomic subsumption. The batch must evidence ≥2. This forecloses the ``subClassOf … some`` monotony.
  * **G3 — permits semantically diverse procedural verbalizations.** Via our own multi-frame library
    (``verbalization.compose_frames``) over the DeepOnto parse tree. A trivial axiom verbalizes ~one way and
    fails; a complex class admits many. This forecloses triviality.

Complementary (orthogonal to collapse, the other half of "meaningful"): **clean-room** (no proprietary
terminology), **anchor validity** (the BFO/CCO leaf is real, not LLM-hallucinated), and **faithfulness**
(the primitive is actually evidenced by the source span). NO novelty/anti-repetition gate. Canonical merge
of genuinely identical concepts only. HermiT global consistency of (seed ∪ derived) is the aggregate gate
(batched, the inc-2a oracle) — run at promotion, not per-primitive.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from aegir.ontology import verbalization as V
from aegir.ontology.patterns import _FORBIDDEN  # clean-room tripwire (shared)
from aegir.ontology.schema import CatalogTemplate, load_catalog

_CATALOG_DIR = Path(__file__).resolve().parent / "catalog"
_ALLOWED_PREFIXES = {"sdg", "bfo", "cco", "fhir", "xsd", "owl", "rdf", "rdfs", "obo"}
_AXIOM_KW = re.compile(r"\b(SubClassOf|EquivalentTo|DisjointWith|DisjointUnionOf|SubPropertyOf|"
                       r"SubPropertyChain|Characteristics)\b")
_PREFIXED = re.compile(r"\b([a-z][a-z0-9]*):[A-Za-z0-9_]+")
_RESIDUAL_SLOT = re.compile(r"\{[^}]*:(ObjectProperty|DataProperty)\}")
_RESTR = re.compile(r"\b(some|only|min|max|exactly|value)\b", re.I)
_CARD = re.compile(r"\b(min|max|exactly)\s+\d+", re.I)
_BOOL = re.compile(r"\b(and|or|not)\b", re.I)
_DISJOINT = re.compile(r"\bDisjoint(With|UnionOf)\b", re.I)


# ── G1: instance structural validity ────────────────────────────────────────────

def instance_well_formed(manchester: str) -> "tuple[bool, str]":
    """Pure instance guard (fast pre-check for G1): the FILLED Manchester string is structurally sound.
    Catches the ``instantiate`` corruption class without the JVM."""
    m = manchester or ""
    if m.count("{") != m.count("}"):
        return False, "unbalanced braces"
    if not _AXIOM_KW.search(m):
        return False, "no recognized axiom keyword"
    if "::" in m:
        return False, "double-colon (corrupted IRI)"
    if "$" in m:
        return False, "residual $DESIGN param"
    if _RESIDUAL_SLOT.search(m):
        return False, "unfilled property/data slot"
    bad = sorted({p for p in _PREFIXED.findall(m) if p not in _ALLOWED_PREFIXES})
    if bad:
        return False, f"unknown IRI prefix(es): {bad}"
    return True, ""


# ── G2: complexity / complex asserted class ─────────────────────────────────────

def complexity(manchester: str) -> dict:
    """Per-primitive structural complexity of the asserted class. ``is_complex_class`` is True for a
    genuinely complex class (≥2 constraints / disjointness / cardinality / boolean combo) — never a bare
    atomic subsumption (a single ``some`` with nothing else, or a named-super only)."""
    m = manchester or ""
    n_restr = len(_RESTR.findall(m))
    n_card = len(_CARD.findall(m))
    n_bool = len(_BOOL.findall(m))
    disjoint = bool(_DISJOINT.search(m))
    # conjunct arity in the superclass RHS (commas joining clauses), minus the leading "Class:" comma noise
    n_conj = m.count(",")
    score = n_restr + n_card + n_bool + n_conj + (2 if disjoint else 0)
    is_complex = bool(disjoint or n_card >= 1 or n_restr >= 2 or (n_restr >= 1 and n_conj >= 1) or n_bool >= 1)
    return {"score": score, "n_restrictions": n_restr, "n_cardinality": n_card, "n_bool": n_bool,
            "n_conjuncts": n_conj, "disjoint": disjoint, "is_complex_class": is_complex}


# ── G3: diverse procedural verbalizations (our multi-frame library over the DeepOnto parse) ──

def verbalization_diversity(template: CatalogTemplate, *, k_min: int = 3, jvm: bool = True) -> dict:
    """G3: the primitive admits ≥k_min DISTINCT procedural verbalization skeletons. Uses
    ``verbalization.compose_frames`` over the DeepOnto parse tree (``extract_parts``, JVM). A trivial axiom
    yields ~1 skeleton and fails; a complex class yields many."""
    if not jvm:
        return {"ok": None, "n_frames": 0, "n_distinct": 0, "reason": "jvm-off"}
    try:
        parts = V.extract_parts(template)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "n_frames": 0, "n_distinct": 0, "reason": f"parse-error:{type(e).__name__}"}
    if parts is None:
        return {"ok": False, "n_frames": 0, "n_distinct": 0, "reason": "no-parse"}
    frames = V.compose_frames(parts, seed_key=template.template_id, k=6)
    # slot-normalize each frame to its structural skeleton, then count distinct
    skeletons = {re.sub(r"\{[^}]+\}", "{}", re.sub(r"\s+", " ", f.strip().lower())) for f in frames}
    n_distinct = len(skeletons)
    return {"ok": n_distinct >= k_min, "n_frames": len(frames), "n_distinct": n_distinct,
            "frames": frames[:6], "reason": "" if n_distinct >= k_min else "low-diversity"}


def deeponto_parses(template: CatalogTemplate) -> "tuple[bool, str]":
    """G1 deep check: the real DeepOnto OntologySyntaxParser accepts the instance (extract_parts → parts)."""
    try:
        parts = V.extract_parts(template)
    except Exception as e:  # noqa: BLE001
        return False, f"deeponto-error:{type(e).__name__}"
    return (parts is not None), ("" if parts is not None else "deeponto-no-parse")


# ── complementary gates ─────────────────────────────────────────────────────────

_CCO_MODULE = _CATALOG_DIR.parents[3] / "build" / "grounding" / "cco-module.ttl"  # π(CCO): the fixed theory


@lru_cache(maxsize=1)
def _theory_anchors() -> frozenset:
    """Real class curies from the fixed theory π(CCO) (``cco-module.ttl``) — ``cco:``/``bfo:`` forms, for the
    anchor fast-path. Graceful-empty if the module isn't built (anchors then defer to HermiT ancestry)."""
    if not _CCO_MODULE.exists():
        return frozenset()
    try:
        import rdflib
        g = rdflib.Graph()
        g.parse(str(_CCO_MODULE), format="turtle")
        out: set[str] = set()
        for s in set(g.subjects()):
            iri = str(s)
            if "commoncoreontologies.org/" in iri:
                out.add("cco:" + iri.rsplit("/", 1)[-1])
            elif "/obo/BFO_" in iri:
                out.add("bfo:" + iri.rsplit("BFO_", 1)[-1])
        return frozenset(out)
    except Exception:  # noqa: BLE001
        return frozenset()


@lru_cache(maxsize=1)
def known_anchors() -> frozenset:
    """Real BFO/CCO/FHIR anchor leaves — the cheap real-anchor allowlist for the anchor gate (new anchors
    fall through to HermiT BFO-ancestry at promotion). Sourced from the FIXED THEORY π(CCO) + our own
    ACCRETING derived classes (``08_derived``) — NOT the retired hand-authored seed families (01-07). The
    deriver grounds to real CCO/FHIR classes + what we have already grounded, adapting to organic inputs."""
    out: set[str] = set(_theory_anchors())
    for f in _CATALOG_DIR.glob("08_*.json"):
        if ".candidate" in f.name:
            continue
        try:
            for t in load_catalog(f).templates:
                out.update(t.bfo_anchor_path or [])
        except Exception:  # noqa: BLE001
            continue
    return frozenset(out)


def anchor_valid(bfo_anchor_path: "list[str]") -> "tuple[bool, str]":
    if not bfo_anchor_path:
        return False, "no anchor"
    leaf = bfo_anchor_path[-1]
    pref = leaf.split(":")[0] if ":" in leaf else ""
    if pref not in ("bfo", "cco", "obo", "fhir"):
        return False, f"anchor leaf {leaf!r} not a BFO/CCO/FHIR IRI"
    if leaf in known_anchors():
        return True, ""
    # unknown but well-formed BFO/CCO/FHIR IRI: allow through to the HermiT ancestry check at promotion
    return True, "unverified-anchor(defer-to-hermit)"


# generic ontology/upper-level vocabulary that should NOT count toward domain faithfulness
_GENERIC_TERMS = frozenset((
    "class", "subclassof", "equivalentto", "disjointwith", "disjointunionof", "some", "only", "exactly",
    "value", "thing", "entity", "information", "content", "descriptive", "directive", "occurrent",
    "continuant", "process", "quality", "role", "realizable", "specifically", "dependent", "generically",
    "material", "object", "aggregate", "spatial", "temporal", "region", "datatype", "decimal", "string",
    "integer", "boolean", "property", "individual", "characteristics", "subpropertyof", "domain", "range",
    "type", "kind", "instance", "concept", "definition", "data", "attribute", "measurement"))


def _domain_tokens(text: str) -> set:
    """Lowercase domain word tokens (≥4 chars), splitting camelCase + snake_case + colons so coined IRIs
    (``sdg:tissueAblation`` → {tissue, ablation}) contribute their domain vocabulary."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")  # camelCase → words
    text = re.sub(r"[:_/.\-]", " ", text)
    return {t for t in re.findall(r"[a-z]{4,}", text.lower())}


def faithful(source_span: str, template: CatalogTemplate, *, tau: float = 0.18) -> dict:
    """Faithfulness signal: does the primitive's COINED vocabulary reflect the source span? Overlap between
    the span's domain tokens and the primitive's surface (manchester coined IRIs camelCase-split + verbal +
    template_id), minus generic upper-level terms. Engine-entailment is the stronger v2 (the engine is free).
    """
    span_toks = _domain_tokens(source_span)
    if not span_toks:
        return {"ok": False, "overlap": 0.0, "reason": "no-span"}
    surf = " ".join([template.manchester_template or "", template.verbal_template or "",
                     template.template_id or ""])
    surf_toks = _domain_tokens(surf) - _GENERIC_TERMS
    if not surf_toks:
        return {"ok": False, "overlap": 0.0, "reason": "no-surface"}
    overlap = len(span_toks & surf_toks) / max(1, len(surf_toks))
    return {"ok": overlap >= tau, "overlap": round(overlap, 3), "reason": "" if overlap >= tau else "low-overlap"}


def clean_room(manchester: str) -> bool:
    return not _FORBIDDEN.search(manchester or "")


# ── the membrane ────────────────────────────────────────────────────────────────

def content_membrane(template: CatalogTemplate, *, source_span: str = "", jvm: bool = True,
                     k_verbal: int = 3) -> dict:
    """The content-first gates for ONE primitive. admit = G1 ∧ G2 ∧ G3 ∧ clean ∧ anchor ∧ faithful.
    NO novelty. utility re-based on richness (complexity × verbalization-diversity × faithfulness),
    NOT structural novelty or tier."""
    m = template.manchester_template
    g: dict = {}
    wf_ok, wf_reason = instance_well_formed(m)
    cx = complexity(m)
    av_ok, av_reason = anchor_valid(template.bfo_anchor_path or [])
    fa = faithful(source_span, template)
    vd = verbalization_diversity(template, k_min=k_verbal, jvm=jvm)
    dp_ok, dp_reason = (deeponto_parses(template) if jvm else (None, "jvm-off"))

    g["g1_well_formed"] = wf_ok
    g["g1_deeponto_parses"] = dp_ok
    g["g1_reason"] = wf_reason or dp_reason
    g["g2_is_complex_class"] = cx["is_complex_class"]
    g["g2_complexity"] = cx
    g["g3_verbal_diversity_ok"] = vd["ok"]
    g["g3_verbal"] = {k: vd[k] for k in ("n_frames", "n_distinct", "reason") if k in vd}
    g["clean_room"] = clean_room(m)
    g["anchor_valid"] = av_ok
    g["anchor_reason"] = av_reason
    g["faithful"] = fa["ok"]
    g["faithful_overlap"] = fa["overlap"]

    # STRUCTURAL leg (#141): the kvasir DDL-relevance signal — does this extension lower to
    # well-formed DDL and drive toward SchemaPile structure parity? REPORT-ONLY for admission
    # (HermiT/OntoClean own logical admission; structural is a forcing function measured across
    # the run, ratcheted into a hard gate later per the EVIDENCE discipline), but it re-weights
    # utility so relationally-rich extensions rank higher, and its reason is surfaced to the
    # agent for the closed re-propose loop (inc-2).
    struct = _structural(m)
    g["structural_verdict"] = struct.get("verdict")
    g["structural_reason"] = struct.get("reason")
    g["structural_well_formed"] = struct.get("well_formed")
    g["structural_yield"] = struct.get("ddl_yield")
    g["structural_parity"] = struct.get("parity")

    # G1 admits on the pure guard AND (DeepOnto parse if the JVM ran)
    g1 = wf_ok and (dp_ok is not False)
    g3 = vd["ok"] if vd["ok"] is not None else wf_ok  # if JVM off, G3 deferred → don't block on it
    g["admit"] = bool(g1 and cx["is_complex_class"] and g3 and g["clean_room"] and av_ok and fa["ok"])

    # richness utility (ranking only): complexity, verbalization diversity, faithfulness,
    # AND the structural verdict (rich > thin > inert) — prefer relationally-yielding primitives
    cx_norm = min(1.0, cx["score"] / 6.0)
    vd_norm = min(1.0, (vd.get("n_distinct") or 0) / 5.0)
    struct_factor = {"rich": 1.0, "thin": 0.7, "inert": 0.4,
                     "malformed": 0.1}.get(struct.get("verdict") or "", 0.7)  # unavailable → neutral
    g["utility"] = round(
        cx_norm * (0.5 + 0.5 * vd_norm) * (0.5 + 0.5 * min(1.0, fa["overlap"] / 0.3)) * struct_factor,
        3)
    return g


def _structural(manchester: str) -> dict:
    """The kvasir DDL-relevance signal for one primitive; degrades gracefully (a neutral
    ``unavailable`` verdict) if the kvasir binary is absent, so the membrane never hard-fails
    on the structural leg."""
    try:
        from aegir.ontology import ddl_membrane
        return ddl_membrane.signal_for_manchester(manchester)
    except Exception as e:  # pragma: no cover — never let the structural leg break the membrane
        return {"verdict": "unavailable", "reason": f"structural signal error: {e}",
                "well_formed": None}


# ── batch gate: "multiple complex asserted classes" + canonical merge ───────────

def _canon_key(t: dict) -> str:
    """Identity of a concept for merge: anchor leaf + normalized head label (NOT structure)."""
    anchor = (t.get("bfo_anchor_path") or [""])[-1]
    label = re.sub(r"[^a-z0-9]+", "", (t.get("template_id") or "").lower())
    return f"{anchor}|{label}"


def batch_gate(admitted: "list[dict]", *, min_complex: int = 2) -> dict:
    """G2 at the output level + canonical merge. ``admitted`` are CatalogTemplate-shaped dicts carrying a
    ``_complexity`` block (from content_membrane). Returns the merged set + whether the batch evidences
    multiple complex asserted classes."""
    by_key: dict[str, dict] = {}
    dropped = 0
    for t in admitted:
        k = _canon_key(t)
        if k in by_key:
            dropped += 1  # genuine duplicate concept — merge (keep the higher-utility one)
            if (t.get("_utility") or 0) > (by_key[k].get("_utility") or 0):
                by_key[k] = t
        else:
            by_key[k] = t
    merged = list(by_key.values())
    n_complex = sum(1 for t in merged if (t.get("_complexity") or {}).get("is_complex_class"))
    tiers = Counter(t.get("_tier", "?") for t in merged)
    groundings = Counter(t.get("_grounds_ddl", "?") for t in merged)
    return {"merged": merged, "n_in": len(admitted), "n_merged": len(merged), "n_dup_dropped": dropped,
            "n_complex_classes": n_complex, "multiple_complex_ok": n_complex >= min_complex,
            "tiers": dict(tiers), "groundings": dict(groundings)}

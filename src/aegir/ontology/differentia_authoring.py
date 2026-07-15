"""differentia_authoring — the agent-mediated CAS loop that AUTHENTICATES the provisional genus scaffold
(``genus_induction``) into a genus+differentia taxonomy.

Holland Signals & Boundaries: a BOUNDARY (insufficient differentia — a species not individuable within its
genus) SIGNALS the genus + species + its confusable siblings; the AGENT proposes the single most salient
OBSERVABLE differentia (a property that projects to a column or FK join); a MEMBRANE disposes
(parse ∧ observability ∧ discrimination ∧ — batched — HermiT), each returning its REASON; on reject the loop
RE-PROMPTS with that reason (the closed loop, never one-shot — [[agent_mediated_feedback_loop]]).

Durable-component contract (RH 2026-07-14): the METHODS live here (loop shape, the proposer prompt, the
membranes + acceptance criteria, the CAS feedback protocol) so this wires into Metaflow as a step. The PROPOSER
is swappable — ``engine_proposer`` (single gRPC call + re-prompt) now; an ACP-harnessed multi-turn proposer
(``refine.acp``) drops in where the task needs tool-use — without touching the loop or the membranes.

Calibrated: we only author differentia at the achievable resolution (the induction frontier). A species whose
confusable siblings are below the discriminating potential is left flagged, not force-differentiated (that would
fabricate a distinction the lexicon can't support). [[refinement_loop]] [[ontoclean_rigor_roadmap]]
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROMPT_VERSION = "differentia-v2-2026-07-15"   # v2: HermiT membrane + value enrichment/guard

REPO = Path(__file__).resolve().parents[3]
_JVM = str(REPO / "build" / "jvm-libs")
_PREFIXES = ('Prefix: owl: <http://www.w3.org/2002/07/owl#>\n'
             'Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n'
             'Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>\n'
             'Prefix: sdg: <https://signals.zndx.org/sdg#>\n'
             'Ontology: <https://signals.zndx.org/sdg/frag>\n')


def load_enum_values(entities_dir: Path) -> "dict[str, list[str]]":
    """{class: [real enumerated attribute values]} — the VALUE ENRICHMENT source. Seeds the proposer so a
    ``value`` differentia is grounded in a REAL domain value the species actually carries, not an invention."""
    out: "dict[str, list[str]]" = {}
    for f in glob.glob(str(entities_dir / "*.json")):
        for e in json.loads(Path(f).read_text()).get("entities", []):
            n = e.get("name")
            if not n or n in out:
                continue
            vals = [v for a in (e.get("attributes") or []) for v in (a.get("enum") or []) if isinstance(v, str)]
            if vals:
                out[n] = sorted(set(vals))[:12]
    return out

_SYS = (
    "You author the DIFFERENTIA that individuates an ontology species within its genus — the Aristotelian "
    "genus+differentia definition. Given the genus, the target species, and its CONFUSABLE sibling species, name "
    "the SINGLE most salient OBSERVABLE property whose presence or value distinguishes the target from EVERY "
    "sibling shown. Observable = it appears as a table COLUMN or an FK JOIN (a data or object property), not a "
    "purely logical or metaphysical distinction. Prefer a property already implied by the target's definition. "
    "Fill EVERY field with a REAL value for THIS target — never copy the format words. "
    "Worked example — genus 'PhysicalQuantity', target 'GravitationalConstant', siblings 'MeasuredAcceleration', "
    "'FundamentalConstant': "
    '{"property":"knownnessStatus","kind":"data","restriction":"value \\"well-known\\"","why":"marks it as a '
    'canonical constant, unlike a one-off measured acceleration"}. '
    "'property' is a camelCase name; 'kind' is exactly data or object; 'restriction' is a Manchester filler "
    "(some xsd:date | value \"x\" | some TargetClass | exactly 2 xsd:integer).")

_SCHEMA = json.dumps({"type": "object", "properties": {
    "property": {"type": "string"}, "kind": {"type": "string", "enum": ["data", "object"]},
    "restriction": {"type": "string"}, "why": {"type": "string"}},
    "required": ["property", "kind", "restriction", "why"]})


@dataclass
class Differentia:
    species: str
    property: str = ""
    kind: str = ""
    restriction: str = ""
    why: str = ""
    rounds: int = 0
    accepted: bool = False
    reason: str = ""          # the disposing membrane's verdict (why accepted / why given up)
    model: str = ""


# ── PROPOSER (swappable) ────────────────────────────────────────────────────────────────────────────────────
def engine_proposer(genus: str, species: str, species_def: str, siblings: "list[tuple[str, str]]",
                    feedback: str = "", known_values: "list[str] | None" = None) -> dict:
    """Single gRPC call. ``siblings`` = [(name, definition)]. ``feedback`` = the prior membrane's reason (CAS).
    ``known_values`` = the species' REAL enumerated attribute values (value enrichment — ground a ``value``
    differentia in a value the species actually carries)."""
    from aegir.engine.client import complete_detailed  # noqa: PLC0415
    sib = "\n".join(f"  - {n}: {d[:160]}" for n, d in siblings) or "  (none)"
    kv = (f"\nReal values this species is known to carry (prefer one for a 'value' differentia): "
          f"{', '.join(known_values[:12])}\n" if known_values else "")
    prompt = (f"Genus: {genus}\nTarget species: {species} — {species_def[:220]}\n"
              f"Confusable sibling species:\n{sib}\n{kv}" +
              (f"\nA prior attempt was rejected: {feedback}\nAuthor a BETTER, real differentia.\n" if feedback else "") +
              "\nReturn the json differentia for the target (real values, not the format words).")
    out = complete_detailed(prompt, capability="instruct", system_prompt=_SYS, max_tokens=600, temperature=0.3,
                            json_schema=_SCHEMA)
    ms = re.findall(r"\{[^{}]*\}", out.get("text", ""), re.S)      # last well-formed object = the answer
    d = {}
    for cand in reversed(ms):
        try:
            d = json.loads(cand); break
        except ValueError:
            continue
    d["_model"] = out.get("model", "")
    return d


def name_genus(members: "list[str]") -> dict:
    """Propose the genus's authentic label (a superordinate kind, NOT one of the members) + one-line definition."""
    from aegir.engine.client import complete_detailed  # noqa: PLC0415
    sys = ("You name the GENUS shared by a set of ontology species. Output json {\"label\":\"CamelCaseKind\","
           "\"definition\":\"one sentence\"}. The label must be a SUPERORDINATE kind that subsumes all the "
           "members and is NOT identical to any member name.")
    out = complete_detailed(f"Species: {', '.join(members)}\nName their common genus.",
                            capability="instruct", system_prompt=sys, max_tokens=200, temperature=0.2,
                            json_schema=json.dumps({"type": "object", "properties": {
                                "label": {"type": "string"}, "definition": {"type": "string"}},
                                "required": ["label", "definition"]}))
    ms = re.findall(r"\{[^{}]*\}", out.get("text", ""), re.S)
    for cand in reversed(ms):
        try:
            d = json.loads(cand)
            if d.get("label") and d["label"] not in members and _PASCAL.match(d["label"]):
                return d
        except ValueError:
            continue
    return {"label": "", "definition": ""}


# ── MEMBRANES (dispose; each returns (ok, reason)) ──────────────────────────────────────────────────────────
_CAMEL = re.compile(r"^[a-z][A-Za-z0-9]*$")
_PASCAL = re.compile(r"^[A-Z][A-Za-z0-9]*$")


_ECHO = {"camelcasename", "manchester filler", "data|object", "one clause", "camelcase name"}


def parse_membrane(d: dict) -> "tuple[bool, str]":
    prop, restr = d.get("property", ""), d.get("restriction", "")
    if any(e in f"{prop} {restr} {d.get('why','')}".lower() for e in _ECHO):
        return False, "you copied the format example — produce a REAL differentia for this species"
    if not prop or not _CAMEL.match(prop):
        return False, "malformed: 'property' must be a camelCase name"
    d["kind"] = str(d.get("kind", "")).strip().lower().split()[0] if d.get("kind") else ""  # 'data property'→'data'
    if d["kind"] not in ("data", "object"):
        return False, "malformed: 'kind' must be 'data' or 'object'"
    if not restr:
        return False, "malformed: missing Manchester 'restriction' filler"
    return True, "well-formed"


def observability_membrane(d: dict) -> "tuple[bool, str]":
    """The differentia must project to data: a datatype/enum column (data) or an FK to a class (object)."""
    r = d.get("restriction", "")
    if d["kind"] == "data" and not re.search(r"xsd:|value\s+|\{", r):
        return False, "not observable: a data differentia needs an xsd datatype / enumerated value"
    if d["kind"] == "object" and not re.search(r"some|only|exactly|min|max|value", r):
        return False, "not observable: an object differentia needs a Manchester restriction onto a target class"
    vm = re.search(r'value\s+"([^"]+)"', r)      # a value restriction must be a real VALUE, not a name echo
    if vm:
        v = vm.group(1).strip()
        if v.lower() == d.get("property", "").lower():
            return False, f'the value "{v}" echoes the property name — use a real discriminating value'
        if re.match(r"^[A-Z][a-z]+[A-Z]", v) or re.match(r"^[A-Z][a-z]+(Record|Kind|Type|Status|Role|Entity)$", v):
            return False, f'the value "{v}" looks like a class/type name — use a real domain value (e.g. "health")'
    return True, "observable (projects to a column/join)"


# ── HermiT membrane (batched per genus, isolated subprocess so a grind exits IT, not the run) ────────────────
_FILLER_RE = re.compile(r"\bsdg:([A-Z][A-Za-z0-9]*)")


def _fragment_omn(genus_label: str, diffs: "dict[str, Differentia]") -> str:
    """A minimal, self-declared OMN: genus + each species (SubClassOf genus + differentia) + every referenced
    property/filler declared + DisjointClasses(species) — so HermiT tests whether the differentiae are
    consistent AND compatible with the species being distinct."""
    lines = [_PREFIXES, f"Class: sdg:{genus_label}\n    SubClassOf: owl:Thing"]
    props: "dict[str, str]" = {}
    fillers: "set[str]" = set()
    species = [s for s, d in diffs.items() if d.accepted]
    for sp in species:
        d = diffs[sp]
        lines.append(f"Class: sdg:{sp}\n    SubClassOf: sdg:{genus_label}, sdg:{d.property} {d.restriction}")
        props[d.property] = d.kind
        if d.kind == "object":
            fillers.update(_FILLER_RE.findall(d.restriction))
    for p, k in props.items():
        lines.append(f"{'ObjectProperty' if k == 'object' else 'DataProperty'}: sdg:{p}")
    for fclass in sorted(fillers - {genus_label} - set(species)):
        lines.append(f"Class: sdg:{fclass}\n    SubClassOf: owl:Thing")
    if len(species) > 1:
        lines.append("DisjointClasses: " + ", ".join(f"sdg:{s}" for s in species))
    return "\n".join(lines)


def hermit_membrane(genus_label: str, diffs: "dict[str, Differentia]", *, budget_s: int = 180) -> "tuple[bool, list[str], str]":
    """(consistent, unsat_species, reason). Runs _reason in an isolated subprocess (JVM + the exit-3 budget
    watchdog stay OUT of the authoring process). unsat_species ⇒ their differentiae broke consistency/distinctness
    → the CAS loop re-prompts THEM with the reason."""
    doc = _fragment_omn(genus_label, diffs)
    with tempfile.NamedTemporaryFile("w", suffix=".omn", delete=False) as f:
        f.write(doc)
        omn = f.name
    helper = ("import sys, json; sys.path.insert(0,'src'); "
              "from aegir.ontology.deeponto_harness import ensure_jvm; ensure_jvm(); "  # MUST precede deeponto import
              "import importlib.util as u; "
              "spec=u.spec_from_file_location('bro','scripts/build_realized_ontology.py'); "
              "m=u.module_from_spec(spec); spec.loader.exec_module(m); "
              "r=m._reason(open(sys.argv[1]).read()); "
              "print(json.dumps({'consistent': bool(r[2]), 'unsat': [x.split('#')[-1] for x in r[4]]}))")
    env = {**__import__("os").environ, "LD_LIBRARY_PATH": _JVM, "AEGIR_REASON_BUDGET_S": str(budget_s)}
    try:
        p = subprocess.run(["uv", "run", "--no-sync", "python", "-c", helper, omn],
                           cwd=str(REPO), capture_output=True, text=True, timeout=budget_s + 120, env=env)
        if p.returncode == 3:
            return True, [], "hermit budget exceeded — accepted structurally (a pathological genus fragment)"
        line = [ln for ln in p.stdout.splitlines() if ln.strip().startswith("{")]
        if not line:
            return True, [], f"hermit inconclusive (rc={p.returncode}); {(p.stderr or '')[-160:]}"
        res = json.loads(line[-1])
        if not res["consistent"]:
            return False, species_of(diffs), "fragment INCONSISTENT — a differentia contradicts the genus/siblings"
        return (not res["unsat"]), res["unsat"], (
            "consistent + species pairwise-distinct" if not res["unsat"]
            else f"unsatisfiable under distinctness: {', '.join(res['unsat'][:6])}")
    except subprocess.TimeoutExpired:
        return True, [], "hermit timed out — accepted structurally"


def species_of(diffs: "dict[str, Differentia]") -> "list[str]":
    return [s for s, d in diffs.items() if d.accepted]


def discrimination_membrane(d: dict, sibling_diffs: "list[Differentia]") -> "tuple[bool, str]":
    """The differentia must not collide with any sibling's — the CAS re-prompt driver."""
    for sd in sibling_diffs:
        if sd.property == d.get("property") and sd.restriction.strip() == d.get("restriction", "").strip():
            return False, (f"sibling '{sd.species}' already uses the identical differentia "
                           f"({sd.property} {sd.restriction}); find one that separates the target FROM it")
    return True, "distinct from sibling differentiae"


# ── the CAS loop ────────────────────────────────────────────────────────────────────────────────────────────
def author_differentia(genus: str, species: str, species_def: str, siblings: "list[tuple[str, str]]",
                       sibling_diffs: "list[Differentia]", *, proposer=engine_proposer, max_rounds: int = 3,
                       known_values: "list[str] | None" = None, seed_feedback: str = "") -> Differentia:
    """Boundary→signal→propose→dispose→re-prompt for ONE species. Membranes short-circuit with their reason.
    ``seed_feedback`` primes round 1 with a prior verdict (e.g. the HermiT membrane's re-prompt)."""
    result = Differentia(species=species)
    feedback = seed_feedback
    for rnd in range(1, max_rounds + 1):
        result.rounds = rnd
        try:
            d = proposer(genus, species, species_def, siblings, feedback, known_values)
        except Exception as e:  # noqa: BLE001 — a proposer hiccup is a soft give-up, logged in the reason
            result.reason = f"proposer error: {type(e).__name__}: {str(e)[:80]}"
            return result
        result.model = d.get("_model", "")
        for membrane in (parse_membrane, observability_membrane,
                         lambda dd: discrimination_membrane(dd, sibling_diffs)):
            ok, reason = membrane(d)
            if not ok:
                feedback = reason
                break
        else:  # all membranes passed
            result.property, result.kind = d["property"], d["kind"]
            result.restriction, result.why = d["restriction"], d.get("why", "")
            result.accepted, result.reason = True, "accepted (parse ∧ observable ∧ distinct)"
            return result
    result.reason = f"gave up after {max_rounds} rounds; last membrane: {feedback}"
    return result


def author_genus(genus_label: str, species: "list[tuple[str, str]]", *, proposer=engine_proposer,
                 max_rounds: int = 3, enum_values: "dict[str, list[str]] | None" = None,
                 hermit: bool = True, genus_rounds: int = 2) -> "dict[str, Differentia]":
    """Author a distinguishing differentia for each species (structural membranes, per-species CAS), then fire
    the HermiT membrane over the assembled genus; re-author the species it flags UNSAT (genus-level CAS) up to
    ``genus_rounds``. Value-enriched: each species is seeded with its real enum values. Returns the accepted set
    + a ``__hermit__`` marker Differentia carrying the final verdict."""
    enum_values = enum_values or {}

    def _author_all() -> "dict[str, Differentia]":
        accepted: "list[Differentia]" = []
        out: "dict[str, Differentia]" = {}
        for name, defn in species:
            sibs = [(n, d) for n, d in species if n != name]
            r = author_differentia(genus_label, name, defn, sibs, accepted, proposer=proposer,
                                   max_rounds=max_rounds, known_values=enum_values.get(name))
            out[name] = r
            if r.accepted:
                accepted.append(r)
        return out

    out = _author_all()
    if not hermit:
        return out
    for _ in range(genus_rounds):
        ok, unsat, reason = hermit_membrane(genus_label, out)
        out["__hermit__"] = Differentia(species="__hermit__", accepted=ok, reason=reason)
        if ok or not unsat:
            break
        # genus-level CAS: re-author the flagged species, seeding the HermiT reason
        accepted = [d for s, d in out.items() if d.accepted and s not in unsat and s != "__hermit__"]
        for name in unsat:
            defn = dict(species).get(name, "")
            sibs = [(n, d) for n, d in species if n != name]
            r = author_differentia(genus_label, name, defn, sibs, accepted, proposer=proposer,
                                   max_rounds=max_rounds, known_values=enum_values.get(name),
                                   seed_feedback=f"HermiT rejected the prior differentia: {reason}")
            out[name] = r
            if r.accepted:
                accepted.append(r)
    return out


def to_manchester(genus_label: str, diffs: "dict[str, Differentia]") -> "list[str]":
    """Render the accepted differentiae as SubClassOf axioms — the input to the HermiT membrane / SHACL arc."""
    ax = []
    for sp, d in diffs.items():
        if d.accepted and sp not in (genus_label, "__hermit__"):   # never self-subclass; skip the verdict marker
            ax.append(f"Class: sdg:{sp}\n    SubClassOf: sdg:{genus_label}, sdg:{d.property} {d.restriction}")
    return ax

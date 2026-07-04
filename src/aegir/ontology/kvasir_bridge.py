"""kvasir bridge — sound-for-refutation KFS lowering + the ``fast_refute`` seam (Kvasir P0, #135).

kvasir is the greenfield Rust refuter (components/kvasir): a fragment-gated, proof-carrying
consequence engine whose verdicts are self-checked by an independent kernel before they leave the
process. TRUST DOCTRINE ([[greenfield_reasoner_direction]]): kvasir changes HOW FAST, never WHAT is
verified — a kvasir REFUTED short-circuits a membrane round in milliseconds (the proof DAG is the
re-prompt reason), a kvasir no-clash is NOT a certificate and falls through to HermiT unchanged.
HermiT signs every certificate until the differential record earns co-signing.

The LOWERING is sound for refutation: every emitted KFS axiom is ENTAILED by the source Manchester
document (atomic subsumptions/disjointness/typings verbatim; ``r exactly/min n≥1 C`` weakened to
``r some C``; everything else skipped-with-count — fewer axioms can only MISS clashes, never invent
them), so a clash in the lowering is a genuine inconsistency of the source. Skips are counted per
construct and returned loudly — refuse-don't-approximate accounting at the bridge.

Every ``fast_refute`` call appends to ``build/kvasir_differential.jsonl``; when the caller later
obtains HermiT's verdict on the same doc it records the pair via ``differential_record`` — the
trust bridge is MEASURED, never assumed.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KVASIR_BIN = REPO / "components/kvasir/target/release/kvasir"
DIFFERENTIAL = REPO / "build/kvasir_differential.jsonl"

# ── Manchester frame extraction (the realize doc is machine-emitted; shapes are regular) ────────
# frames come in two emitted forms: a bare header with indented section lines, and the one-line
# skeleton form `Class: bfo:0000015 SubClassOf: bfo:0000003` (the numeric-BFO injection)
_FRAME = re.compile(r"^(Class|ObjectProperty|Individual|Datatype|DataProperty|AnnotationProperty)"
                    r":\s*(\S+)([^\n]*)$", re.M)
_SECTION = re.compile(r"^\s+(SubClassOf|EquivalentTo|DisjointWith|Domain|Range|Types):\s*(.*)$")
_INLINE = re.compile(r"\s+(SubClassOf|EquivalentTo|DisjointWith|Domain|Range|Types):\s*(.*)$")
_SOME = re.compile(r"^\(?\s*(\S+)\s+some\s+([^()\s]+)\s*\)?$")
_MIN1 = re.compile(r"^\(?\s*(\S+)\s+(?:exactly|min)\s+([1-9]\d*)\s+([^()\s]+)\s*\)?$")
_ATOM = re.compile(r"^[^\s()]+$")


def _clean(tok: str) -> str:
    return tok.strip().rstrip(",").strip("<>")


def _split_conjuncts(expr: str) -> "list[str]":
    """Split a Manchester conjunction on top-level ``and`` (paren-aware; the emitted docs never
    nest deeper than one paren level around restrictions)."""
    parts, depth, cur = [], 0, []
    for tok in expr.split():
        depth += tok.count("(") - tok.count(")")
        if tok == "and" and depth == 0:
            parts.append(" ".join(cur))
            cur = []
        else:
            cur.append(tok)
    if cur:
        parts.append(" ".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _split_items(rhs: str) -> "list[str]":
    """Split a frame-section rhs on TOP-LEVEL commas (paren-aware). A Manchester section list
    (``SubClassOf: bfo:0000023, p exactly 1 B, q some C``) is SEPARATE AXIOMS, not one class
    expression — treating the whole rhs as one expression silently skipped every comma-form
    frame in the certified artifact (84 ``expr:other`` lines, incl. all role frames and every
    ``exactly 1``; caught 2026-07-04 by the smoke-the-real-artifact rule)."""
    parts, depth, cur = [], 0, []
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _lower_expr(subject: str, expr: str, out: "list[str]", skipped: Counter) -> None:
    """Lower one superclass-position expression for ``subject``. Emits only entailed axioms."""
    expr = expr.strip().rstrip(",")
    if not expr:
        return
    if _ATOM.match(expr) and " some " not in expr:
        out.append(f"SubClassOf <{_clean(subject)}> <{_clean(expr)}>")
        return
    m = _SOME.match(expr)
    if m:
        out.append(f"SubClassOfExistential <{_clean(subject)}> <{_clean(m.group(1))}> "
                   f"<{_clean(m.group(2))}>")
        return
    m = _MIN1.match(expr)
    if m:  # exactly/min n≥1 ⊨ some — the entailed weakening
        out.append(f"SubClassOfExistential <{_clean(subject)}> <{_clean(m.group(1))}> "
                   f"<{_clean(m.group(3))}>")
        return
    key = ("only" if " only " in expr else
           "value" if " value " in expr else
           "max0" if re.search(r"\b(?:max|exactly)\s+0\b", expr) else
           "nested" if "(" in expr else "other")
    skipped[f"expr:{key}"] += 1


def lower_manchester(doc: str) -> "tuple[str, dict]":
    """Lower a Manchester document to KFS text (sound for refutation). Returns (kfs, skip_counts)."""
    out: list[str] = []
    skipped: Counter = Counter()
    # top-level DisjointClasses: a, b, … axioms (the BFO skeleton's form — not inside any frame);
    # n-ary lowers to all pairs (entailed)
    for m in re.finditer(r"^DisjointClasses:\s*(.+)$", doc, re.M):
        atoms = [_clean(t) for t in m.group(1).split(",")]
        atoms = [a for a in atoms if a and _ATOM.match(a)]
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                out.append(f"DisjointClasses <{a}> <{b}>")
    frames = list(_FRAME.finditer(doc))
    for i, fm in enumerate(frames):
        kind, name = fm.group(1), _clean(fm.group(2))
        body = doc[fm.end(): frames[i + 1].start() if i + 1 < len(frames) else len(doc)]
        if kind in ("Datatype", "DataProperty", "AnnotationProperty"):
            skipped[f"frame:{kind}"] += 1
            continue
        sections: "list[tuple[str, str]]" = []
        im = _INLINE.match(fm.group(3) or "")
        if im:  # the one-line skeleton form
            sections.append((im.group(1), im.group(2)))
        for line in body.splitlines():
            sm = _SECTION.match(line)
            if sm:
                sections.append((sm.group(1), sm.group(2)))
        for section, rhs in sections:
            if kind == "Class" and section in ("SubClassOf", "EquivalentTo"):
                # comma items are separate axioms; each item's top-level conjunct is then
                # entailed as a superclass of the subject (≡ entails ⊑ per conjunct)
                for item in _split_items(rhs):
                    for conj in _split_conjuncts(item):
                        _lower_expr(name, conj, out, skipped)
            elif kind == "Class" and section == "DisjointWith":
                for other in rhs.split(","):
                    other = _clean(other)
                    if other and _ATOM.match(other):
                        out.append(f"DisjointClasses <{name}> <{other}>")
                    elif other:
                        skipped["disjoint:non-atomic"] += 1
            elif kind == "ObjectProperty" and section == "Domain":
                d = _clean(rhs)
                if _ATOM.match(d):
                    out.append(f"PropertyDomain <{name}> <{d}>")
                else:
                    skipped["domain:non-atomic"] += 1
            elif kind == "ObjectProperty" and section == "Range":
                r = _clean(rhs)
                if _ATOM.match(r):
                    out.append(f"PropertyRange <{name}> <{r}>")
                else:
                    skipped["range:non-atomic"] += 1
            elif kind == "Individual" and section == "Types":
                for t in rhs.split(","):
                    t = _clean(t)
                    if t and _ATOM.match(t):
                        out.append(f"ClassAssertion <{t}> <{name}>")
                    elif t:
                        skipped["types:non-atomic"] += 1
    return "\n".join(out) + "\n", dict(skipped)


# ── fast_refute: the millisecond membrane pre-pass ───────────────────────────────────────────────
def fast_refute(manchester_doc: str, *, source: str = "unnamed", timeout_s: int = 30) -> dict:
    """Lower → kvasir check → structured verdict. Never raises on engine outcomes; the caller
    branches on ``verdict`` ∈ {refuted, no-clash, out-of-fragment, error, unavailable}.

    A ``refuted`` result carries ``unsat_classes`` + ``reason`` (rendered from the proof DAG's
    input citations — the minimal justification, the agent's re-prompt signal) and is DEFINITIVE
    (the CLI self-checks the proof through the independent kernel before exiting 1). Any other
    result means: proceed to HermiT exactly as before."""
    if not KVASIR_BIN.exists():
        return {"verdict": "unavailable",
                "detail": f"{KVASIR_BIN} missing — cargo build --release in components/kvasir"}
    t0 = time.monotonic()
    kfs, skipped = lower_manchester(manchester_doc)
    n_axioms = sum(1 for ln in kfs.splitlines() if ln.strip())
    with tempfile.NamedTemporaryFile("w", suffix=".kfs", delete=False) as f:
        f.write(kfs)
        path = f.name
    try:
        p = subprocess.run([str(KVASIR_BIN), "check", path, "--json"],
                           capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _logged({"verdict": "error", "detail": f"kvasir timed out at {timeout_s}s",
                        "source": source, "ms": _ms(t0)})
    finally:
        Path(path).unlink(missing_ok=True)
    ms = _ms(t0)
    if p.returncode == 0:
        return _logged({"verdict": "no-clash", "n_axioms": n_axioms, "skipped": skipped,
                        "source": source, "ms": ms})
    if p.returncode == 1:
        v = json.loads(p.stdout)
        reason = _render_reason(kfs, v)
        return _logged({"verdict": "refuted", "unsat_classes": v.get("unsat_classes", []),
                        "refuted_individuals": v.get("refuted_individuals", []),
                        "proof_steps": len(v.get("proof", {}).get("steps", [])),
                        "reason": reason, "n_axioms": n_axioms, "skipped": skipped,
                        "source": source, "ms": ms})
    if p.returncode == 2:
        return _logged({"verdict": "out-of-fragment", "detail": p.stderr.strip()[:300],
                        "source": source, "ms": ms})
    return _logged({"verdict": "error", "detail": p.stderr.strip()[:300],
                    "source": source, "ms": ms})


def _render_reason(kfs: str, verdict: dict) -> str:
    """The minimal justification: the input axioms reachable from the first refutation's proof —
    legible WHY, not just a name (the same contract as HermiT's explain_unsatisfiable)."""
    steps = verdict.get("proof", {}).get("steps", [])
    lines = kfs.splitlines()
    by_id = {s["id"]: s for s in steps}
    target = next((s for s in steps if "KbRefuted" in json.dumps(s.get("conclusion", {}))), None)
    target = target or next((s for s in reversed(steps)
                             if "Unsat" in json.dumps(s.get("conclusion", {}))), None)
    if not target:
        return "refuted (proof rendering unavailable)"
    cited: set[int] = set()
    stack = [target["id"]]
    seen: set[int] = set()
    while stack:
        s = by_id.get(stack.pop())
        if not s or s["id"] in seen:
            continue
        seen.add(s["id"])
        if s.get("axiom") is not None:
            cited.add(s["axiom"])
        stack.extend(s.get("premises", []))
    axs = [lines[i] for i in sorted(cited) if i < len(lines)]
    return "; ".join(axs[:6]) + (" …" if len(axs) > 6 else "")


def exist_cycle_lint(kfs_text: str) -> "list[list[str]]":
    """∃-cycle lint (P0 adjunct) — the tableau-grind early-warning.

    The JointInformationEnvironment incident: activating one ∃-axiom phase-changed HermiT's TBox
    pass 478s → >52min (AnywhereBlocking under expandExistentials). The hazard shape is a CYCLE in
    the ∃-obligation graph: following told subsumption (c ⊑ d inherits d's obligations) and
    existential fillers (c ⊑ ∃r.d obliges a d-successor), a class that can reach itself forces
    unbounded model-building that only blocking terminates — exactly where tableau cost explodes.
    Returns the non-trivial SCCs that contain at least one ∃-edge, each as a sorted class list.
    A lint, not a gate: cycles are legal OWL — the signal is "budget accordingly / consider
    re-authoring", surfaced BEFORE the pass instead of discovered by a jstack 40 minutes in."""
    sub_edges: dict[str, set[str]] = {}
    ex_edges: dict[str, set[str]] = {}
    for line in kfs_text.splitlines():
        toks = line.split()
        if not toks:
            continue
        strip = lambda s: s.strip("<>")  # noqa: E731
        if toks[0] == "SubClassOf" and len(toks) == 3:
            sub_edges.setdefault(strip(toks[1]), set()).add(strip(toks[2]))
        elif toks[0] == "EquivalentToIntersection" and len(toks) >= 3:
            sub_edges.setdefault(strip(toks[1]), set()).update(strip(t) for t in toks[2:])
        elif toks[0] == "SubClassOfExistential" and len(toks) == 4:
            ex_edges.setdefault(strip(toks[1]), set()).add(strip(toks[3]))
    graph: dict[str, set[str]] = {}
    for src, dsts in list(sub_edges.items()) + list(ex_edges.items()):
        graph.setdefault(src, set()).update(dsts)
        for d in dsts:
            graph.setdefault(d, set())
    # iterative Tarjan SCC
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = [0]
    for root in graph:
        if root in index:
            continue
        work = [(root, iter(sorted(graph[root])))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(sorted(graph[w]))))
                    advanced = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1 or v in graph.get(v, set()):
                    sccs.append(sorted(comp))
    # keep only SCCs threaded by at least one ∃-edge (pure told-subsumption cycles are a different,
    # cheaper defect the metrology's cycle proxy already covers)
    out = []
    for comp in sccs:
        cs = set(comp)
        if any(d in cs for c in comp for d in ex_edges.get(c, ())):
            out.append(comp)
    return out


def differential_record(source: str, kvasir_verdict: str, hermit_consistent: "bool | None",
                        **extra) -> None:
    """Record a (kvasir, HermiT) verdict pair — the trust bridge is measured, never assumed.
    Disagreement in the dangerous direction (kvasir refuted, HermiT consistent) would be a
    SOUNDNESS bug and must fail loud in the caller."""
    entry = {"source": source, "kvasir": kvasir_verdict, "hermit_consistent": hermit_consistent,
             "agree": None if hermit_consistent is None
             else (kvasir_verdict == "refuted") == (not hermit_consistent), **extra}
    DIFFERENTIAL.parent.mkdir(parents=True, exist_ok=True)
    with DIFFERENTIAL.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _logged(entry: dict) -> dict:
    DIFFERENTIAL.parent.mkdir(parents=True, exist_ok=True)
    with DIFFERENTIAL.open("a") as f:
        f.write(json.dumps({k: v for k, v in entry.items() if k != "reason"} | {"event": "call"})
                + "\n")
    return entry

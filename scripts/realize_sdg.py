#!/usr/bin/env python
"""Realize the sdg entity-derived ontology: merge → HermiT certify → DDL/shapes → score.

The realize boundary for the entity-centric (post-template) derivation. Inputs are per-passage
entity JSONs (``derive_loop`` output); outputs are the SHIPPED artifacts — the trust doctrine
holds: kvasir fast-refutes in-loop, **HermiT signs the certificate** here.

  build/sdg/<run>/entities/*.json
    → merge_entities → sdg-ontology.omn        (the real ontology, HermiT/DeepOnto-feedable)
    → HermiT                → certificate.json    (consistent, n_classes, unsat)
    → kvasir ddl --sql      → ddl.sql             (proof-carrying, sqlparser-self-checked)
    → kvasir shapes         → shapes.ttl          (SHACL Core — the fixpoint/constraint view)
    → score_ontology_ddl    → structure.json      (width dist + shape EMD vs SchemaPile)

Run: LD_LIBRARY_PATH=$(pwd)/build/jvm-libs uv run --no-sync python scripts/realize_sdg.py \
       --entities-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KVASIR = REPO / "components/kvasir/target/release/kvasir"
if str(REPO) not in sys.path:  # `scripts.` imports when exec'd as a file (sys.path[0] = scripts/)
    sys.path.insert(0, str(REPO))


def _splice_ontology(base_omn: str, extra_path: Path) -> str:
    """GENERATION UNIFICATION (RH 2026-07-21): OMN-text union of an additional realized
    ontology (the catalog realization) into the entity-derived doc. The extra doc's prefix
    header + Ontology: frame are stripped; missing prefixes are added to the base header;
    duplicate bare property/annotation declarations dedup; repeated Class: frames are LEGAL
    OMN (axiom union) — shared-IRI collisions surface as HermiT signals, which is the point."""
    extra = extra_path.read_text()
    base_pfx = dict(re.findall(r"^Prefix:\s*(\w*):\s*<([^>]+)>", base_omn, re.M))
    add_pfx = []
    for k, v in re.findall(r"^Prefix:\s*(\w*):\s*<([^>]+)>", extra, re.M):
        if k not in base_pfx:
            add_pfx.append(f"Prefix: {k}: <{v}>")
            base_pfx[k] = v
    body = re.sub(r"^Prefix:[^\n]*\n", "", extra, flags=re.M)
    body = re.sub(r"^Ontology:[^\n]*\n", "", body, flags=re.M)
    base_kind = {m.group(2): m.group(1) for m in
                 re.finditer(r"^(Object|Data|Annotation)Property:\s*(\S+)$", base_omn, re.M)}
    # cross-generation PROPERTY PUNS (same sdg: name, different kind) are illegal in OWL 2 DL
    # and collapse reasoners globally — shim the SPLICED side to <name>__cat (loud, worklisted;
    # the semantic merge of each pun is explicit review work, never a silent rewrite)
    conflicts = []
    for m in re.finditer(r"^(Object|Data|Annotation)Property:\s*(\S+)$", body, re.M):
        k, nm = m.group(1), m.group(2)
        if nm in base_kind and base_kind[nm] != k:
            conflicts.append({"property": nm, "base_kind": base_kind[nm], "spliced_kind": k})
    for c in conflicts:
        nm = c["property"]
        body = re.sub(rf"(?<![\w:]){re.escape(nm)}(?![\w])", nm + "__cat", body)
        print(f"  PUN SHIM: {nm} ({c['base_kind']} in base vs {c['spliced_kind']} in spliced) "
              f"→ spliced side renamed {nm}__cat (review worklist)")
    if conflicts:
        import json as _json
        Path("build/unification_worklist.json").write_text(_json.dumps(
            {"property_kind_conflicts": conflicts}, indent=1))
    base_decls = set(re.findall(r"^((?:Object|Data|Annotation)Property: \S+)$", base_omn, re.M))
    body_lines = []
    for ln in body.split("\n"):
        if ln.strip() and re.fullmatch(r"(?:Object|Data|Annotation)Property: \S+", ln.strip()) \
                and ln.strip() in base_decls:
            continue
        body_lines.append(ln)
    if add_pfx:
        base_omn = base_omn.replace("Ontology:", "\n".join(add_pfx) + "\nOntology:", 1)
    return base_omn + "\n\n" + "\n".join(body_lines)




_CONT = {"bfo:0000002", "bfo:0000004", "bfo:0000040", "bfo:0000031", "bfo:0000020",
         "bfo:0000019", "bfo:0000027", "cco:ont00000995", "cco:ont00000958"}
_OCC = {"bfo:0000003", "bfo:0000015", "bfo:0000017", "bfo:0000023"}


def _split_items(rest: str) -> "list[str]":
    """Split a Manchester expression list on TOP-LEVEL commas (depth- and quote-aware) —
    the only safe grain for editing SubClassOf/EquivalentTo lists."""
    items, depth, in_str, cur = [], 0, False, ""
    for i, ch in enumerate(rest):
        if ch == '"' and (i == 0 or rest[i - 1] != "\\"):
            in_str = not in_str
        elif not in_str:
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                depth -= 1
            elif ch == "," and depth == 0:
                items.append(cur)
                cur = ""
                continue
        cur += ch
    if cur.strip():
        items.append(cur)
    return items


_IRI2PFX = [
    (re.compile(r"<https://signals\.zndx\.org/sdg#([\w.-]+)>"), r"sdg:\1"),
    (re.compile(r"<http://purl\.obolibrary\.org/obo/BFO_(\d{7})>"), r"bfo:\1"),
    (re.compile(r"<https://www\.commoncoreontologies\.org/(ont\d+)>"), r"cco:\1"),
]


def _norm_iris(text: str) -> str:
    for rx, rep in _IRI2PFX:
        text = rx.sub(rep, text)
    return text


_BARE = re.compile(r"^(?:bfo:\d{7}|cco:\w+|sdg:[\w.-]+)$")
# BFO 2020 sides — realizables (0000017), roles (0000023), functions (0000034) and
# dispositions (0000016) are CONTINUANTS (the v1 misassignment fed the false worklist).
_CONT = {"bfo:0000002", "bfo:0000004", "bfo:0000040", "bfo:0000031", "bfo:0000020",
         "bfo:0000019", "bfo:0000027", "bfo:0000017", "bfo:0000023", "bfo:0000034",
         "bfo:0000016", "cco:ont00000995", "cco:ont00000958"}
_OCC = {"bfo:0000003", "bfo:0000015", "bfo:0000035", "bfo:0000008", "bfo:0000011"}
_SECT = re.compile(r"^(\s*(?:SubClassOf|EquivalentTo):\s*)(.*)$")
# Frame: prefixed OR full-IRI name; body on the SAME line (catalog single-line form)
# and/or 4-space-indented following lines. Full-IRI headers are the CATALOG (certified)
# side of the union; prefixed multi-line headers are the entity side.
_FRAME = re.compile(r"^Class:\s*(<[^>\n]+>|(?:sdg|bfo|cco):[\w.-]+)([^\n]*)\n"
                    r"((?:    [^\n]*\n)*)", re.M)


def _reconcile_categories(omn: str) -> "tuple[str, list]":
    """Tier-1 mechanical category repair (#27 pattern; RH worklist 3), justification-shaped:

    The 1,836-class TBox cascade flows from a handful of WRONG-SIDE spine anchors
    (e.g. EducationalEntity ⊑ bfo:0000015) and direct double-category assertions
    (e.g. SupportService ⊑ bfo:0000002 AND ⊑ bfo:0000015) through ∃/min-card filler
    propagation. Repairs, in order:

      A. ANCHOR FLIP — an sdg class whose direct category anchor is contradicted by its
         own SUBTREE's direct anchors (opposite side ≥ 3 classes and ≥ 2× same side)
         gets the anchor rewritten to the subtree-majority generic root.
      B. MINORITY DROP — an sdg class carrying direct parents on BOTH sides keeps the
         evidence-majority side; ties prefer the side asserted by the CATALOG (certified)
         origin, then continuant. Losing bare items are dropped from its frames.

    Only bare TOP-LEVEL items are ever rewritten (never conjuncts inside ``and`` — those
    count as evidence only). Every action is worklisted, never silent."""
    frames: "dict[str, list]" = {}          # cls → [(header_tail, body, origin), ...]
    for m in _FRAME.finditer(omn):
        name = _norm_iris(m.group(1))
        origin = "catalog" if m.group(1).startswith("<") else "entity"
        frames.setdefault(name, []).append((m.group(2), m.group(3), origin))

    parents: "dict[str, set]" = {}          # bare top-level items only
    conj_parents: "dict[str, set]" = {}     # bare conjuncts of and-chains (evidence only)
    origins: "dict[tuple, set]" = {}        # (cls, parent) → asserting origins
    for cls, fl in frames.items():
        ps, cps = set(), set()
        for tail, body, origin in fl:
            for line in ([tail] if tail.strip() else []) + body.splitlines():
                sm = _SECT.match(line)
                if not sm:
                    continue
                for it in _split_items(_norm_iris(sm.group(2))):
                    it = it.strip()
                    if _BARE.match(it):
                        if it != cls:
                            ps.add(it)
                            origins.setdefault((cls, it), set()).add(origin)
                    elif " and " in it:
                        for c_ in re.split(r"\s+and\s+", it):
                            c_ = c_.strip().strip("()")
                            if _BARE.match(c_) and c_ != cls:
                                cps.add(c_)
        parents[cls] = ps
        conj_parents[cls] = cps

    children: "dict[str, set]" = {}
    for c, ps in parents.items():
        for p_ in ps:
            children.setdefault(p_, set()).add(c)

    memo: "dict[str, frozenset]" = {}

    def cats(c: str, seen=frozenset()) -> frozenset:
        if c in memo:
            return memo[c]
        if c in seen:
            return frozenset()
        out = set()
        if c in _CONT:
            out.add("cont")
        if c in _OCC:
            out.add("occ")
        for p_ in parents.get(c, set()) | conj_parents.get(c, set()):
            out |= cats(p_, seen | {c})
        r = frozenset(out)
        memo[c] = r
        return r

    def side(c: str) -> str:
        k = cats(c)
        return "both" if k == {"cont", "occ"} else ("cont" if "cont" in k
                                                    else "occ" if "occ" in k else "none")

    def direct_anchor_sides(c: str) -> "set[str]":
        out = set()
        for p_ in parents.get(c, ()):
            if p_ in _CONT:
                out.add("cont")
            elif p_ in _OCC:
                out.add("occ")
        return out

    def subtree_evidence(root: str) -> "tuple[int, int]":
        seen, stack, c_n, o_n = set(), list(children.get(root, ())), 0, 0
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            ds = direct_anchor_sides(c)
            c_n += "cont" in ds
            o_n += "occ" in ds
            stack.extend(children.get(c, ()))
        return c_n, o_n

    def rewrite_frames(cls: str, xform) -> None:
        """Apply xform(items)->items to every SubClassOf/EquivalentTo section of every
        frame of cls (both name forms, both frame shapes)."""
        nonlocal omn

        def _do(mm):
            name = _norm_iris(mm.group(1))
            if name != cls:
                return mm.group(0)
            parts = []
            for line in ([mm.group(2)] if mm.group(2).strip() else []) + \
                        mm.group(3).splitlines():
                sm = _SECT.match(line)
                if not sm:
                    parts.append(line)
                    continue
                # split RAW text; xform matches in normalized space but emission keeps
                # the original tokens — prefixed names need a PRIOR declaration where a
                # full IRI never does, so normalizing the output breaks forward refs.
                items = [x.strip() for x in _split_items(sm.group(2))]
                kept = xform(items)
                if kept:
                    parts.append(sm.group(1) + ", ".join(kept))
            head = f"Class: {mm.group(1)}"
            if not parts:
                return head + "\n"
            # re-emit uniformly as indented body lines (legal OMN, both input shapes)
            return head + "\n" + "".join(
                ("    " + x.lstrip() + "\n") if x.strip() else "" for x in parts)
        omn = _FRAME.sub(_do, omn)

    worklist = []

    # ---- phase A: anchor flips on subtree evidence (batch-decided, then applied)
    flips = []
    for cls in sorted(frames):
        if not cls.startswith("sdg:"):
            continue
        for a_ in sorted(parents.get(cls, ())):
            a_side = "cont" if a_ in _CONT else "occ" if a_ in _OCC else None
            if not a_side:
                continue
            c_n, o_n = subtree_evidence(cls)
            opp_n, same_n = (o_n, c_n) if a_side == "cont" else (c_n, o_n)
            if opp_n >= 3 and opp_n >= 2 * max(same_n, 1):
                tgt = "bfo:0000003" if a_side == "cont" else "bfo:0000002"
                flips.append((cls, a_, tgt))
                worklist.append({"class": cls, "action": "flip_anchor",
                                 "from": a_, "to": tgt,
                                 "subtree_cont": c_n, "subtree_occ": o_n})
    for cls, a_, tgt in flips:
        rewrite_frames(cls, lambda items, a_=a_, tgt=tgt:
                       [tgt if _norm_iris(x) == a_ else x for x in items])
        parents[cls] = (parents[cls] - {a_}) | {tgt}
    memo.clear()

    # ---- phase B: direct both-side minority drops
    for cls in sorted(frames):
        if not cls.startswith("sdg:"):
            continue
        direct = parents.get(cls, set())
        cont_p = sorted(p_ for p_ in direct if side(p_) == "cont")
        occ_p = sorted(p_ for p_ in direct if side(p_) == "occ")
        if not cont_p or not occ_p:
            continue
        c_n, o_n = subtree_evidence(cls)
        c_ev = len(cont_p) + c_n
        o_ev = len(occ_p) + o_n
        if c_ev != o_ev:
            keep = "cont" if c_ev > o_ev else "occ"
        else:
            cat_sides = {("cont" if side(p_) == "cont" else "occ")
                         for p_ in direct if "catalog" in origins.get((cls, p_), set())}
            keep = next(iter(cat_sides)) if len(cat_sides) == 1 else "cont"
        drop = set(occ_p if keep == "cont" else cont_p)
        rewrite_frames(cls, lambda items, drop=drop:
                       [x for x in items if _norm_iris(x) not in drop])
        worklist.append({"class": cls, "action": "drop_minority",
                         "kept": "continuant" if keep == "cont" else "occurrent",
                         "dropped_parents": sorted(drop),
                         "evidence": {"cont": c_ev, "occ": o_ev},
                         "catalog_asserted": sorted(
                             p_ for p_ in direct
                             if "catalog" in origins.get((cls, p_), set()))})
        parents[cls] = direct - drop
        memo.clear()

    # ---- phase C: justification-derived cuts (derive_category_cuts.py output).
    # The ≡-identity-genus rule adjudicates edges the closure machinery cannot
    # (disjoint continuant sub-branches, domain-forced categories). Imported truth
    # is never cut; escalations in the cuts file stay CAS work.
    cuts_file = Path("build/category_cuts.json")
    if cuts_file.exists():
        for c_ in json.loads(cuts_file.read_text()).get("cuts", []):
            cls, gone = c_["class"], c_["cut_parent"]
            if cls not in frames:
                continue
            rewrite_frames(cls, lambda items, gone=gone:
                           [x for x in items if _norm_iris(x) != gone])
            worklist.append({"class": cls, "action": "cut_by_justification",
                             "dropped_parents": [gone], "genus": c_.get("genus"),
                             "via": c_.get("via")})

    # ---- phase D: functional over-assertion (measured 2026-07-22: ablating the 7,601
    # entity-side Characteristics: Functional lines collapses the full residual cascade
    # 1,810 → 0; the catalog asserts ZERO functionality). A per-passage "exactly one"
    # reading does not license a GLOBAL functional axiom on a property name shared
    # across 454 independent passages — union scope-overreach, dropped wholesale.
    # Generation-side fix (emit Functional only under union-wide type-consistency
    # evidence) is the durable repair; this is the shakedown tier-1.
    n_func = len(re.findall(r"^\s*Characteristics:\s*Functional\s*$", omn, flags=re.M))
    if n_func:
        omn = re.sub(r"^\s*Characteristics:\s*Functional\s*$\n?", "", omn, flags=re.M)
        omn = re.sub(r"(Characteristics:[^\n]*?)\bFunctional,\s*", r"\1", omn)
        omn = re.sub(r",\s*Functional(\s*$)", r"\1", omn, flags=re.M)
        worklist.append({"action": "drop_functional_overassertion", "n": n_func,
                         "reason": "per-passage =1 readings do not lift to union-global "
                                   "functional axioms (catalog asserts none)"})
    return omn, worklist


def realize(entities_dir: Path, output_dir: Path, *, skip_hermit: bool = False,
            merge_ontology: "Path | None" = None, ground_entity_props: bool = False,
            reconcile_categories: bool = False, arm_property_domains: bool = False,
            inline_theory: bool = False) -> dict:
    from aegir.ontology.derive_loop import merge_entities
    from aegir.ontology.entities import from_json, to_manchester

    sets = []
    for p in sorted(entities_dir.glob("*.json")):
        obj = json.loads(p.read_text())
        sets.append(from_json(obj if "entities" in obj else {"entities": obj}))
    merged = merge_entities(sets)
    omn = to_manchester(merged)
    if merge_ontology:
        omn = _splice_ontology(omn, merge_ontology)
        print(f"unified: merged ontology {merge_ontology} spliced (prefixes reconciled)")
    if reconcile_categories:
        omn, _wl = _reconcile_categories(omn)
        Path("build/category_reconciliation.json").write_text(json.dumps(
            {"n": len(_wl), "repairs": _wl}, indent=1))
        print(f"category reconciliation: {len(_wl)} double-category classes repaired "
              f"(majority side kept) → build/category_reconciliation.json")
    if inline_theory:
        theory = Path("build/grounding/cco-taxonomy.omn")
        if theory.exists():
            omn += "\n" + theory.read_text()
            print(f"inlined theory: {theory} (told CCO taxonomy — text-tier consumers "
                  f"see the same theory HermiT imports)")
    if arm_property_domains:
        sys.path.insert(0, str(REPO / "src"))
        from aegir.ontology.property_domains import DOMAINS_OMN, armed_domains_omn
        armed_omn = armed_domains_omn()   # one accessor everywhere: demoted ∪ deferred filtered
        n_frames = armed_omn.count("Domain:")
        n_all = DOMAINS_OMN.count("Domain:")
        omn += "\n" + armed_omn                       # OMN has no comment syntax — bare append
        print(f"armed: {n_frames} derived rdfs:domain frames appended "
              f"({n_all - n_frames} excluded: demoted ∪ deferred)")
    if ground_entity_props:
        # SIGNATURE-CHECKED grounding v2 (RH adjudication 2026-07-22): stem-ground the
        # entity-side sdg: properties, but check EVERY usage site against the BFO
        # property's category signature first — conforming sites ground; partOf sites
        # matching the occ→occ MIRROR split to sdg:occurrentPartOf ⊑ bfo:0000132;
        # contradicting sites are rewritten to <prop>__escalated (declared, UNgrounded)
        # and worklisted — escalated ≠ dropped, per the never-drop organ.
        from aegir.ontology.relation_signatures import (grounding_of, BFO_SIGNATURES,
                                                        MIRROR_SPLIT)
        props = set(re.findall(r"^ObjectProperty:\s*sdg:(\w+)$", omn, re.M))
        grounded = {q: grounding_of(q) for q in sorted(props) if grounding_of(q)}

        # self-contained category lift over the CURRENT text (theory already inlined
        # when enabled) — bare parents + ≡-conjunct genera, the standing harvest shape
        g_parents: "dict[str, set]" = {}
        for fm in _FRAME.finditer(omn):
            cname = _norm_iris(fm.group(1))
            ps = g_parents.setdefault(cname, set())
            for line in ([fm.group(2)] if fm.group(2).strip() else []) + fm.group(3).splitlines():
                sm = _SECT.match(line)
                if not sm:
                    continue
                for it in _split_items(_norm_iris(sm.group(2))):
                    it = it.strip()
                    if _BARE.match(it) and it != cname:
                        ps.add(it)
                    elif " and " in it:
                        for c_ in re.split(r"\s+and\s+", it):
                            c_ = c_.strip().strip("()")
                            if _BARE.match(c_) and c_ != cname:
                                ps.add(c_)
        g_memo: "dict[str, frozenset]" = {}

        def g_cats(c: str, seen=frozenset()) -> frozenset:
            if c in g_memo:
                return g_memo[c]
            if c in seen:
                return frozenset()
            out = set()
            if c in _CONT:
                out.add("cont")
            if c in _OCC:
                out.add("occ")
            for p_ in g_parents.get(c, ()):
                out |= g_cats(p_, seen | {c})
            r = frozenset(out)
            g_memo[c] = r
            return r

        def _side_of(c: str) -> str:
            k = g_cats(c) if c else frozenset()
            return "cont" if k == {"cont"} else "occ" if k == {"occ"} else "?"

        escal: "list[dict]" = []
        n_split = 0
        # measured-refutation site ledger (the grounding-grain demotion loop): sites a
        # kvasir witness implicated in a prior round escalate REGARDLESS of the local
        # side-judgment — the checker in the loop is the theory-strength engine, not
        # the hand lift (the ConferenceMarketplace lesson: "?" subjects can be
        # theory-determined).
        site_ledger: "set[tuple]" = set()
        led_file = Path("build/grounding_site_escalations.json")
        if led_file.exists():
            for e_ in json.loads(led_file.read_text()).get("sites", []):
                site_ledger.add((e_["class"], e_["property"], e_["filler"]))

        def _site_rewrite(fm):
            nonlocal n_split
            subj = _norm_iris(fm.group(1))
            def _one(um):
                nonlocal n_split
                q = um.group(1)
                if q not in grounded:
                    return um.group(0)
                bfo = grounded[q]
                exp_s, exp_f = BFO_SIGNATURES.get(bfo, ("any", "any"))
                filler = _norm_iris(um.group(3)).rstrip(",")
                s_side = _side_of(subj)
                f_side = _side_of(filler) if _BARE.match(filler) else "?"
                if (subj, q, filler) in site_ledger or \
                        (subj, f"{q}", filler) in site_ledger:
                    escal.append({"property": q, "bfo": bfo, "class": subj,
                                  "filler": filler, "subject_side": s_side,
                                  "filler_side": f_side, "via": "witness-ledger"})
                    return um.group(0).replace(f"sdg:{q}", f"sdg:{q}__escalated", 1)
                ok = ((exp_s == "any" or s_side in (exp_s, "?")) and
                      (exp_f == "any" or f_side in (exp_f, "?")))
                if ok:
                    return um.group(0)
                mirror = MIRROR_SPLIT.get(bfo)
                if mirror:
                    m_bfo, coin = mirror
                    m_s, m_f = BFO_SIGNATURES[m_bfo]
                    if (s_side in (m_s, "?") and f_side in (m_f, "?")
                            and (s_side, f_side) != ("?", "?")
                            and (subj, coin, filler) not in site_ledger):
                        n_split += 1
                        return um.group(0).replace(f"sdg:{q}", f"sdg:{coin}", 1)
                escal.append({"property": q, "bfo": bfo, "class": subj,
                              "filler": filler, "subject_side": s_side,
                              "filler_side": f_side,
                              "expected": {"subject": exp_s, "filler": exp_f}})
                return um.group(0).replace(f"sdg:{q}", f"sdg:{q}__escalated", 1)
            return re.sub(r"sdg:(\w+)(\s+(?:some|only|exactly \d+|min \d+|max \d+)\s+)(\S+)",
                          _one, fm.group(0))

        omn = _FRAME.sub(_site_rewrite, omn)

        deferred_lifts: "set[str]" = set()
        dl_file = Path("build/grounding_lift_deferrals.json")
        if dl_file.exists():
            deferred_lifts = set(json.loads(dl_file.read_text()).get("deferred_lifts", []))
        lines = []
        for q, bfo in grounded.items():
            if q in deferred_lifts:
                # tractability-deferred lift (inverse-pair activation): declared, not
                # lifted — semantics ratified, proof pending decomposed certification
                lines.append(f"ObjectProperty: sdg:{q}")
                continue
            lines.append(f"ObjectProperty: sdg:{q}\n    SubPropertyOf: {bfo}")
        for bfo, (m_bfo, coin) in MIRROR_SPLIT.items():
            if f"sdg:{coin}" in omn:
                lines.append(f"ObjectProperty: sdg:{coin}\n    SubPropertyOf: {m_bfo}")
        for e_ in sorted({x["property"] for x in escal}):
            lines.append(f"ObjectProperty: sdg:{e_}__escalated")
        omn += "\n\n" + "\n".join(lines) + "\n"
        Path("build/grounding_escalations.json").write_text(json.dumps(
            {"n_sites": len(escal), "sites": escal}, indent=1))
        print(f"grounded {len(grounded)}/{len(props)} entity properties by stem "
              f"(signature-checked: {n_split} sites split to occurrentPartOf · "
              f"{len(escal)} sites escalated) → build/grounding_escalations.json")
    # OL participation (RH 2026-07-23): kvasir emits OpenLineage RunEvents natively
    # (KVASIR_OL_DIR, file transport); after the run they ingest into the governed
    # provenance (governance.ol → aegir_hx/Atlas) — every stage, deterministic and
    # not, lands in PROVENANCE.
    import os as _os
    _os.environ.setdefault("KVASIR_OL_DIR", str(Path("build/ol_events").resolve()))
    output_dir.mkdir(parents=True, exist_ok=True)
    omn_path = output_dir / "sdg-ontology.omn"
    omn_path.write_text(omn)
    print(f"merged {sum(len(s) for s in sets)} entities from {len(sets)} passages "
          f"→ {len(merged)} classes → {omn_path}")

    # ARTIFACTS FIRST (RH shakedown doctrine): shapes/ddl/structure land even when the
    # certificate refuses — exit-3-before-kvasir would blind the unification shakedown.
    for sub, out in (("ddl", "ddl.sql"), ("shapes", "shapes.ttl")):
        args = [str(KVASIR), sub, str(omn_path)] + (["--sql"] if sub == "ddl" else [])
        r = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            (output_dir / out).write_text(r.stdout)
        else:
            print(f"WARN kvasir {sub} failed: {r.stderr[:200]}", file=sys.stderr)

    structure: dict = {}
    try:
        from scripts.score_ontology_ddl import score
        structure = score(omn_path)
        (output_dir / "structure.json").write_text(json.dumps(structure, indent=2, default=str))
        print(f"structure: {structure['n_elected']} elected | {structure['total_fks']} FKs | "
              f"{structure['n_junctions']} junctions | {structure['n_lookups']} lookups | "
              f"median {structure['width']['median']} | EMD {structure['shape_emd']}")
    except Exception as e:  # noqa: BLE001 — shakedown doctrine: a refused score never blinds
        print(f"WARN structure scoring unavailable: {str(e)[:200]}", file=sys.stderr)

    cert: dict = {"reasoner": "HermiT", "skipped": skip_hermit}
    if not skip_hermit:
        from aegir.ontology.deeponto_harness import ensure_jvm
        ensure_jvm()  # MUST precede any deeponto import (click.prompt hangs non-interactively)
        from scripts.build_realized_ontology import _reason
        _onto, _tmp, consistent, n_classes, unsat, why = _reason(omn, explain=bool(0))
        if int(n_classes) == 0 and merged:
            raise RuntimeError(
                f"vacuous HermiT parse: {len(merged)} merged classes but OWLAPI loaded 0 — "
                "the certificate would be meaningless; inspect the omn for punning/declaration "
                "issues (OWLAPI warnings name the entities)")
        cert.update({"isConsistent": bool(consistent), "n_classes": int(n_classes),
                     "unsat": list(unsat or [])})
        print(f"HermiT: consistent={consistent} classes={n_classes} unsat={len(unsat or [])}")
        if unsat:
            cert["why"] = why or {}
            (output_dir / "certificate.json").write_text(json.dumps(cert, indent=2))
            names = [u.rsplit('#', 1)[-1] for u in list(unsat)[:5]]
            raise SystemExit(
                f"REFUSED: {len(unsat)} unsatisfiable classes (first: {names}) — a sick TBox "
                "is not scored or shipped; certificate.json carries the full list (exit 3)")
    (output_dir / "certificate.json").write_text(json.dumps(cert, indent=2))

    return {"n_classes": len(merged), "certificate": cert, "structure": structure,
            "omn": str(omn_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--skip-hermit", action="store_true",
                    help="fast pre-flight only (kvasir); the certificate step needs the JVM")
    ap.add_argument("--merge-ontology", type=Path, default=None,
                    help="GENERATION UNIFICATION: splice an additional realized OMN (the "
                         "catalog realization) into the union before certify/emit")
    ap.add_argument("--inline-theory", action="store_true",
                    help="append build/grounding/cco-taxonomy.omn (told module taxonomy) "
                         "so text-tier consumers judge with the full theory")
    ap.add_argument("--arm-property-domains", action="store_true",
                    help="append the W1-derived DOMAINS_OMN frames (worklist-4 shakedown; "
                         "kvasir witnesses gives the fast structural verdict, HermiT gates)")
    ap.add_argument("--reconcile-categories", action="store_true",
                    help="tier-1 mechanical repair: drop minority-side parents of "
                         "double-category classes (worklisted, never silent)")
    ap.add_argument("--ground-entity-props", action="store_true",
                    help="ground bare entity sdg: properties by stem (signatures/domains bite; "
                         "phase-2 signal harvest)")
    a = ap.parse_args()
    res = realize(a.entities_dir, a.output_dir, skip_hermit=a.skip_hermit,
                  merge_ontology=a.merge_ontology, ground_entity_props=a.ground_entity_props,
                  reconcile_categories=a.reconcile_categories,
                  arm_property_domains=a.arm_property_domains, inline_theory=a.inline_theory)
    ok = res["certificate"].get("isConsistent", True) is not False
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        print(e, file=sys.stderr)
        rc = 3
    from aegir.utils.clean_exit import clean_exit
    clean_exit(rc)


def ingest_ol_events(events_dir: "Path | str" = "build/ol_events") -> int:
    """Ingest kvasir-emitted OpenLineage RunEvents into the governed provenance
    (governance.ol → aegir_hx). Idempotent (MERGE); ingested files gain a .ingested
    marker so re-runs skip them. Call after any kvasir-invoking pass."""
    import json as _json
    d = Path(events_dir)
    if not d.exists():
        return 0
    try:
        from aegir.governance.ol import ingest_run_event
    except Exception:  # noqa: BLE001 — no DB in this environment; events stay on disk
        return 0
    n = 0
    for f in sorted(d.glob("*.json")):
        marker = f.with_suffix(f.suffix + ".ingested")
        if marker.exists():
            continue
        try:
            ingest_run_event(_json.loads(f.read_text()))
            marker.touch()
            n += 1
        except Exception:  # noqa: BLE001 — a bad event never blocks the pass
            continue
    if n:
        print(f"OL: ingested {n} kvasir run event(s) into governed provenance")
    return n

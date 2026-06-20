#!/usr/bin/env python
"""Head-to-head: the template (parse-tree recomposition) verbalizer vs DeepOnto's single-string output.

The honest question (RH): is the template-based verbalizer *superior* to DeepOnto's verbaliser — or even
close? This produces the evidence on the axes that matter for a pretraining corpus:

  * **Diversity** — distinct syntactic skeletons, top-frame dominance, top-5 share, skeleton + n-gram
    entropy, distinct-skeletons-per-template. (A flat "X is a Y"-dominated corpus is low-value.)
  * **Faithfulness** — fraction of template frames whose ``{slot}`` set EXACTLY equals DeepOnto's for the
    same template. The template frames are *recomposed from DeepOnto's own CfgNode parse tree* (subject,
    property, quantifier, filler), so they inherit its semantics; this checks nothing is dropped/added.
  * **Capability** — DeepOnto emits ONE declarative form per axiom; the template verbalizer adds procedural
    / normative / relational-fronted framings DeepOnto cannot produce. Side-by-side examples let a reader
    judge readability directly.

Arm A (DeepOnto) = each template's ``verbal_template``. Arm B (template) = the frames json from
``build_verbalization_frames`` (or ``--frames``). No JVM needed — reads precomputed artifacts.

    uv run --no-sync python scripts/compare_verbalizers.py --frames /tmp/verbal_frames.json --md report.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aegir.ontology.schema import load_catalog  # noqa: E402
from audit_verbalization_entropy import skeleton, _ngram_entropy  # noqa: E402

_SLOT_RE = re.compile(r"\{[^}]*\}")


def _slots(t: str) -> set[str]:
    return set(_SLOT_RE.findall(t))


def _entropy(counts) -> float:
    tot = sum(counts)
    return -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0) if tot else 0.0


def _arm_stats(verbals: list[str]) -> dict:
    sk = Counter(skeleton(v) for v in verbals)
    n = len(verbals) or 1
    return {
        "n_verbalizations": len(verbals),
        "distinct_skeletons": len(sk),
        "top1_share": round(sk.most_common(1)[0][1] / n, 3) if sk else 0.0,
        "top5_share": round(sum(c for _, c in sk.most_common(5)) / n, 3),
        "skeleton_entropy_bits": round(_entropy(list(sk.values())), 3),
        "bigram_entropy_bits": round(_ngram_entropy(verbals, 2), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="src/aegir/ontology/catalog/combined.json")
    ap.add_argument("--frames", default="/tmp/verbal_frames.json",
                    help="json {template_id: [frames]} from build_verbalization_frames")
    ap.add_argument("--md", default=None, help="write a markdown report here")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    cat = load_catalog(Path(args.catalog))
    frames_by_tid = json.loads(Path(args.frames).read_text())
    by_tid = {t.template_id: t for t in cat.templates}

    # only templates present in BOTH arms (fair comparison on the same set)
    tids = [tid for tid in frames_by_tid if by_tid.get(tid) and (by_tid[tid].verbal_template or "").strip()]
    deeponto = [(by_tid[tid].verbal_template or "").strip() for tid in tids]
    template = [f for tid in tids for f in frames_by_tid[tid]]

    A = _arm_stats(deeponto)
    B = _arm_stats(template)
    A["distinct_per_template"] = 1.0
    B["distinct_per_template"] = round(
        sum(len(set(skeleton(f) for f in frames_by_tid[tid])) for tid in tids) / len(tids), 2)

    # Faithfulness measured against the AXIOM's declared slots (the ground truth), not DeepOnto's possibly-
    # lossy string. Three facts: (1) no invented slots — every frame's slots ⊆ declared; (2) internal
    # consistency — all frames of a template share one slot set (the generator can substitute uniformly);
    # (3) clause recovery — templates where the frames cover MORE axiom slots than DeepOnto did (DeepOnto
    # drops conjuncts; e.g. existential_two_clauses → DeepOnto keeps 1 of 2 clauses).
    no_invented = total = 0
    consistent_templates = recovery = 0
    invented_examples = []
    recovery_examples = []
    for tid in tids:
        declared = {"{" + k + "}" for k in by_tid[tid].slot_types}
        fr = frames_by_tid[tid]
        fr_slotsets = [_slots(f) for f in fr]
        for f, s in zip(fr, fr_slotsets):
            total += 1
            if s <= declared:
                no_invented += 1
            elif len(invented_examples) < 5:
                invented_examples.append((tid, f, sorted(s - declared)))
        if len({frozenset(s) for s in fr_slotsets}) == 1:
            consistent_templates += 1
        dont = _slots(by_tid[tid].verbal_template or "")
        frame_union = set().union(*fr_slotsets) if fr_slotsets else set()
        if frame_union > dont:  # frames cover strictly more axiom slots than DeepOnto
            recovery += 1
            if len(recovery_examples) < 4:
                recovery_examples.append((tid, sorted(dont), sorted(frame_union)))
    no_invented_pct = round(100 * no_invented / max(1, total), 2)
    consistent_pct = round(100 * consistent_templates / max(1, len(tids)), 2)

    def fmt(label, key, better):
        a, b = A[key], B[key]
        arrow = "→" if a == b else ("↑" if (b > a) == (better == "up") else "↓")
        win = "template" if ((b > a) == (better == "up")) and a != b else ("deeponto" if a != b else "tie")
        return f"| {label} | {a} | {b} | {arrow} {win} |"

    lines = []
    lines.append(f"# Verbalizer head-to-head — DeepOnto vs template ({len(tids)} templates)\n")
    lines.append("| metric | DeepOnto (1/template) | template frames | winner |")
    lines.append("|---|---|---|---|")
    lines.append(fmt("verbalizations", "n_verbalizations", "up"))
    lines.append(fmt("distinct skeletons", "distinct_skeletons", "up"))
    lines.append(fmt("top-1 frame share", "top1_share", "down"))
    lines.append(fmt("top-5 frame share", "top5_share", "down"))
    lines.append(fmt("skeleton entropy (bits)", "skeleton_entropy_bits", "up"))
    lines.append(fmt("bigram entropy (bits)", "bigram_entropy_bits", "up"))
    lines.append(fmt("distinct skeletons / template", "distinct_per_template", "up"))
    lines.append(f"\n**Faithfulness** (vs the axiom's declared slots — the ground truth):")
    lines.append(f"  - **no invented slots:** {no_invented_pct}% of {total} frames have slots ⊆ the "
                 f"template's declared slots ({'no hallucinated slots' if no_invented==total else 'see below'}).")
    lines.append(f"  - **internal consistency:** {consistent_pct}% of {len(tids)} templates have all frames "
                 f"sharing one slot set (uniform substitution by the generator).")
    lines.append(f"  - **clause recovery:** in {recovery} templates the frames cover STRICTLY MORE axiom "
                 f"slots than DeepOnto's string (DeepOnto drops conjuncts; the template verbalizer recovers them).")
    for tid, dont, frame in recovery_examples:
        lines.append(f"      e.g. {tid}: DeepOnto slots {dont} → template slots {frame}")
    for tid, f, extra in invented_examples:
        lines.append(f"  - INVENTED slot in {tid}: {extra} — {f!r}")

    lines.append(f"\n## Side-by-side (first {args.examples})\n")
    for tid in tids[:args.examples]:
        lines.append(f"**{tid}** — DeepOnto: `{by_tid[tid].verbal_template}`")
        for f in frames_by_tid[tid]:
            lines.append(f"  - {f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.md:
        Path(args.md).write_text(report + "\n")
        print(f"\nwrote {args.md}")
    # machine-readable summary
    print("\nJSON:", json.dumps({"deeponto": A, "template": B, "no_invented_slots_pct": no_invented_pct,
                                 "internal_consistency_pct": consistent_pct, "clause_recovery_templates": recovery,
                                 "n_templates": len(tids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

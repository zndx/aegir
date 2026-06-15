#!/usr/bin/env python
"""G-cov — register-fair coverage-close (R1).

The E5 coverage metric (formal-template ↔ document-centroid cosine) was
register-confounded (R0 confirmed: formal Manchester text vs prose centroid caps
cosine). R1 compares LIKE WITH LIKE at the DOMAIN-TERM level:

  V_c (construct)  = minted domain terms — template_id tokens + slot names +
                     verbal content terms, minus BFO/CCO/ontology boilerplate.
  V_t (topic)      = top TF-IDF terms of the topic (over the 200 topic reprs).
  R1(c, t)         = max( weighted-Jaccard(V_c, V_t), cosine(mean-embed terms) ).

Register-fair (short noun-phrase terms vs terms). Rewards exactly what the
generator must do — mint domain-specific concepts for a gap topic.

INSTRUMENT VALIDATION (must pass before R1 gates): a construct's R1 to its OWN
target topic beats its R1 to a shuffled topic, CI-clean (specificity). Seeds are
generic (slots X/Y/p → ~no domain terms) so the bar is "beats the shuffled null
+ beats the generic-seed floor", not an absolute seed threshold. Deterministic
except the null RNG.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from aegir.ontology.schema import CatalogTemplate, load_catalog  # noqa: E402

WORD = re.compile(r"[a-z][a-z]{2,}")
# Ontology/BFO/CCO boilerplate + generic relational glue — NOT domain content.
BOILER = set("""entity process artifact information content descriptive directive designative
continuant independent material object quality realizable role function occurrent thing
has have is are be the of in on to for with by and or not some only min max exactly value
that this it its concept subclassof equivalentto class objectproperty dataproperty individual
relates relation property type kind some_value""".split())


def _toks(text: str) -> list[str]:
    return [w for w in WORD.findall((text or "").lower()) if w not in BOILER]


def camel_snake_tokens(name: str) -> list[str]:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    s = s.replace("_", " ").replace("-", " ").lower()
    return [w for w in WORD.findall(s) if w not in BOILER]


def construct_terms(t: CatalogTemplate) -> set[str]:
    """Domain terms a construct mints: id tokens + slot names + verbal content."""
    terms: set[str] = set()
    terms.update(camel_snake_tokens(t.template_id))
    for slot in (t.slot_types or {}):
        terms.update(camel_snake_tokens(slot))
    # local-names of non-anchor CURIEs in the manchester body
    for m in re.finditer(r"\b(?:sdg|cco|bfo|obi|iao):([A-Za-z0-9_]+)", t.manchester_template or ""):
        terms.update(camel_snake_tokens(m.group(1)))
    terms.update(_toks(t.verbal_template))
    return {w for w in terms if len(w) >= 3}


def topic_term_signatures(reprs: list[str], top_k: int = 25) -> list[set[str]]:
    """Per-topic top-TF-IDF terms over the 200 topic representative docs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r"[a-z][a-z]{2,}", stop_words="english", max_df=0.4)
    X = vec.fit_transform([(r or "").lower() for r in reprs])
    vocab = np.array(vec.get_feature_names_out())
    sigs = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        top = vocab[np.argsort(row)[::-1][:top_k]]
        sigs.append({w for w in top if w not in BOILER})
    return sigs


_ENC = None


def _embed(terms: list[str]) -> np.ndarray | None:
    global _ENC
    if not terms:
        return None
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    v = _ENC.encode(terms, normalize_embeddings=True, convert_to_numpy=True).mean(0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def r1(vc: set[str], vt: set[str], use_embed: bool) -> float:
    if not vc or not vt:
        return 0.0
    jac = len(vc & vt) / len(vc | vt)
    if not use_embed:
        return jac
    ec, et = _embed(sorted(vc)), _embed(sorted(vt))
    cos = float(ec @ et) if ec is not None and et is not None else 0.0
    return max(jac, cos)


def boot_ci(x, rng, n=5000):
    x = np.asarray(x)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage-run", default="/raid/checkpoints/aegir-artifacts/coverage_v1/043d7dcc185245c8")
    ap.add_argument("--generated", default="/raid/checkpoints/aegir-artifacts/evidence/e6/08_generated_fixed.candidate.json")
    ap.add_argument("--embed", action="store_true",
                    help="ALSO take mpnet term-embedding cosine via max(jaccard,cosine) "
                         "— EXPERIMENTAL: high cosine floor on homogeneous domains (off≈0.78) "
                         "washes out discrimination; Jaccard (default) is the validated metric.")
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--out", default="/raid/checkpoints/aegir-artifacts/evidence/gcov/r1.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    use_embed = args.embed
    import pyarrow.parquet as pq

    rows = pq.read_table(Path(args.coverage_run) / "topic_coverage.parquet").to_pylist()
    by_topic = {r["topic_id"]: r for r in rows}
    reprs = [by_topic[i]["topic_repr_text"] if i in by_topic else "" for i in range(len(rows))]
    Vt = topic_term_signatures(reprs)
    print(f"topics={len(Vt)}  mean |V_t|={np.mean([len(s) for s in Vt]):.1f}")

    # ── Mechanics sanity: a topic's OWN terms must score ~max to itself, low elsewhere
    on, off = [], []
    for i, vt in enumerate(Vt):
        if not vt:
            continue
        j = int(rng.integers(0, len(Vt)))
        while j == i:
            j = int(rng.integers(0, len(Vt)))
        on.append(r1(vt, Vt[i], use_embed))
        off.append(r1(vt, Vt[j], use_embed))
    om, olo, ohi = boot_ci(on, rng); fm, flo, fhi = boot_ci(off, rng)
    print(f"[sanity: topic-own-terms]  on {om:.3f}[{olo:.3f},{ohi:.3f}]  off {fm:.3f}[{flo:.3f},{fhi:.3f}]")

    out = {"coverage_run": args.coverage_run, "n_topics": len(Vt),
           "sanity": {"on": om, "off": fm}}

    # ── Instrument validation on GENERATED constructs: on-topic vs shuffled null
    gen = json.load(open(args.generated))
    cons = [CatalogTemplate(**d) for d in gen["templates"]]
    con_on, con_off, detail = [], [], []
    for c in cons:
        tid = c.provenance.get("generated_from_topic")
        try:
            tid = int(tid)
        except Exception:
            continue
        if tid >= len(Vt):
            continue
        vc = construct_terms(c)
        son = r1(vc, Vt[tid], use_embed)
        j = int(rng.integers(0, len(Vt)))
        while j == tid:
            j = int(rng.integers(0, len(Vt)))
        soff = r1(vc, Vt[j], use_embed)
        con_on.append(son); con_off.append(soff)
        detail.append({"id": c.template_id, "topic": tid, "on": round(son, 3),
                       "off": round(soff, 3), "terms": sorted(vc)[:8]})
    if con_on:
        dm, dlo, dhi = boot_ci(np.array(con_on) - np.array(con_off), rng)
        gm, *_ = boot_ci(con_on, rng)
        clean = dlo > 0
        print(f"\n[GENERATED n={len(con_on)}]  on-topic R1 mean {gm:.3f}  "
              f"Δ(on−shuffled) {dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]  {'SPECIFIC' if clean else 'not-specific'}")
        for d in detail:
            print(f"    {d['id']:36} t{d['topic']:<4} on {d['on']:.3f} off {d['off']:.3f}  {d['terms']}")
        out["generated"] = {"n": len(con_on), "on_mean": gm, "delta": dm, "delta_ci": [dlo, dhi],
                            "specific_ci_clean": bool(clean), "detail": detail}
        out["R1_valid_instrument"] = bool(clean and (om - fm) > 0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nR1 valid instrument (sanity discriminates + generated on>shuffled CI-clean): "
          f"{out.get('R1_valid_instrument')}  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

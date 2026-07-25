"""probe_gittables_vocab — empirical-vocabulary evidence probe (seed-value provenance doctrine).

Scans a seeded GitTables sample for columns matching COL_RE, extracts observed value
sets, and banks per-value lineage in the gtvs shape (value@file:column) under
build/foreign/gittables/. The evidence artifact backs sdg:valueProvenance annotations
on empirical vocabulary members (values without observation stay FLAGGED authored —
provenance over exclusion, never silent fabrication). Edit TARGETS/COL_RE per
vocabulary; seed recorded for reproducibility."""
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path("/raid/datasets/gittables")
TARGETS = {"planned", "released", "inprogress", "in progress", "completed",
           "cancelled", "canceled", "onhold", "on hold"}
COL_RE = re.compile(r"status|state|stage|phase", re.I)
files = sorted(ROOT.glob("*.parquet"))
random.seed(44)
sample = random.sample(files, min(60000, len(files)))
value_obs = defaultdict(list)      # normalized value → [ "file:col", ... ]
vocab_sets = Counter()             # frozenset of observed normalized values per column
scanned = hits = 0
for f in sample:
    scanned += 1
    try:
        schema = pq.read_schema(f)
    except Exception:
        continue
    cols = [c for c in schema.names if COL_RE.search(c)]
    if not cols:
        continue
    try:
        t = pq.read_table(f, columns=cols)
    except Exception:
        continue
    for c in cols:
        vals = {str(v).strip() for v in t.column(c).to_pylist() if v is not None}
        vals = {v for v in vals if 2 <= len(v) <= 24}
        if not vals or len(vals) > 40:
            continue
        norm = {re.sub(r"[\s_-]+", "", v.lower()) for v in vals}
        overlap = norm & {re.sub(r"\s+", "", x) for x in TARGETS}
        if overlap:
            hits += 1
            lin = f"{f.stem}:{c}"
            for v in sorted(vals):
                nv = re.sub(r"[\s_-]+", "", v.lower())
                if nv in {re.sub(r"\s+", "", x) for x in TARGETS}:
                    value_obs[v].append(lin)
            vocab_sets[frozenset(sorted(norm))] += 1
    if scanned % 10000 == 0:
        print(f"  … {scanned}/{len(sample)} scanned · {hits} status-vocab columns", flush=True)

out = {
    "sampled_files": len(sample), "status_vocab_columns": hits,
    "value_observations": {v: {"n": len(l), "lineage": l[:12]} for v, l in
                           sorted(value_obs.items(), key=lambda kv: -len(kv[1]))},
    "top_cooccurring_vocabs": [dict(values=sorted(k), n=n)
                               for k, n in vocab_sets.most_common(10)],
}
Path("build/foreign/gittables/status_vocab_probe.json").parent.mkdir(parents=True, exist_ok=True)
Path("build/foreign/gittables/status_vocab_probe.json").write_text(json.dumps(out, indent=1))
print(f"DONE scanned={scanned} status-vocab columns={hits} distinct-observed={len(value_obs)}")
print("→ build/foreign/gittables/status_vocab_probe.json")

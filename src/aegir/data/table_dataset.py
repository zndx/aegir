"""
CTA/CPA benchmark dataset classes.

Supports the 6 benchmarks from REVEAL (some still stub):
  - SOTAB CTA (91 types) / CPA (176 types)
  - GitTables DBpedia (101 types) / Schema (53 types)
  - WikiTable CTA (255 types) / CPA (121 types)

Plus one locally-available benchmark implemented end-to-end for BDD:
  - ``gt-signals-dbpedia`` — 120 DBpedia types over 814 GitTables tables,
    sourced from ``~/local/src/cldr/signals/build/datasets/gittables/``.

Each dataset class loads table data, applies MMR context selection,
serializes tables with role markers, and produces training samples.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from aegir.data.serialization import serialize_table

# Default location of the cldr/signals GitTables drop — overridable per-instance.
GT_SIGNALS_DBPEDIA_DEFAULT_DIR = Path(
    "~/local/src/cldr/signals/build/datasets/gittables"
).expanduser()

# Number of classes per benchmark. Filled in lazily for tasks where the
# label vocabulary is discovered at load time (SOTAB, GitTables-full);
# the table below is the pinned vocabulary count for static / published
# benchmarks, used by ``train.py::make_model`` when the dataset hasn't
# been loaded yet (e.g. smoke-test path).
TASK_NUM_CLASSES = {
    "sotab": 91,           # Schema.org CTA
    "sotab-re": 176,       # Schema.org CPA
    "sotab-dbp": 101,      # DBpedia CTA (official: 101 types)
    "sotab-dbp-re": 53,    # DBpedia CPA
    "gt-semtab22-dbpedia-all": 101,
    "gt-semtab22-schema-property-all": 53,
    "turl": 255,
    "turl-re": 121,
    "gt-signals-dbpedia": 120,      # Atelier's 2,517-column sample
    "gittables-dbpedia": 835,       # Full GitTables 1M, DBpedia types (per Zenodo record)
    "gittables-schemaorg": 677,     # Full GitTables 1M, Schema.org types
}

# CPA (relation) tasks use multi-label classification
CPA_TASKS = {
    "sotab-re", "sotab-dbp-re",
    "gt-semtab22-schema-property-all", "turl-re",
}


class TableAnnotationDataset(Dataset):
    """Base dataset for column annotation tasks.

    Subclasses implement ``_load_tables()`` to provide table data in a
    standard format.

    Args:
        data_dir: Path to dataset directory.
        task: Task identifier (e.g. "sotab", "turl-re").
        split: Data split ("train", "val", "test").
        tokenizer: Tokenizer with encode() method.
        max_length: Maximum sequence length.
        max_context_cols: Maximum number of context columns to select.
        lambda_param: MMR lambda (relevance vs diversity trade-off).
    """

    def __init__(
        self,
        data_dir: str | Path,
        task: str,
        split: str = "train",
        tokenizer=None,
        max_length: int = 512,
        max_context_cols: int = 8,
        lambda_param: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.task = task
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_context_cols = max_context_cols
        self.lambda_param = lambda_param
        self.num_classes = TASK_NUM_CLASSES[task]
        self.is_multilabel = task in CPA_TASKS

        self.samples = self._load_and_prepare()

    def _load_and_prepare(self) -> list[dict]:
        """Load tables and prepare serialized samples.

        Returns:
            List of dicts with keys: token_ids, role_ids, cls_index, label.
        """
        tables = self._load_tables()
        samples = []

        for table in tables:
            target_col_idx = table["target_col_idx"]
            all_cols = table["columns"]
            col_names = table["column_names"]
            label = table["label"]

            target_col = all_cols[target_col_idx]
            target_name = col_names[target_col_idx]

            # Gather context candidates (all columns except target)
            context_indices = [i for i in range(len(all_cols)) if i != target_col_idx]

            if context_indices and self.max_context_cols > 0:
                # MMR context selection (lazy import to avoid requiring sentence-transformers at import time)
                from aegir.data.context_select import embed_columns, maximal_marginal_relevance

                all_embeddings = embed_columns(all_cols)
                target_emb = all_embeddings[target_col_idx]
                candidate_embs = all_embeddings[context_indices]

                selected = maximal_marginal_relevance(
                    target_emb,
                    candidate_embs,
                    top_k=self.max_context_cols,
                    lambda_param=self.lambda_param,
                )
                selected_indices = [context_indices[i] for i in selected]
            else:
                selected_indices = []

            context_cols = [all_cols[i] for i in selected_indices]
            context_names = [col_names[i] for i in selected_indices]

            serialized = serialize_table(
                target_col=target_col,
                target_col_name=target_name,
                context_cols=context_cols,
                context_col_names=context_names,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

            samples.append({
                "token_ids": serialized.token_ids,
                "role_ids": serialized.role_ids,
                "cls_index": serialized.cls_index,
                "label": label,
            })

        return samples

    def _load_tables(self) -> list[dict]:
        """Load raw table data. Override in subclasses.

        Returns:
            List of dicts with keys:
                columns: list of columns (each a list of cell strings)
                column_names: list of column name strings
                target_col_idx: index of the target column
                label: integer label (CTA) or list of labels (CPA)
        """
        raise NotImplementedError("Subclass must implement _load_tables()")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["token_ids"], dtype=torch.long),
            "role_ids": torch.tensor(sample["role_ids"], dtype=torch.long),
            "cls_index": torch.tensor(sample["cls_index"], dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


# ── SOTAB V2 loader ────────────────────────────────────────────────
#
# Actual layout produced by scripts/download_sotab.py when the zips unpack
# (derived from probing the uni-mannheim mirror, 2026-04):
#
#   {data_dir}/
#     Train/               — all CTA+CPA table JSONs, filename suffix ``_CTA`` or ``_CPA``
#     Validation/          — same (mixed)
#     Test/                — same
#     sotab_v2_cta_{split}_set.csv          — GT for CTA: table_name, column_index, label
#     sotab_v2_cta_training_set_small.csv   — smaller train variant for fast runs
#     sotab_v2_cta_{corner_cases,missing_values,format_heterogeneity,random}_test_set.csv
#     sotab_v2_cpa_{split}_set.csv          — GT for CPA (expected once CPA zip lands)
#
# Each table file is a *gzipped JSONL* (one JSON object per line, keys are
# stringified column indices, values are cell strings). 91 Schema.org CTA
# types + 176 CPA properties per the published benchmark; the live CSVs
# report 82 types in training (rest appear only in val/test splits).
#
# Aegir task ids map onto SOTAB axes:
#   sotab        = Schema.org CTA (single-label)
#   sotab-re     = Schema.org CPA (single-label per column pair)
#   sotab-dbp    = DBpedia CTA (not yet wired; same CSV pattern with ``dbpedia`` suffix)
#   sotab-dbp-re = DBpedia CPA

_SOTAB_SPLIT_MAP = {
    "train": "training", "training": "training",
    "val": "validation", "validation": "validation",
    "test": "test",
}
_SOTAB_SPLIT_DIRS = {
    "train": "Train", "training": "Train",
    "val": "Validation", "validation": "Validation",
    "test": "Test",
}


class _SotabDatasetBase(TableAnnotationDataset):
    """Common SOTAB loader. Subclasses pin the (task, label_space, annotation_kind) tuple.

    Subclass class-vars:
      _sotab_kind: "CTA" | "CPA"
      _sotab_label_space: "schemaorg" | "dbpedia"
    """

    _sotab_kind: str = "CTA"
    _sotab_label_space: str = "schemaorg"
    # Class-level memo for string→int label vocabs, keyed by
    # (kind, label_space, data_dir). Declared here so Pyright knows it
    # exists at class scope.
    _label_vocab_cache: dict[tuple, dict[str, int]] = {}

    def _load_tables(self):
        import pandas as pd

        gt_path = self._gt_csv_path(self.split)
        split_dir_name = _SOTAB_SPLIT_DIRS.get(self.split, "Train")
        tables_dir = self.data_dir / split_dir_name

        if not gt_path.exists():
            raise FileNotFoundError(
                f"SOTAB ground truth CSV not found at {gt_path}. "
                f"Run `just get-sotab` to materialize the dataset under {self.data_dir}."
            )
        if not tables_dir.is_dir():
            raise FileNotFoundError(
                f"SOTAB tables directory missing at {tables_dir}. "
                f"Expected {split_dir_name}/ with .json.gz files after `just get-sotab`."
            )

        gt = pd.read_csv(gt_path)
        # CTA CSV:  table_name, column_index, label
        # CPA CSV:  table_name, column_index, target_column_index, label
        # Normalize column access via .to_dict() per row (see below).

        label_vocab = self._build_label_vocab()

        tables: list[dict] = []
        # Cache loaded table cells so multiple labeled columns in the same
        # table don't re-parse the JSONL file.
        loaded_cells: dict[str, list[list[str]]] = {}

        grouped = gt.groupby("table_name")
        for table_name, group in grouped:
            json_path = tables_dir / str(table_name)
            if not json_path.exists():
                continue
            try:
                if str(table_name) in loaded_cells:
                    columns_cells = loaded_cells[str(table_name)]
                else:
                    columns_cells = self._load_table_cells(json_path)
                    loaded_cells[str(table_name)] = columns_cells
            except Exception:
                continue
            if not columns_cells:
                continue

            column_names = [f"col{i}" for i in range(len(columns_cells))]

            for _, row in group.iterrows():
                row_dict = row.to_dict()
                try:
                    col_idx = int(row_dict["column_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                if col_idx < 0 or col_idx >= len(columns_cells):
                    continue
                label_str = str(row_dict.get("label", "")).strip()
                if not label_str or label_str == "nan":
                    continue
                label_int = label_vocab.get(label_str)
                if label_int is None:
                    continue
                tables.append({
                    "columns": columns_cells,
                    "column_names": column_names,
                    "target_col_idx": col_idx,
                    "label": label_int,
                })

        return tables

    def _gt_csv_path(self, split: str) -> Path:
        """Resolve the ground-truth CSV for (kind, label_space, split).

        SOTAB V2 ships two naming conventions:
          Schema.org: sotab_v2_{kind}_{split_long}_set.csv
                      e.g. sotab_v2_cta_training_set.csv
          DBpedia:    sotab_{kind}_{split_short}_dbpedia.csv
                      e.g. sotab_cta_train_dbpedia.csv
        Schema.org uses ``training/validation/test``; DBpedia uses
        ``train/validation/test`` (no -ing).
        """
        kind_lower = self._sotab_kind.lower()
        split_long = _SOTAB_SPLIT_MAP.get(split, split)
        split_short = {"training": "train", "validation": "validation", "test": "test"}[split_long]
        if self._sotab_label_space == "schemaorg":
            return self.data_dir / f"sotab_v2_{kind_lower}_{split_long}_set.csv"
        return self.data_dir / f"sotab_{kind_lower}_{split_short}_{self._sotab_label_space}.csv"

    def _build_label_vocab(self) -> dict[str, int]:
        """Collect distinct labels across train/val/test GT CSVs into a stable str→int map."""
        import pandas as pd

        cache_key = (self._sotab_kind, self._sotab_label_space, str(self.data_dir))
        if cache_key in type(self)._label_vocab_cache:
            return type(self)._label_vocab_cache[cache_key]

        labels: set[str] = set()
        for split in ("training", "validation", "test"):
            gt_path = self._gt_csv_path(split)
            if not gt_path.exists():
                continue
            df = pd.read_csv(gt_path)
            if "label" in df.columns:
                for x in df["label"].dropna():
                    s = str(x).strip()
                    if s and s != "nan":
                        labels.add(s)

        vocab = {lbl: i for i, lbl in enumerate(sorted(labels))}
        type(self)._label_vocab_cache[cache_key] = vocab
        return vocab

    @staticmethod
    def _load_table_cells(json_path) -> list[list[str]]:
        """Parse a SOTAB JSONL table file and return list-of-columns.

        Each line is a JSON object ``{"0": cell00, "1": cell01, ..., "N-1": cell0N-1}``
        representing one row. Stringified column indices, arbitrary cell values.
        """
        import gzip
        import json

        opener = gzip.open if str(json_path).endswith(".gz") else open
        columns: dict[int, list[str]] = {}
        max_col = -1
        with opener(json_path, "rt", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                for k, v in row.items():
                    try:
                        ci = int(k)
                    except (TypeError, ValueError):
                        continue
                    columns.setdefault(ci, []).append("" if v is None else str(v))
                    if ci > max_col:
                        max_col = ci
        if max_col < 0:
            return []
        return [columns.get(i, []) for i in range(max_col + 1)]


class SotabCTADataset(_SotabDatasetBase):
    """SOTAB Schema.org Column Type Annotation (91 types, single-label)."""
    _sotab_kind = "CTA"
    _sotab_label_space = "schemaorg"

    def __init__(self, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = Path("/raid/datasets/sotab")
        super().__init__(data_dir, task="sotab", **kwargs)


class SotabCPADataset(_SotabDatasetBase):
    """SOTAB Schema.org Column Property Annotation (176 properties)."""
    _sotab_kind = "CPA"
    _sotab_label_space = "schemaorg"

    def __init__(self, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = Path("/raid/datasets/sotab")
        super().__init__(data_dir, task="sotab-re", **kwargs)


class SotabCTADbpediaDataset(_SotabDatasetBase):
    """SOTAB DBpedia Column Type Annotation."""
    _sotab_kind = "CTA"
    _sotab_label_space = "dbpedia"

    def __init__(self, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = Path("/raid/datasets/sotab")
        super().__init__(data_dir, task="sotab-dbp", **kwargs)


class SotabCPADbpediaDataset(_SotabDatasetBase):
    """SOTAB DBpedia Column Property Annotation."""
    _sotab_kind = "CPA"
    _sotab_label_space = "dbpedia"

    def __init__(self, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = Path("/raid/datasets/sotab")
        super().__init__(data_dir, task="sotab-dbp-re", **kwargs)


class GitTablesCTADataset(TableAnnotationDataset):
    """GitTables DBpedia Column Type Annotation (101 types)."""

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, task="gt-semtab22-dbpedia-all", **kwargs)

    def _load_tables(self):
        return []


class GitTablesCPADataset(TableAnnotationDataset):
    """GitTables Schema Property Annotation (53 types)."""

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, task="gt-semtab22-schema-property-all", **kwargs)

    def _load_tables(self):
        return []


class WikiTableCTADataset(TableAnnotationDataset):
    """WikiTable Column Type Annotation (255 types)."""

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, task="turl", **kwargs)

    def _load_tables(self):
        return []


class WikiTableCPADataset(TableAnnotationDataset):
    """WikiTable Column Property Annotation (121 types)."""

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, task="turl-re", **kwargs)

    def _load_tables(self):
        return []


def _build_gt_signals_label_vocab(gt_json_path: Path) -> dict[str, int]:
    """Build a deterministic label → int mapping from the GT JSON.

    Sorted alphabetically so the mapping is stable across runs without
    needing a cached vocab file. 120 labels total in the gt-signals drop.
    """
    with open(gt_json_path) as f:
        gt = json.load(f)["mappings"]
    return {label: i for i, label in enumerate(sorted(set(gt.values())))}


class GitTablesFullDataset(TableAnnotationDataset):
    """Full GitTables 1M from Zenodo record 6517052 (Hulsebos et al. 2023).

    Reads the ``*_licensed/*.parquet`` files produced by
    ``scripts/download_gittables.py``. Each parquet file = one table; the
    file's schema includes per-column semantic annotations as attrs. We
    materialize ``{columns, column_names, target_col_idx, label}`` per
    annotated column so every labeled column becomes one sample.

    Two label spaces: ``dbpedia`` (835 types) or ``schemaorg`` (677
    types). Select via subclass: ``GitTablesDbpediaDataset`` /
    ``GitTablesSchemaorgDataset``. Deterministic train/val/test split via
    BLAKE2b hash of the table filename.
    """

    _label_space: str = "dbpedia"  # overridden by subclasses
    _label_vocab_cache: dict[tuple, dict[str, int]] = {}

    def __init__(
        self,
        data_dir=None,
        *,
        split: str = "train",
        tokenizer=None,
        max_length: int = 512,
        max_context_cols: int = 8,
        lambda_param: float = 0.5,
        split_seed: int = 4649,
        max_tables: int | None = None,
    ):
        if data_dir is None:
            data_dir = Path("/raid/datasets/gittables")
        self._split_seed = split_seed
        self._max_tables = max_tables
        task = f"gittables-{self._label_space}"
        super().__init__(
            data_dir=data_dir, task=task, split=split, tokenizer=tokenizer,
            max_length=max_length, max_context_cols=max_context_cols,
            lambda_param=lambda_param,
        )

    def _load_tables(self):
        """Materialize labeled training samples from the extracted parquet tree.

        GitTables stores its annotations as a single JSON blob at the schema
        level under the ``gittables`` metadata key. The relevant subkey is
        ``{label_space}_semantic_column_types`` mapping column-name → {id,
        cleaned_label, domain, range, ...}. We use the ``id`` (DBpedia IRI
        or Schema.org URL) as the canonical label, build a deterministic
        vocab, and emit one training sample per labeled column.
        """
        import hashlib
        import json
        import pyarrow.parquet as pq

        def _split_of(table_id: str) -> str:
            h = int(hashlib.blake2b(
                f"{self._split_seed}:{table_id}".encode(), digest_size=4,
            ).hexdigest(), 16) % 100
            if h < 80: return "train"
            if h < 90: return "val"
            return "test"

        sem_key = f"{self._label_space}_semantic_column_types" if self._label_space == "dbpedia" else "schema_semantic_column_types"

        cache_key = (self._label_space, str(self.data_dir))
        if cache_key in type(self)._label_vocab_cache:
            label_vocab = type(self)._label_vocab_cache[cache_key]
        else:
            label_vocab = _load_or_discover_gittables_labels(
                self.data_dir, sem_key, self._label_space,
            )
            type(self)._label_vocab_cache[cache_key] = label_vocab

        parquets = sorted(self.data_dir.rglob("*.parquet"))
        if self._max_tables:
            parquets = parquets[: self._max_tables]

        tables: list[dict] = []
        for pq_path in parquets:
            table_id = pq_path.stem
            if _split_of(table_id) != self.split:
                continue
            try:
                pf = pq.read_table(pq_path)
                gt_meta_bytes = (pf.schema.metadata or {}).get(b"gittables")
                if not gt_meta_bytes:
                    continue
                gt_meta = json.loads(gt_meta_bytes.decode())
            except Exception:
                continue

            semantic_types = gt_meta.get(sem_key) or {}
            if not semantic_types:
                continue

            column_names = list(pf.column_names)
            name_to_idx = {n: i for i, n in enumerate(column_names)}
            # Cap cells per column to bound memory; 200 values is enough for MMR.
            columns_cells = [
                [str(v) for v in pf.column(i).to_pylist()[:200]]
                for i in range(len(column_names))
            ]

            for col_name, ann in semantic_types.items():
                if not isinstance(ann, dict):
                    continue
                iri = ann.get("id")
                # For columns annotated with a list of candidate IRIs, pick
                # the first one in the vocab (deterministic fallback).
                if isinstance(iri, list):
                    iri = next(
                        (x for x in iri if isinstance(x, str) and x in label_vocab),
                        None,
                    )
                if not isinstance(iri, str):
                    continue
                col_idx = name_to_idx.get(col_name)
                if col_idx is None:
                    continue
                label_int = label_vocab.get(iri)
                if label_int is None:
                    continue
                tables.append({
                    "columns": columns_cells,
                    "column_names": column_names,
                    "target_col_idx": col_idx,
                    "label": label_int,
                })

        return tables


def _load_or_discover_gittables_labels(
    data_dir: Path, sem_key: str, label_space: str,
) -> dict[str, int]:
    """Load the label vocab from disk cache, falling back to a full scan.

    Discovery over the full 562k-parquet corpus takes ~5 minutes — painful
    for every fresh process. The cache is a tiny JSON sidecar next to the
    data that invalidates only when you add or remove parquet files (not
    when you re-run). One file per label space so DBpedia and Schema.org
    caches don't collide.
    """
    import os

    if os.environ.get("AEGIR_GITTABLES_VOCAB_CACHE_DISABLE") == "1":
        return _discover_gittables_labels(data_dir, sem_key)

    cache_path = data_dir / f".aegir_vocab_{label_space}.json"
    # Cheap staleness signal: directory mtime. Walking 500k parquets just to
    # count them defeats the cache — a full rglob on GitTables takes as long
    # as the discovery itself. If the user wants to force a rescan, either
    # set AEGIR_GITTABLES_VOCAB_CACHE_DISABLE=1 or delete the sidecar.
    try:
        dir_mtime = int(data_dir.stat().st_mtime)
    except OSError:
        dir_mtime = 0

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("dir_mtime") == dir_mtime:
                return {lbl: int(i) for lbl, i in payload["vocab"].items()}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    vocab = _discover_gittables_labels(data_dir, sem_key)
    try:
        cache_path.write_text(
            json.dumps({
                "dir_mtime": dir_mtime,
                "label_space": label_space,
                "vocab": vocab,
            })
        )
    except OSError:
        pass
    return vocab


def _discover_gittables_labels(data_dir: Path, sem_key: str) -> dict[str, int]:
    """One-pass scan of all parquet schemas to build the label vocab.

    Parses the ``gittables`` JSON blob in each parquet's schema metadata
    and collects every distinct ``id`` value found under ``sem_key``
    (``dbpedia_semantic_column_types`` or ``schema_semantic_column_types``).
    """
    import json
    import pyarrow.parquet as pq

    labels: set[str] = set()
    for pq_path in data_dir.rglob("*.parquet"):
        try:
            schema = pq.read_schema(pq_path)
        except Exception:
            continue
        md = schema.metadata or {}
        raw = md.get(b"gittables")
        if not raw:
            continue
        try:
            gt = json.loads(raw.decode())
        except json.JSONDecodeError:
            continue
        types = gt.get(sem_key) or {}
        for ann in types.values():
            if not isinstance(ann, dict):
                continue
            iri = ann.get("id")
            # A small fraction of GitTables parquets carry a list of candidate
            # IRIs rather than a single canonical one. Treat each list entry
            # as a discoverable label; the main loader will have to resolve
            # the per-column pick separately.
            if isinstance(iri, str):
                labels.add(iri)
            elif isinstance(iri, list):
                labels.update(x for x in iri if isinstance(x, str))
    return {lbl: i for i, lbl in enumerate(sorted(labels))}


class GitTablesDbpediaDataset(GitTablesFullDataset):
    """Full GitTables 1M with DBpedia semantic type annotations."""
    _label_space = "dbpedia"


class GitTablesSchemaorgDataset(GitTablesFullDataset):
    """Full GitTables 1M with Schema.org semantic type annotations."""
    _label_space = "schemaorg"


class GitTablesSignalsDataset(TableAnnotationDataset):
    """GitTables DBpedia CTA using the cldr/signals local drop.

    Data layout (on disk):
        {data_dir}/gittables_columns.parquet   — 2517 rows, schema:
            source_table (str), column_name (str),
            column_type (STRING/INT/FLOAT/BOOL), sample_values (JSON str),
            sibling_columns (JSON str)
        {data_dir}/gittables_gt.json           — {"mappings": {"table.col": "type"}}

    Splits are derived deterministically from a hash of source_table:
    ``train`` = 80%, ``val`` = 10%, ``test`` = 10``%``.
    """

    def __init__(
        self,
        data_dir=None,
        *,
        split: str = "train",
        tokenizer=None,
        max_length: int = 512,
        max_context_cols: int = 8,
        lambda_param: float = 0.5,
        split_seed: int = 4649,
    ):
        if data_dir is None:
            data_dir = GT_SIGNALS_DBPEDIA_DEFAULT_DIR
        self._split_seed = split_seed
        super().__init__(
            data_dir=data_dir,
            task="gt-signals-dbpedia",
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            max_context_cols=max_context_cols,
            lambda_param=lambda_param,
        )

    def _load_tables(self):
        import pandas as pd  # lazy import; pandas is in aegir deps

        columns_path = self.data_dir / "gittables_columns.parquet"
        gt_path = self.data_dir / "gittables_gt.json"
        if not columns_path.exists() or not gt_path.exists():
            raise FileNotFoundError(
                f"gt-signals-dbpedia requires {columns_path} and {gt_path}. "
                f"See scripts/make_tiny_fixture.py to materialize a fixture subset."
            )

        label_vocab = _build_gt_signals_label_vocab(gt_path)
        with open(gt_path) as f:
            gt_mappings = json.load(f)["mappings"]

        df = pd.read_parquet(columns_path)
        # column_name is "" for the implicit id column — preserve it; GT keys use "table." in that case.
        df["column_name"] = df["column_name"].fillna("")

        # Deterministic split by source_table hash.
        def _split_of(table_id: str) -> str:
            import hashlib
            h = int(hashlib.blake2b(
                f"{self._split_seed}:{table_id}".encode(), digest_size=4,
            ).hexdigest(), 16) % 100
            if h < 80:
                return "train"
            if h < 90:
                return "val"
            return "test"

        tables: list[dict] = []
        for source_table, group in df.groupby("source_table"):
            if _split_of(str(source_table)) != self.split:
                continue
            group = group.reset_index(drop=True)

            # Cell values: JSON-decode the sample_values field.
            columns_cells = []
            for sv in group["sample_values"].tolist():
                try:
                    vals = json.loads(sv) if isinstance(sv, str) else list(sv)
                except json.JSONDecodeError:
                    vals = [str(sv)]
                columns_cells.append([str(v) for v in vals])

            column_names = [str(n) for n in group["column_name"].tolist()]

            # For every labeled column in this table, emit one sample.
            for col_idx, col_name in enumerate(column_names):
                key = f"{source_table}.{col_name}"
                if key not in gt_mappings:
                    continue
                label_int = label_vocab[gt_mappings[key]]
                tables.append({
                    "columns": columns_cells,
                    "column_names": column_names,
                    "target_col_idx": col_idx,
                    "label": label_int,
                })

        return tables

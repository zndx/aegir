"""
CTA/CPA benchmark dataset classes.

Supports the 6 benchmarks from REVEAL:
  - SOTAB CTA (91 types) / CPA (176 types)
  - GitTables DBpedia (101 types) / Schema (53 types)
  - WikiTable CTA (255 types) / CPA (121 types)

Each dataset class loads table data, applies MMR context selection,
serializes tables with role markers, and produces training samples.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from aegir.data.serialization import serialize_table

# Number of classes per benchmark
TASK_NUM_CLASSES = {
    "sotab": 91,
    "sotab-re": 176,
    "gt-semtab22-dbpedia-all": 101,
    "gt-semtab22-schema-property-all": 53,
    "turl": 255,
    "turl-re": 121,
}

# CPA (relation) tasks use multi-label classification
CPA_TASKS = {"sotab-re", "gt-semtab22-schema-property-all", "turl-re"}


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


class SotabCTADataset(TableAnnotationDataset):
    """SOTAB Column Type Annotation (91 types)."""

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, task="sotab", **kwargs)

    def _load_tables(self):
        # Placeholder: implement CSV/pickle loading for SOTAB format
        return []


class SotabCPADataset(TableAnnotationDataset):
    """SOTAB Column Property Annotation (176 types)."""

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, task="sotab-re", **kwargs)

    def _load_tables(self):
        return []


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

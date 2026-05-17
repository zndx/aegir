"""Phase 0.5 intermediate-pretrain dataset.

Reads parquet outputs from
``scripts/generate_synth_sql.py`` and
``scripts/generate_synth_counterfactual.py`` and emits LM-style byte
sequences for ``train_pretrain.py``::

    [BOS] <table_bytes> [SEP] <statement_or_text_bytes> [SEP] [LABEL_TOKEN] [EOS]

Where ``[LABEL_TOKEN]`` is :data:`TRUE_TOKEN_ID` or :data:`FALSE_TOKEN_ID`
(IDs 6, 7 in the reserved-special range, complementing the existing
PAD/CLS/SEP/BOS/EOS/CELL_BOUNDARY block).

This packaging lets the same byte-level next-token CE loss train against
binary truth labels — no classification head required. At eval time the
trainer can condition on the prefix up to the second ``SEP`` and inspect
``argmax(logits[TRUE_TOKEN_ID], logits[FALSE_TOKEN_ID])``.

Both source parquets share columns ``table_bytes`` and ``label``; their
second-text column differs (``statement_bytes`` vs ``text_bytes``). The
dataset transparently handles both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from aegir.data.tokenizer import (
    BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID,
    TRUE_TOKEN_ID, FALSE_TOKEN_ID,
)


class Phase05Dataset(Dataset):
    """LM-formatted Phase 0.5 corpus loader.

    Args:
        synth_sql_parquet: Optional path to synth-SQL parquet
            (``aegir_synth_sql_*.parquet``).
        counterfactual_parquet: Optional path to counterfactual parquet
            (``aegir_synth_counterfactual_*.parquet``).
        max_length: Truncate each example to this many byte IDs.
        max_samples: If >0, cap the total number of examples loaded
            (useful for smoke testing).
    """

    def __init__(
        self,
        synth_sql_parquet: str | Path | None = None,
        counterfactual_parquet: str | Path | None = None,
        max_length: int = 1024,
        max_samples: int = 0,
    ):
        if not synth_sql_parquet and not counterfactual_parquet:
            raise ValueError(
                "At least one of synth_sql_parquet or counterfactual_parquet "
                "must be provided."
            )
        self.max_length = max_length

        table_bytes_list: list[list[int]] = []
        second_bytes_list: list[list[int]] = []
        labels: list[bool] = []

        if synth_sql_parquet is not None:
            t = pq.read_table(str(synth_sql_parquet),
                              columns=["table_bytes", "statement_bytes", "label"])
            table_bytes_list.extend(t["table_bytes"].to_pylist())
            second_bytes_list.extend(t["statement_bytes"].to_pylist())
            labels.extend(t["label"].to_pylist())

        if counterfactual_parquet is not None:
            t = pq.read_table(str(counterfactual_parquet),
                              columns=["table_bytes", "text_bytes", "label"])
            table_bytes_list.extend(t["table_bytes"].to_pylist())
            second_bytes_list.extend(t["text_bytes"].to_pylist())
            labels.extend(t["label"].to_pylist())

        if max_samples and max_samples < len(labels):
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(labels))[:max_samples]
            table_bytes_list = [table_bytes_list[i] for i in idx]
            second_bytes_list = [second_bytes_list[i] for i in idx]
            labels = [bool(labels[i]) for i in idx]

        self.table_bytes = table_bytes_list
        self.second_bytes = second_bytes_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> torch.Tensor:
        table = self.table_bytes[idx]
        second = self.second_bytes[idx]
        label_token = TRUE_TOKEN_ID if self.labels[idx] else FALSE_TOKEN_ID

        # Compose: [BOS] table [SEP] second [SEP] [LABEL] [EOS]
        # Reserve 4 slots for BOS, two SEPs, label, EOS — distribute the
        # remaining budget between table and second.
        # Strip any leading specials in the inputs (the converter scripts
        # don't emit BOS but defend in depth).
        budget = self.max_length - 5  # BOS + SEP + SEP + LABEL + EOS = 5 tokens
        if budget <= 4:
            raise ValueError(f"max_length too small: {self.max_length}")

        # Reserve at most half the budget for the second segment so the
        # table always has presence even when statements are long.
        max_second = budget // 2
        second_trunc = second[:max_second]
        max_table = budget - len(second_trunc)
        table_trunc = table[:max_table]

        seq: list[int] = [BOS_TOKEN_ID]
        seq.extend(table_trunc)
        seq.append(SEP_TOKEN_ID)
        seq.extend(second_trunc)
        seq.append(SEP_TOKEN_ID)
        seq.append(label_token)
        seq.append(EOS_TOKEN_ID)
        return torch.tensor(seq, dtype=torch.long)

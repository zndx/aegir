"""
Table → token sequence serialization with role markers.

Converts tabular data (target column + context columns) into a flat token
sequence with role IDs for the Aegir column annotation model.

Format:
    [CLS] col_name [SEP] val1 [SEP] val2 ... [CLS] col_name [SEP] val1 ...

Target column gets role_id=0, context columns get role_id=1, 2, ...
"""

from dataclasses import dataclass


@dataclass
class SerializedTable:
    """A table serialized into token sequences with role markers.

    Attributes:
        token_ids: Flat list of token IDs.
        role_ids: Parallel list of role IDs (0=target, 1+=context).
        cls_index: Position of the target column's CLS token.
    """

    token_ids: list[int]
    role_ids: list[int]
    cls_index: int


# Special token IDs (RWKV tokenizer specific — these map to common tokens)
CLS_TOKEN_ID = 1  # Placeholder — should be configured per tokenizer
SEP_TOKEN_ID = 2  # Placeholder — should be configured per tokenizer


def serialize_table(
    target_col: list[str],
    target_col_name: str,
    context_cols: list[list[str]],
    context_col_names: list[str],
    tokenizer,
    max_length: int = 512,
    adaptive_length: bool = True,
) -> SerializedTable:
    """Serialize a table into a token sequence with role markers.

    Target column is placed first, followed by context columns.
    Token budget is distributed proportionally across columns when
    adaptive_length is True.

    Args:
        target_col: Cell values for the target column.
        target_col_name: Name of the target column.
        context_cols: List of context columns (each a list of cell values).
        context_col_names: Names of context columns.
        tokenizer: Tokenizer with encode() method.
        max_length: Maximum total token length.
        adaptive_length: Whether to distribute token budget proportionally.

    Returns:
        SerializedTable with token_ids, role_ids, and cls_index.
    """
    num_cols = 1 + len(context_cols)

    if adaptive_length:
        per_col_budget = max(1, (max_length - num_cols * 2) // num_cols)
    else:
        per_col_budget = max_length

    all_token_ids = []
    all_role_ids = []
    cls_index = -1

    def _serialize_column(col_name: str, col_values: list[str], role_id: int):
        """Serialize a single column within the token budget."""
        tokens = [CLS_TOKEN_ID]
        tokens.extend(tokenizer.encode(str(col_name))[:per_col_budget // 4])
        tokens.append(SEP_TOKEN_ID)

        remaining = per_col_budget - len(tokens)
        for val in col_values:
            val_tokens = tokenizer.encode(str(val))
            if len(val_tokens) + 1 > remaining:
                break
            tokens.extend(val_tokens)
            tokens.append(SEP_TOKEN_ID)
            remaining -= len(val_tokens) + 1

        return tokens

    # Target column (role=0)
    cls_index = len(all_token_ids)
    target_tokens = _serialize_column(target_col_name, target_col, role_id=0)
    all_token_ids.extend(target_tokens)
    all_role_ids.extend([0] * len(target_tokens))

    # Context columns (role=1, 2, ...)
    for i, (col, col_name) in enumerate(zip(context_cols, context_col_names)):
        if len(all_token_ids) >= max_length:
            break
        ctx_tokens = _serialize_column(col_name, col, role_id=i + 1)
        remaining = max_length - len(all_token_ids)
        ctx_tokens = ctx_tokens[:remaining]
        all_token_ids.extend(ctx_tokens)
        all_role_ids.extend([i + 1] * len(ctx_tokens))

    return SerializedTable(
        token_ids=all_token_ids,
        role_ids=all_role_ids,
        cls_index=cls_index,
    )

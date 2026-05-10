"""
Byte-level tokenizer for Aegir's table serialization pipeline.

Aegir is byte-native: vocab_size defaults to 65536 (for future expansion) but
the base alphabet is 256 bytes + a few special IDs used by serialization.py
(CLS and SEP markers that delimit column names and cell values).

The tokenizer is trivially deterministic — no training, no vocabulary files —
and is a drop-in shim satisfying the `encode(str) -> list[int]` contract that
``aegir.data.serialization.serialize_table`` expects.

Special IDs are placed in the 0..15 range where they collide with control
bytes, but since table text is either UTF-8 or printable ASCII, the overlap
is benign for CTA/CPA tasks. We remap bytes 0..15 → 16..31 to keep the
0..15 range reserved for specials.
"""

from __future__ import annotations


# Special token IDs — kept in sync with serialization.py (CLS_TOKEN_ID=1, SEP_TOKEN_ID=2)
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1
SEP_TOKEN_ID = 2
BOS_TOKEN_ID = 3
EOS_TOKEN_ID = 4
# 5..15 reserved for future special IDs
_BYTE_OFFSET = 16


class ByteTokenizer:
    """Deterministic byte-level tokenizer.

    IDs 0..15 are reserved for special tokens. Bytes 0..255 are mapped to
    IDs 16..271. This leaves plenty of headroom below vocab_size=65536.
    """

    vocab_size = _BYTE_OFFSET + 256  # 272
    pad_token_id = PAD_TOKEN_ID
    cls_token_id = CLS_TOKEN_ID
    sep_token_id = SEP_TOKEN_ID
    bos_token_id = BOS_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID

    def encode(self, text) -> list[int]:
        """Encode text → list of token IDs in the range [16, 272). Non-str inputs are coerced via ``str()`` because dataset rows often come as numpy scalars or pandas NAs."""
        s = text if isinstance(text, str) else str(text)
        return [b + _BYTE_OFFSET for b in s.encode("utf-8", errors="replace")]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to string. Skips special IDs (< 16)."""
        bs = bytes(i - _BYTE_OFFSET for i in ids if _BYTE_OFFSET <= i < _BYTE_OFFSET + 256)
        return bs.decode("utf-8", errors="replace")

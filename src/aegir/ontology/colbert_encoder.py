"""ColBERT late-interaction encoder for Qdrant multi-vector scoring (the domain-filter encoder).

Loads a ColBERT model (BERT backbone + 768→128 linear projection) and produces per-token 128-dim vectors
suitable for Qdrant's native MaxSim. Both sides of the domain filter — the SKOS concept nodes (the index)
and each streamed document (the query) — go through this same encoder; Qdrant handles the late-interaction
MaxSim at query time.

Adapted from the Atelier classify stack (same owner; the shared ColBERT/Qdrant pattern). Late-interaction
discriminates fine-grained domain/subdomain distinctions a single-vector cosine centroid cannot — which is
why it fits a hierarchical (domain→subdomain) input filter.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "colbert-ir/colbertv2.0"
_model_name: str = _DEFAULT_MODEL
_encoder: "_ColBERTEncoder | None" = None
_lock = threading.Lock()


class _ColBERTEncoder:
    """Thread-safe ColBERT encoder wrapping BERT + the checkpoint's linear projection."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from transformers import AutoModel, AutoTokenizer

        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._bert = AutoModel.from_pretrained(model_name)
        self._bert.eval()
        self._bert.to(device)
        self._projection = self._load_projection(model_name, device)  # (128, 768)
        self._dim = self._projection.shape[0]

    @staticmethod
    def _load_projection(model_name: str, device: str):
        """Load the ``linear.weight`` projection (768→128) from the model checkpoint."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        path = hf_hub_download(model_name, "model.safetensors")
        with safe_open(path, framework="pt", device=device) as f:
            return f.get_tensor("linear.weight")

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: "str | list[str]", *, batch_size: int = 16) -> "list[np.ndarray]":
        """Encode text(s) → per-token ColBERT vectors. Returns one ``(num_tokens, dim)`` array per input;
        [CLS]/[SEP]/[PAD] are stripped so only content tokens contribute to MaxSim."""
        import torch

        if isinstance(texts, str):
            texts = [texts]
        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True, max_length=512,
                                  return_tensors="pt", return_attention_mask=True,
                                  return_special_tokens_mask=True).to(self._device)
            with torch.no_grad():
                hidden = self._bert(**{k: v for k, v in enc.items()
                                       if k in ("input_ids", "attention_mask", "token_type_ids")}).last_hidden_state
                projected = torch.nn.functional.normalize(hidden @ self._projection.T, p=2, dim=-1)
            content = enc["attention_mask"] & (~enc["special_tokens_mask"].bool()).long()
            for i in range(projected.shape[0]):
                mask = content[i].bool()
                toks = projected[i][mask].cpu().numpy()
                out.append(toks if toks.shape[0] else projected[i][:1].cpu().numpy())
        return out

    def encode_single(self, text: str) -> "np.ndarray":
        return self.encode(text)[0]


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — cuda init can fail under the Nix lib mask; CPU is fine for the filter
        pass
    return "cpu"


def get_encoder() -> "_ColBERTEncoder":
    """Lazy-load the ColBERT encoder (thread-safe singleton)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is None:
            device = _get_device()
            logger.info("loading ColBERT encoder %s on %s", _model_name, device)
            _encoder = _ColBERTEncoder(_model_name, device=device)
    return _encoder


def warmup() -> dict:
    enc = get_encoder()
    r = enc.encode_single("probe")
    return {"dim": enc.dim, "tokens": int(r.shape[0]), "device": enc._device}

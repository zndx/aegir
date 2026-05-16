"""Constrained decoding for the (template_id, slot_fillers) emission.

A composition is a JSON list of ``CompositionEntry``-shaped dicts.
The policy must emit valid JSON in this shape or R_A clamps the
reward to zero. Constrained decoding eliminates that failure mode
during early training when the policy hasn't yet learned the
JSON envelope.

Two backends are supported (lazy imports — neither is a hard
dependency until P5 actually starts):

- ``outlines`` — regex / JSON-schema constrained sampling.
- ``lm-format-enforcer`` — token-level format enforcement via a
  TokenEnforcer that fits into a HuggingFace ``LogitsProcessor``.

The default backend is ``outlines`` because its JSON-schema
support is the most direct match for the ``CompositionEntry``
shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aegir.ontology.schema import Catalog


@dataclass
class DecodingConfig:
    backend: str = "outlines"  # "outlines" | "lmformatenforcer"
    max_new_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    n_compositions_per_prompt: int = 8  # group_size for GRPO


def composition_json_schema_lite(catalog: Catalog | None = None) -> dict:
    """Loose JSON Schema: just enforces the array-of-objects shape with
    ``template_id`` (any string) and ``slot_fillers`` (any object).

    Compared to :func:`composition_json_schema`, this skips the
    discriminated ``oneOf`` over all 540 catalog templates — that
    enumeration is what makes lmfe's per-token ``prefix_allowed_tokens_fn``
    expensive (each step traverses 540-branch state). The lite schema
    runs ~10-50× faster while still guaranteeing parseability.

    Trade-off: the policy can emit a ``template_id`` that isn't in the
    catalog. ``verify()``'s R_A hard gate rejects unknown template_ids
    (clamps reward to 0), so the lite schema doesn't change reward
    landscape correctness — it just allows the policy to "waste" some
    of its sample budget on hallucinated templates.

    For early-stage bootstrap (when the policy hasn't learned the
    catalog yet anyway) the lite schema is the better trade.
    """
    del catalog  # signature kept symmetric with composition_json_schema
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": 32,
        "items": {
            "type": "object",
            "properties": {
                "template_id": {"type": "string", "minLength": 1},
                "slot_fillers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["template_id", "slot_fillers"],
            "additionalProperties": False,
        },
    }


def composition_json_schema(catalog: Catalog) -> dict:
    """Build a JSON Schema constraining the policy to emit a list
    of ``CompositionEntry`` items whose ``template_id`` is in the
    catalog and whose ``slot_fillers`` match the declared slot
    names for that template.

    The returned schema uses an ``oneOf`` over per-template
    schemas so that for a chosen ``template_id`` only the slots
    declared by that template are required. This is a
    discriminated-union pattern that constrained-decoding
    backends parse efficiently.

    See :func:`composition_json_schema_lite` for a much faster loose
    schema that doesn't enumerate template_ids (relies on R_A to reject
    unknown ones post-hoc).
    """
    one_of: list[dict] = []
    for tmpl in catalog.templates:
        slot_props = {
            slot_name: {"type": "string", "minLength": 1}
            for slot_name in tmpl.slot_types.keys()
        }
        per_template_schema = {
            "type": "object",
            "properties": {
                "template_id": {"type": "string", "const": tmpl.template_id},
                "slot_fillers": {
                    "type": "object",
                    "properties": slot_props,
                    "required": list(tmpl.slot_types.keys()),
                    "additionalProperties": False,
                },
            },
            "required": ["template_id", "slot_fillers"],
            "additionalProperties": False,
        }
        one_of.append(per_template_schema)

    return {
        "type": "array",
        "minItems": 1,
        "maxItems": 32,
        "items": {"oneOf": one_of},
    }


def make_outlines_generator(model, tokenizer, schema: dict):
    """Outlines JSON-schema generator. Lazy import."""
    import outlines
    schema_str = json.dumps(schema)
    omodel = outlines.models.Transformers(model, tokenizer)
    return outlines.generate.json(omodel, schema_str)


def make_lmfe_prefix_allowed_tokens_fn(tokenizer, schema: dict):
    """Build the lm-format-enforcer ``prefix_allowed_tokens_fn`` compatible
    with ``transformers.GenerationMixin.generate``.

    Newer transformers releases moved ``PreTrainedTokenizerBase`` out of
    ``transformers.tokenization_utils`` (it now lives in
    ``tokenization_utils_base``). lm-format-enforcer 0.11.3 still imports
    from the old path, so we shim the attribute back on before importing
    its integration module.
    """
    import transformers.tokenization_utils as _ttu  # type: ignore[import-not-found]
    from transformers import PreTrainedTokenizerBase as _PTB

    if not hasattr(_ttu, "PreTrainedTokenizerBase"):
        _ttu.PreTrainedTokenizerBase = _PTB

    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )
    parser = JsonSchemaParser(schema)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


def make_lmfe_logits_processor(tokenizer, schema: dict):
    """Backwards-compatible alias for :func:`make_lmfe_prefix_allowed_tokens_fn`.

    The "logits processor" name is misleading — lm-format-enforcer's
    transformers integration actually returns a ``prefix_allowed_tokens_fn``,
    not a ``LogitsProcessor`` instance. New code should use the function
    with the accurate name.
    """
    return make_lmfe_prefix_allowed_tokens_fn(tokenizer, schema)


def make_xgrammar_logits_processor_factory(tokenizer, schema: dict,
                                            vocab_size: int | None = None):
    """Build a factory that yields fresh xgrammar ``LogitsProcessor``s.

    xgrammar pre-compiles a token-trie / pushdown automaton at schema-build
    time, then per-token enforcement is O(1) — independent of vocab size.
    On Qwen3.5-9B-Base (248K vocab) with our 540-branch composition schema,
    xgrammar compiles in ~0.7s and runs generation 5-15× faster than lmfe's
    per-token O(V) walk.

    Returns a zero-arg callable that produces a *new* ``LogitsProcessor``
    each call. This is required because xgrammar's processor maintains
    internal matcher state and can only be used for one ``generate()`` call.
    The compiled grammar object inside is reused (cheap to instantiate the
    processor wrapper on it).

    Usage::

        factory = make_xgrammar_logits_processor_factory(tokenizer, schema)
        for batch in dataset:
            model.generate(..., logits_processor=[factory()])
    """
    import json

    import torch
    import xgrammar as xgr  # type: ignore[import-not-found]
    from xgrammar.contrib.hf import LogitsProcessor as _XgrLogitsProcessor  # type: ignore[import-not-found]

    if vocab_size is None:
        # Qwen3.5 stores vocab_size on the (potentially multi-modal) config.
        # Fall back to tokenizer.vocab_size; for Qwen this matches the actual
        # full-vocab size used by the LM head.
        vocab_size = tokenizer.vocab_size

    tok_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size
    )
    compiler = xgr.GrammarCompiler(tok_info)
    compiled = compiler.compile_json_schema(json.dumps(schema))

    class _Patched(_XgrLogitsProcessor):
        """xgrammar 0.2.0 ships an ``hf.LogitsProcessor`` whose ``__call__``
        does ``sampled_token = input_ids[i][-1]`` and passes the resulting
        0-d ``torch.Tensor`` straight to ``GrammarMatcher.accept_token``.
        The tvm-ffi binding (also 0.2.0) is strictly typed and rejects the
        tensor with ``Mismatched type on argument #1 ... Expected 'int'
        but got 'ffi.Tensor'``. We override ``__call__`` and coerce the
        sampled-token tensor to a Python ``int`` before each
        ``accept_token`` call. Everything else (bitmask fill, mask apply)
        is delegated to the parent's logic.
        """

        def __call__(self, input_ids: torch.LongTensor,
                     scores: torch.FloatTensor) -> torch.FloatTensor:
            if len(self.matchers) == 0:
                self.batch_size = input_ids.shape[0]
                self.compiled_grammars = (
                    self.compiled_grammars
                    if len(self.compiled_grammars) > 1
                    else self.compiled_grammars * self.batch_size
                )
                assert len(self.compiled_grammars) == self.batch_size
                self.matchers = [
                    xgr.GrammarMatcher(self.compiled_grammars[i])
                    for i in range(self.batch_size)
                ]
                self.token_bitmask = xgr.allocate_token_bitmask(
                    self.batch_size, self.full_vocab_size,
                )

            if input_ids.shape[0] != self.batch_size:
                raise RuntimeError(
                    f"Expected input_ids.shape[0]=={self.batch_size}, "
                    f"got {input_ids.shape[0]}"
                )

            if not self.prefilled:
                self.prefilled = True
            else:
                for i in range(self.batch_size):
                    if not self.matchers[i].is_terminated():
                        sampled_token = int(input_ids[i][-1].item())
                        assert self.matchers[i].accept_token(sampled_token)

            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    self.matchers[i].fill_next_token_bitmask(
                        self.token_bitmask, i,
                    )

            device_type = scores.device.type
            if device_type != "cuda":
                scores = scores.to("cpu")
            xgr.apply_token_bitmask_inplace(
                scores, self.token_bitmask.to(scores.device),
            )
            if device_type != "cuda":
                scores = scores.to(device_type)

            return scores

    def _factory():
        return _Patched(compiled)

    return _factory


def parse_compositions(raw: str) -> list[dict[str, Any]]:
    """Parse the constrained-decode output back into a list of
    composition-entry dicts. Returns ``[]`` on malformed JSON
    (in which case R_A will clamp the reward to zero — this is
    the expected failure mode and we surface it as zero-length
    rather than raising)."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [
        e for e in parsed
        if isinstance(e, dict)
        and "template_id" in e
        and isinstance(e.get("slot_fillers"), dict)
    ]


def schema_dry_run(catalog: Catalog) -> dict:
    """Return summary of the constrained-decode schema without
    invoking outlines or lmformatenforcer."""
    schema = composition_json_schema(catalog)
    return {
        "backend_default": "outlines",
        "n_template_branches": len(schema["items"]["oneOf"]),
        "min_items": schema["minItems"],
        "max_items": schema["maxItems"],
        "schema_size_chars": len(json.dumps(schema)),
    }

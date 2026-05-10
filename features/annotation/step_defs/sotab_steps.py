"""Step definitions for features/annotation/sotab_cta.feature.

Tier-0 only: validates the loader against whatever the real SOTAB bundle
has placed on disk. Skips gracefully if /raid/datasets/sotab isn't
populated yet — BDD should never hard-depend on terabytes of external data.
"""

from __future__ import annotations

from pathlib import Path

from behave import given, then, when  # type: ignore[import]

from aegir.data.table_dataset import SotabCTADataset
from aegir.data.tokenizer import ByteTokenizer


SOTAB_DIR = Path("/raid/datasets/sotab")


@given("the SOTAB V2 Schema.org CTA data is present under /raid/datasets/sotab")
def step_given_sotab_present(context):
    if not (SOTAB_DIR / "sotab_v2_cta_validation_set.csv").exists():
        context.scenario.skip(
            f"SOTAB data not yet materialized at {SOTAB_DIR}. "
            "Run `just get-sotab` first."
        )
        return
    if not (SOTAB_DIR / "Validation").is_dir():
        context.scenario.skip(f"{SOTAB_DIR}/Validation not extracted yet")
        return
    context.sotab_dir = SOTAB_DIR


@when("I instantiate SotabCTADataset for the val split with max_context_cols=0")
def step_when_load_val(context):
    context.ds_val = SotabCTADataset(
        data_dir=context.sotab_dir,
        split="val",
        tokenizer=ByteTokenizer(),
        max_length=128,
        max_context_cols=0,
    )


@then("the dataset has at least {n:d} samples")
def step_then_nsamples(context, n):
    assert len(context.ds_val) >= n, f"only {len(context.ds_val)} samples"


@then("the discovered label vocab has at least {n:d} entries")
def step_then_vocab_size(context, n):
    vocab = context.ds_val._label_vocab_cache.get(
        ("CTA", "schemaorg", str(context.sotab_dir))
    )
    assert vocab is not None, "label vocab cache not populated"
    assert len(vocab) >= n, f"vocab has only {len(vocab)} entries"
    context.label_vocab = vocab


@then("every sample's label index is within the vocab size")
def step_then_labels_in_range(context):
    vocab_size = len(context.label_vocab)
    # Probe a random-ish set rather than scanning all 1769 samples.
    for i in (0, 1, len(context.ds_val) // 2, len(context.ds_val) - 1):
        lbl = context.ds_val[i]["label"].item()
        assert 0 <= lbl < vocab_size, f"sample {i} label {lbl} outside vocab size {vocab_size}"


@then("role_ids are all 0 (context columns disabled)")
def step_then_role_ids_zero(context):
    role_ids = context.ds_val[0]["role_ids"]
    unique = set(role_ids.unique().tolist())
    assert unique == {0}, f"expected role_ids={{0}} with max_context_cols=0, got {unique}"


@when("I load the SOTAB-CTA val and test splits")
def step_when_load_val_and_test(context):
    tok = ByteTokenizer()
    context.ds_val = SotabCTADataset(
        data_dir=context.sotab_dir, split="val",
        tokenizer=tok, max_length=128, max_context_cols=0,
    )
    context.ds_test = SotabCTADataset(
        data_dir=context.sotab_dir, split="test",
        tokenizer=tok, max_length=128, max_context_cols=0,
    )


@then("both splits index into the same label vocabulary instance")
def step_then_shared_vocab(context):
    key = ("CTA", "schemaorg", str(context.sotab_dir))
    v_val = type(context.ds_val)._label_vocab_cache.get(key)
    v_test = type(context.ds_test)._label_vocab_cache.get(key)
    assert v_val is not None and v_test is not None
    assert v_val is v_test, "val and test saw different vocab cache entries"

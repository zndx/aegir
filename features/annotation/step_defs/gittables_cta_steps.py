"""Step definitions for ``features/annotation/gittables_cta.feature``.

Orientation: these steps orchestrate *research questions* (does Aegir learn?
does loss decrease? do boundaries converge?) by delegating to
``features.annotation.step_defs.harness``. Assertions target observable
behavior, not internal shapes.
"""

from __future__ import annotations

from pathlib import Path

from behave import given, when, then  # type: ignore[import]

from aegir.data.table_dataset import GitTablesSignalsDataset, _build_gt_signals_label_vocab
from aegir.data.tokenizer import ByteTokenizer

from features.annotation.step_defs.harness import (
    build_tiny_model,
    evaluate_micro_macro,
    make_harness,
    train_n_steps,
)


# ── Tier-0 steps ───────────────────────────────────────────────


@given("a ByteTokenizer")
def step_given_tokenizer(context):
    context.tokenizer = ByteTokenizer()


@when('I encode "{text}" and decode the result')
def step_when_encode_decode(context, text):
    ids = context.tokenizer.encode(text)
    context.encoded_ids = ids
    context.decoded = context.tokenizer.decode(ids)


@then('the decoded string equals "{expected}"')
def step_then_decoded_equals(context, expected):
    assert context.decoded == expected, f"decoded={context.decoded!r} expected={expected!r}"


@then("every encoded id is between 16 and 272")
def step_then_ids_in_range(context):
    assert all(16 <= i < 272 for i in context.encoded_ids), (
        f"out-of-range id(s): {[i for i in context.encoded_ids if not (16 <= i < 272)]}"
    )


@given("the tiny gittables fixture is available")
def step_given_fixture(context):
    fixture = context.fixtures_dir / "gittables_tiny"
    parquet = fixture / "gittables_columns.parquet"
    gt = fixture / "gittables_gt.json"
    if not (parquet.exists() and gt.exists()):
        context.scenario.skip(
            f"tiny fixture missing at {fixture}; run `just sotab-fixture` first"
        )
        return
    context.fixture_dir = fixture
    context.fixture_parquet = parquet
    context.fixture_gt = gt


@when("I build the label vocabulary from the fixture ground-truth JSON")
def step_when_build_vocab(context):
    context.label_vocab = _build_gt_signals_label_vocab(context.fixture_gt)


@then("the vocabulary maps at least {n:d} distinct DBpedia types")
def step_then_vocab_size(context, n):
    assert len(context.label_vocab) >= n, (
        f"label vocab has only {len(context.label_vocab)} entries, expected ≥ {n}"
    )


@then("the mapping is alphabetically ordered")
def step_then_vocab_sorted(context):
    labels = list(context.label_vocab.keys())
    assert labels == sorted(labels), "label vocabulary is not alphabetically sorted"


@when("I load the dataset for each split")
def step_when_load_splits(context):
    context.split_samples = {}
    for split in ("train", "val", "test"):
        ds = GitTablesSignalsDataset(
            data_dir=context.fixture_dir, split=split,
            tokenizer=ByteTokenizer(), max_length=128, max_context_cols=0,
        )
        context.split_samples[split] = [ds[i]["label"].item() for i in range(len(ds))]


@then("the train split has at least {n:d} samples")
def step_then_train_size(context, n):
    assert len(context.split_samples["train"]) >= n, (
        f"train has {len(context.split_samples['train'])}, expected ≥ {n}"
    )


@then("the val and test splits are non-empty")
def step_then_val_test_nonempty(context):
    for split in ("val", "test"):
        assert len(context.split_samples[split]) > 0, f"{split} split is empty"


@then("the same random seed produces identical splits on re-load")
def step_then_deterministic_splits(context):
    for split in ("train", "val", "test"):
        ds = GitTablesSignalsDataset(
            data_dir=context.fixture_dir, split=split,
            tokenizer=ByteTokenizer(), max_length=128, max_context_cols=0,
        )
        labels_again = [ds[i]["label"].item() for i in range(len(ds))]
        assert labels_again == context.split_samples[split], (
            f"{split} split changed on re-load (non-deterministic)"
        )


# ── Tier-1 steps ───────────────────────────────────────────────


@given("a tiny Aegir model configured for the 120-class gt-signals task")
def step_given_tiny_model(context):
    context.model = build_tiny_model(task="gt-signals-dbpedia", device=context.device)


@given("a batch of {batch_size:d} samples from the tiny gittables fixture")
def step_given_batch(context, batch_size):
    from torch.utils.data import DataLoader
    from train import collate_fn

    ds = GitTablesSignalsDataset(
        data_dir=context.fixture_dir, split="train",
        tokenizer=ByteTokenizer(), max_length=128, max_context_cols=4,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    context.sample_batch = {k: v.to(context.device) for k, v in next(iter(loader)).items()}
    context.batch_size = batch_size


@when("I run a forward and backward pass on a new batch")
def step_when_fwd_bwd_from_harness(context):
    """Grab a fresh batch from the harness's train loader and reuse the main
    fwd/bwd step. Used by scenarios that train first then want to observe
    gradients post-training."""
    batch = next(iter(context.harness.train_loader))
    context.sample_batch = {k: v.to(context.harness.device) for k, v in batch.items()}
    context.model = context.harness.model
    context.batch_size = context.sample_batch["input_ids"].shape[0]
    step_when_fwd_bwd(context)


@when("I run a forward and backward pass on the batch")
def step_when_fwd_bwd(context):
    import torch
    from aegir.utils.train import load_balancing_loss

    for p in context.model.parameters():
        p.grad = None
    b = context.sample_batch
    output = context.model(
        b["input_ids"], role_ids=b["role_ids"],
        cls_indexes=b["cls_indexes"], mask=b["mask"],
    )
    task_loss = torch.nn.functional.cross_entropy(output.logits, b["labels"])
    # Mirror train.py's full training objective so routing Q/K projections
    # receive gradients from the load-balancing term, not just via STE through
    # the residual branch. Without this, Q/K gradients can legitimately be zero
    # on a single batch.
    lb_loss = torch.tensor(0.0, device=task_loss.device)
    if output.bpred_output:
        for bp in output.bpred_output:
            lb_loss = lb_loss + load_balancing_loss(bp, N=2.0)
        lb_loss = lb_loss / len(output.bpred_output)
    loss = task_loss + 0.1 * lb_loss
    loss.backward()
    context.fwd_output = output
    context.fwd_loss = loss


@then("the logits have shape (batch_size, {num_classes:d})")
def step_then_logits_shape(context, num_classes):
    shape = tuple(context.fwd_output.logits.shape)
    assert shape == (context.batch_size, num_classes), (
        f"logits shape {shape} ≠ ({context.batch_size}, {num_classes})"
    )


@then("at least the outermost routing module receives gradients")
def step_then_outer_routing_grads(context):
    # Why only the outermost at init time: the routing Q/K projections are
    # identity-initialized, which makes cos_sim(q, k_{t-1}) = 1 at every
    # position for inner stages whose inputs haven't been perturbed by any
    # prior routing/chunking. The identity-init cos_sim=1 has an effectively
    # zero gradient w.r.t. Q/K. Outer-stage routing sees real input variation
    # and trains normally. After a few steps of training, inner-stage Q/K
    # start receiving gradients; the convergence scenario (@slow) asserts that.
    outer_routing = [
        (n, p) for n, p in context.model.named_parameters()
        if "routing_module" in n and "main_network.routing_module" not in n
    ]
    assert outer_routing, "no outermost routing_module parameters found"
    dead = [n for n, p in outer_routing if p.grad is None or p.grad.abs().sum().item() == 0.0]
    assert not dead, f"outer-stage routing params with zero gradient: {dead}"


@then("the loss is finite and positive")
def step_then_loss_finite_positive(context):
    import math
    v = context.fwd_loss.item()
    assert math.isfinite(v) and v > 0, f"loss is {v} (not finite+positive)"


@given("the tiny gittables fixture loaded with batch size {batch_size:d}")
def step_given_harness_tiny(context, batch_size):
    context.harness = make_harness(
        data_dir=context.fixture_dir,
        device=context.device,
        batch_size=batch_size,
        max_context_cols=4,
    )


@given("the full gt-signals-dbpedia dataset loaded with batch size {batch_size:d}")
def step_given_harness_full(context, batch_size):
    context.harness = make_harness(
        data_dir=None,  # default path: ~/local/src/cldr/signals/…
        device=context.device,
        batch_size=batch_size,
        max_context_cols=4,
    )


@when("I train for {n:d} steps with AdamW lr={lr:g}")
def step_when_train_steps(context, n, lr):
    context.harness.args.lr = lr
    for group in context.harness.optimizer.param_groups:
        group["lr"] = lr
    context.train_summary = train_n_steps(context.harness, n)


@then("the mean loss over the last {k:d} steps is at least {pct:g}% below the first {k2:d}")
def step_then_loss_decrease(context, k, pct, k2):
    # k and k2 should match in the scenario phrasing
    summary = context.train_summary
    first = summary["first_mean"]
    last = summary["last_mean"]
    assert first > 0, f"first_mean={first}"
    improvement = (first - last) / first * 100
    assert improvement >= pct, (
        f"loss decrease is {improvement:.1f}% (first={first:.4f} last={last:.4f}), "
        f"expected ≥ {pct}%. history[:5]={context.harness.loss_history[:5]}, "
        f"history[-5:]={context.harness.loss_history[-5:]}"
    )


@then("the boundary diagnostics show non-zero selection rate at every routing stage")
def step_then_boundary_nonzero(context):
    b = context.train_summary["boundary"]
    assert b, "no boundary diagnostics recorded (no routing stages?)"
    f_keys = [k for k in b if k.endswith("_mean_F")]
    assert f_keys, f"no mean_F keys in {list(b)}"
    zeros = [k for k in f_keys if b[k] <= 0.0]
    assert not zeros, f"stages with zero selection rate: {zeros} (diagnostics: {b})"


@when("I train for {n:d} epochs and evaluate on the val split")
def step_when_train_epochs_eval(context, n):
    steps_per_epoch = max(1, len(context.harness.train_loader))
    train_n_steps(context.harness, n * steps_per_epoch)
    context.val_metrics = evaluate_micro_macro(context.harness)


@then("dev micro-F1 exceeds {threshold:g}")
def step_then_micro_threshold(context, threshold):
    micro = context.val_metrics["micro_f1"]
    assert micro > threshold, (
        f"micro-F1={micro:.4f} ≤ {threshold}. full metrics={context.val_metrics}"
    )


@then("dev macro-F1 is reported alongside micro-F1")
def step_then_macro_reported(context):
    assert "macro_f1" in context.val_metrics, "macro_f1 not in val_metrics"
    assert isinstance(context.val_metrics["macro_f1"], float)

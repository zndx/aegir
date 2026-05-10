"""Step definitions for ``features/annotation/context_selection.feature``.

Tier-0 only: the MMR ablation against real training (MMR-vs-random F1 delta)
was drafted in the plan but is deferred — running two full trainings with
identical seeds is tier-1 @slow territory and will be added once the tier-1
convergence scenarios are stable. For now we validate the selector's core
properties (diversity, uniqueness, size) on synthetic embeddings.
"""

from __future__ import annotations

import torch
from behave import given, when, then  # type: ignore[import]

from aegir.data.context_select import maximal_marginal_relevance


@given("candidate column embeddings with {k:d} near-duplicates of the first")
def step_given_with_dupes(context, k):
    torch.manual_seed(4649)
    # Use an orthogonal basis so "diverse" candidates have near-zero cosine similarity
    # to the dup group. Without this, random 16-dim vectors occasionally happen to
    # point close to the dup direction and MMR's noise dominates.
    d = 16
    basis = torch.eye(d)[:k + 6]  # k+1 dups + 5 diverse, all mutually orthogonal
    dup_base = basis[0]
    candidates = [dup_base + 1e-3 * torch.randn(d) for _ in range(k + 1)]
    for i in range(5):
        candidates.append(basis[k + 1 + i] + 1e-3 * torch.randn(d))
    context.candidates = torch.stack(candidates)
    context.dup_group = set(range(k + 1))


@given("a target embedding near the first candidate")
def step_given_target_near_first(context):
    context.target = context.candidates[0] + 0.01 * torch.randn(16)


@when("I run MMR with top_k={top_k:d} and lambda={lam:g}")
def step_when_mmr(context, top_k, lam):
    context.selected = maximal_marginal_relevance(
        context.target.unsqueeze(0),
        context.candidates,
        top_k=top_k, lambda_param=lam,
    )


@then("the selected set contains at most {m:d} of the near-duplicate group")
def step_then_dedupe(context, m):
    overlap = [i for i in context.selected if i in context.dup_group]
    assert len(overlap) <= m, (
        f"MMR selected {len(overlap)} from dup group (size {len(context.dup_group)}): {overlap}"
    )


@given("{n:d} random candidate embeddings of dim {d:d}")
def step_given_random_candidates(context, n, d):
    torch.manual_seed(4649)
    context.candidates = torch.randn(n, d)
    context.target = torch.randn(d)


@then("exactly {k:d} indices are returned")
def step_then_k_returned(context, k):
    assert len(context.selected) == k, f"got {len(context.selected)} indices, expected {k}"


@then("all returned indices are distinct")
def step_then_distinct(context):
    assert len(set(context.selected)) == len(context.selected), (
        f"duplicate indices in {context.selected}"
    )

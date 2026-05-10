Feature: MMR context selection on wide tables
  As a data engineer processing tables with many irrelevant columns, I need
  MMR context selection to pick the most informative subset for the target
  column so that training sees a denser signal than random concatenation.

  @tier-0
  Scenario: MMR picks diverse columns from a wide synthetic candidate pool
    Given candidate column embeddings with 3 near-duplicates of the first
    And a target embedding near the first candidate
    When I run MMR with top_k=5 and lambda=0.3
    Then the selected set contains at most 2 of the near-duplicate group
    # Lambda=0.3 weights diversity (0.7) more than relevance (0.3). We expect 1 duplicate
    # to be picked first (maximum relevance), then MMR should prefer diverse candidates
    # over the remaining near-duplicates. Allow ≤2 to tolerate stochasticity from
    # random-projection embeddings in the synthetic pool.

  @tier-0
  Scenario: MMR returns up to top_k unique indices
    Given 10 random candidate embeddings of dim 16
    When I run MMR with top_k=6 and lambda=0.7
    Then exactly 6 indices are returned
    And all returned indices are distinct

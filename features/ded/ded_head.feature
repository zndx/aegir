Feature: AegirForDED head + supervised contrastive loss
  Scaffolds cross-table Data Element Discovery (M2). Locks in the three
  invariants the training loop will depend on: per-column L2-normalized
  embeddings, padded columns zeroed and flagged in col_mask, and SupCon
  loss finite-and-positive when at least one cluster has ≥2 members.

  @tier-0
  Scenario: DED head produces unit-norm embeddings for valid columns
    Given an AegirForDED head built from a tiny config with proj_dim=64
    And a synthetic batch with 4 samples, seq_len=48, 3 valid columns + 1 padding each
    When I run the DED head forward
    Then every valid-column embedding has L2 norm close to 1
    And every padded-column embedding has L2 norm 0
    And col_mask flags exactly 3 valid columns per sample

  @tier-0
  Scenario: supcon_loss is positive and finite for non-trivial clustering
    Given column embeddings for 4 samples with 3 valid columns each
    And cluster_ids forming 2 clusters with at least 2 members each
    When I compute supcon_loss
    Then the loss is finite
    And the loss is positive

  @tier-0
  Scenario: supcon_loss returns zero when there are no positive pairs
    Given column embeddings for 4 samples with 3 valid columns each
    And every column assigned a unique cluster_id
    When I compute supcon_loss
    Then the loss is exactly 0

  @tier-0
  Scenario: bcubed_f1 of a perfect prediction is 1.0
    Given ground-truth cluster_ids with 3 clusters of size 2
    When I compute bcubed_f1 with predictions matching the ground truth
    Then the F1 score is 1.0

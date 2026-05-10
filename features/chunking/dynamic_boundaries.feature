Feature: Dynamic chunking learns content-dependent boundaries
  As a researcher I need to confirm that the routing module learns non-trivial
  boundaries — not uniform chunking or collapsed-to-zero — on the target task.
  This feature probes the H-Net inheritance: routing gradients, load-balancing
  convergence, and per-stage statistics on real GitTables batches.

  @tier-0
  Scenario: boundary_diagnostics returns empty for non-chunked outputs
    When I call boundary_diagnostics with an empty bpred_output list
    Then the result is an empty dict

  @tier-0
  Scenario: boundary_diagnostics computes per-stage metrics
    Given a synthetic bpred_output with 2 stages at selection rates 0.5 and 0.25
    When I call boundary_diagnostics with target_N 2.0
    Then stage0 mean_F is close to 0.5
    And stage1 mean_F is close to 0.25
    And stage0 abs_ratio_err is close to 0.0
    And each stage's achieved_N matches 1 over its mean_F

  @tier-1 @gpu
  Scenario: STE pass-through produces finite gradients on boundary probabilities
    Given a tiny Aegir model configured for the 120-class gt-signals task
    And a batch of 8 samples from the tiny gittables fixture
    When I run a forward and backward pass on the batch
    Then the STE pass-through produces finite gradients on boundary probabilities

  @tier-1 @gpu @slow
  Scenario: Every stage's routing receives gradients after brief training
    Given a tiny Aegir model configured for the 120-class gt-signals task
    And the tiny gittables fixture loaded with batch size 4
    When I train for 50 steps with AdamW lr=5e-4
    And I run a forward and backward pass on a new batch
    Then every stage's RoutingModule has non-zero gradient magnitude
    # Identity-init makes inner-stage Q/K gradients exactly zero at step 0.
    # After ~50 steps of training, outer routing perturbs inner inputs enough
    # that cos_sim ≠ 1 and Q/K gradients become non-zero.

  @tier-1 @gpu @slow
  Scenario: Load balancing pulls selection rate toward 1/N on tiny fixture
    Given a tiny Aegir model configured for the 120-class gt-signals task
    And the tiny gittables fixture loaded with batch size 4
    When I train for 100 steps with AdamW lr=5e-4
    Then at every routing stage the achieved N is finite
    And the outermost stage's mean_F drifts from the untrained baseline

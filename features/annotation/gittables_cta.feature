Feature: Aegir learns column types from real GitTables tables
  As a research engineer I need end-to-end evidence that Aegir produces
  non-trivial predictions on real labeled tables, not just synthetic tensors.
  The gt-signals-dbpedia task has 120 DBpedia label types over 814 tables
  from the local cldr/signals drop — no downloads, ~250 KB total.

  @tier-0
  Scenario: Tokenizer round-trips UTF-8 through byte IDs
    Given a ByteTokenizer
    When I encode "hello — UTF-8 ✓" and decode the result
    Then the decoded string equals "hello — UTF-8 ✓"
    And every encoded id is between 16 and 272

  @tier-0
  Scenario: Label vocabulary is deterministic and covers the benchmark
    Given the tiny gittables fixture is available
    When I build the label vocabulary from the fixture ground-truth JSON
    Then the vocabulary maps at least 20 distinct DBpedia types
    And the mapping is alphabetically ordered

  @tier-0
  Scenario: GitTables dataset splits the tiny fixture deterministically
    Given the tiny gittables fixture is available
    When I load the dataset for each split
    Then the train split has at least 20 samples
    And the val and test splits are non-empty
    And the same random seed produces identical splits on re-load

  @tier-1 @gpu
  Scenario: Aegir produces valid logits on a real GitTables batch
    Given a tiny Aegir model configured for the 120-class gt-signals task
    And a batch of 8 samples from the tiny gittables fixture
    When I run a forward and backward pass on the batch
    Then the logits have shape (batch_size, 120)
    And at least the outermost routing module receives gradients
    And the loss is finite and positive

  @tier-1 @gpu @slow
  Scenario: Training loss decreases over 100 steps on GitTables
    Given a tiny Aegir model configured for the 120-class gt-signals task
    And the tiny gittables fixture loaded with batch size 4
    When I train for 100 steps with AdamW lr=5e-4
    Then the mean loss over the last 10 steps is at least 10% below the first 10
    And the boundary diagnostics show non-zero selection rate at every routing stage

  @tier-1 @gpu @slow
  Scenario: Aegir beats random baseline on GitTables dev after brief training
    Given a tiny Aegir model configured for the 120-class gt-signals task
    And the full gt-signals-dbpedia dataset loaded with batch size 16
    When I train for 3 epochs and evaluate on the val split
    Then dev micro-F1 exceeds 0.04
    # Random baseline is 1/120 ≈ 0.008; 0.04 is a ~5× improvement over chance.
    And dev macro-F1 is reported alongside micro-F1

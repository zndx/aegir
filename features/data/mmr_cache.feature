Feature: MMR embedding cache skips re-encoding
  Dataset __init__ calls embed_columns once per labeled column to drive MMR
  context selection. On GitTables-scale corpora that's the dominant startup
  cost. A disk-backed blake2b cache avoids re-encoding across runs and
  across dataloader workers within a run.

  @tier-0
  Scenario: cache hit returns identical embeddings within fp16 precision
    Given a fresh MMR cache dir under /tmp
    And three toy columns
    When I embed them twice in the same process
    Then the second call returns the same embeddings within 1e-3
    And the second call is at least 100x faster than the first

  @tier-0
  Scenario: cache can be disabled via env var
    Given the env var AEGIR_MMR_CACHE_DISABLE=1
    And four toy columns
    When I embed four toy columns twice in the same process
    Then both calls trigger the model encoder

Feature: GitTables label vocab cache skips re-discovery
  Discovering the DBpedia / Schema.org label vocab requires walking every
  parquet's schema metadata — 5+ minutes on the full 562k-file GitTables
  corpus. A JSON sidecar next to the data caches the vocab, keyed on
  directory mtime. Invalidates automatically when new parquets are added
  (and mtime bumps); can be force-rebuilt via env var.

  @tier-0
  Scenario: cache hit returns the same vocab without re-scanning
    Given a synthetic GitTables dir with 5 parquets and DBpedia annotations
    When I load the vocab twice in the same process
    Then the second call returns an identical vocab
    And a .aegir_vocab_dbpedia.json sidecar exists next to the data

  @tier-0
  Scenario: cache disabled via env var forces re-scan
    Given a synthetic GitTables dir with 5 parquets and DBpedia annotations
    And the env var AEGIR_GITTABLES_VOCAB_CACHE_DISABLE=1
    When I load the vocab once
    Then no .aegir_vocab_dbpedia.json sidecar is written

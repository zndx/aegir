Feature: Training runs emit inspectable sidecar artifacts
  As a research engineer I need every training run to leave a self-contained
  artifact on disk (metadata + metrics + Bokeh plots) so the leaderboard UI
  can render it offline, without a tracking daemon.

  @tier-0
  Scenario: run_id format is stable and sortable
    Given a fresh RunArtifacts with task "gt-signals-dbpedia" and model_size "tiny"
    Then the run_id contains the UTC date as a prefix
    And the run_id contains "tiny"
    And the run_id contains "gt-signals-dbpedia"

  @tier-0
  Scenario: sidecar layout is exactly what the gateway consumes
    Given a fresh RunArtifacts with task "gt-signals-dbpedia" and model_size "tiny"
    When I record 3 synthetic epochs and finalize
    Then the run directory contains "metadata.json"
    And the run directory contains "metrics.json"
    And the run directory has 4 plot files
    And each plot file parses as a Bokeh JSON document with a "doc" key

  @tier-0
  Scenario: metrics are tolerant of missing fields
    Given a fresh RunArtifacts with task "gt-signals-dbpedia" and model_size "tiny"
    When I record an epoch with only loss and skip F1
    Then the resulting metrics.json has one epoch with null F1 values

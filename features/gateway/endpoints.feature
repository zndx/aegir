Feature: Gateway exposes leaderboard and run-artifact endpoints
  As a UI developer I need stable HTTP contracts for the leaderboard,
  run details, and plot JSON so the React shell can consume them without
  coordination with the backend each release.

  @tier-0
  Scenario: health endpoint reports config-resolved versions and reachability
    Given a gateway app bound to a temp runs dir with 0 runs
    When I GET "/api/health"
    Then the response status is 200
    And the response has field "ok" equal to true
    And the response has field "runs_count" equal to 0

  @tier-0
  Scenario: leaderboard aggregates existing sidecars
    Given a gateway app bound to a temp runs dir with 2 synthetic runs
    When I GET "/api/leaderboard"
    Then the response status is 200
    And the response has field "count" equal to 2
    And each leaderboard row has a non-null run_id and task

  @tier-0
  Scenario: run detail returns metadata + metrics + plot manifest
    Given a gateway app bound to a temp runs dir with 2 synthetic runs
    When I GET the detail endpoint for the most recent run
    Then the response has a "plots" list containing "loss" and "f1"

  @tier-0
  Scenario: plot endpoint returns a Bokeh JSON document
    Given a gateway app bound to a temp runs dir with 2 synthetic runs
    When I GET the loss plot for the most recent run
    Then the response is a Bokeh JSON document

  @tier-0
  Scenario: plot endpoint rejects directory traversal attempts
    Given a gateway app bound to a temp runs dir with 2 synthetic runs
    When I GET a plot named "../../etc/passwd"
    Then the response status is 404

  @tier-0
  Scenario: ontology endpoint returns DBpedia types when GT JSON is present
    Given a gateway app configured to the cldr/signals gittables dir
    When I GET "/api/ontology/dbpedia-types"
    Then the response has field "exists" equal to true
    And the response has at least 100 types

  @tier-0
  Scenario: stats endpoint returns the 4-card Landing payload
    Given a gateway app bound to a temp runs dir with 2 synthetic runs
    When I GET "/api/stats"
    Then the response status is 200
    And the response has nested field "runs.count" equal to 2
    And the response has nested field "service.ok" equal to true
    And the response has nested field "tasks.count" greater than 0

  @tier-0
  Scenario: classification catalog exposes task registry in Atlas shape
    Given a gateway app bound to a temp runs dir with 0 runs
    When I GET "/api/classifications/catalog"
    Then the response status is 200
    And the response has an "AegirTask" root classification
    And every child has "num_labels" and "attributes"

  @tier-0
  Scenario: ontology vocabulary endpoint returns BFO-rooted skeleton
    Given a gateway app bound to a temp runs dir with 0 runs
    When I GET "/api/ontology/vocabulary"
    Then the response status is 200
    And the response contains an "ICE:DataElement" node

  @tier-0
  Scenario: ontology subsume endpoint accepts a terms payload
    Given a gateway app bound to a temp runs dir with 0 runs
    When I POST to "/api/ontology/subsume" with 3 terms
    Then the response status is 200
    And the response has exactly 3 suggestions
    And the predictor is identified as a stub

  @tier-0
  Scenario: gateway serves a graceful 404 when the UI bundle is missing
    Given a gateway app bound to a temp runs dir with 0 runs
    When I GET "/"
    Then the response status is 404
    And the response text mentions "just ui-build"

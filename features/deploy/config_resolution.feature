Feature: HOCON config loader respects env-variable overrides
  As a deployer I need env vars set by CAI / Kubernetes / devenv to
  override config/base.conf defaults so the same image deploys
  identically across environments without code changes.

  @tier-0
  Scenario: default config matches the aegir port allocation
    When I load the default config
    Then the gateway port is 8091
    And the postgres URL targets port 5555
    And the qdrant HTTP port is 6355

  @tier-0
  Scenario: AEGIR_GATEWAY_PORT overrides the default
    Given the env var AEGIR_GATEWAY_PORT is set to "9001"
    When I load the default config
    Then the gateway port is 9001

  @tier-0
  Scenario: CDSW_APP_PORT overrides the gateway port (CAI deployment)
    Given the env var CDSW_APP_PORT is set to "8100"
    When I load the default config
    Then the gateway port is 8100

  @tier-0
  Scenario: AEGIR_DB_URL overrides the default postgres URL
    Given the env var AEGIR_DB_URL is set to "postgresql+psycopg://pg.k8s.local:5432/aegir_prod"
    When I load the default config
    Then the DB URL is "postgresql+psycopg://pg.k8s.local:5432/aegir_prod"

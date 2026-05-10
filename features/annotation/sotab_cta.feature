Feature: SOTAB-CTA loader reads the official benchmark bundle
  As a research engineer I need the SOTAB V2 ground-truth CSVs + Train/
  directory layout produced by ``just get-sotab`` to drop straight into
  Aegir's training pipeline, so leaderboard rows reference the canonical
  benchmark rather than a local sample.

  @tier-0
  Scenario: SOTAB-CTA validation split loads with a stable 82-label vocab
    Given the SOTAB V2 Schema.org CTA data is present under /raid/datasets/sotab
    When I instantiate SotabCTADataset for the val split with max_context_cols=0
    Then the dataset has at least 1000 samples
    And the discovered label vocab has at least 80 entries
    And every sample's label index is within the vocab size
    And role_ids are all 0 (context columns disabled)

  @tier-0
  Scenario: SOTAB-CTA test split shares the same label vocab as val
    Given the SOTAB V2 Schema.org CTA data is present under /raid/datasets/sotab
    When I load the SOTAB-CTA val and test splits
    Then both splits index into the same label vocabulary instance

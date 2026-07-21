# The lineup's FedWiki semantics as feature-scenarios (RH 2026-07-21) — the BDD source of
# truth for the trail state machine + the content-addressed path primitive. UI scenarios are
# exercised against the pure trail semantics (extend/replace/back/deep-link); path scenarios
# run against aegir.lineup.path over the built projection.

Feature: Lineup browsing path — Ward's semantics, content-addressed
  The browsing path is the computational context: links on the newest (rightmost)
  panel EXTEND the lineup; links on earlier panels REPLACE content to their right;
  the back button walks the path leftward; a path over a kasten version is a
  durable, federated computation spec.

  Scenario: a link on the rightmost panel extends the lineup
    Given a trail of panels A, B, C
    When a link on panel C opens panel D
    Then the trail is A, B, C, D

  Scenario: a link on an earlier panel replaces everything to its right
    Given a trail of panels A, B, C
    When a link on panel A opens panel E
    Then the trail is A, E

  Scenario: the back button walks the lineup leftward
    Given a trail of panels A, B, C reached by extension
    When the browser navigates back
    Then the trail is A, B
    And the focused panel is B

  Scenario: a pasted deep link seeds a fresh trail
    Given no saved trail for the target layer
    When the lineup loads with ?open=X&root=current
    Then the trail is X

  Scenario: an in-place URL edit extends the current lineup
    Given a trail of panels A, B
    When the address bar's open parameter changes to a novel panel Y
    Then the trail is A, B, Y

  Scenario: identical trails on the same kasten version share a path id
    Given two sites holding the same kasten version
    When each encodes the trail A, B, C
    Then both derive the same path_id
    And every prefix hash matches pairwise

  Scenario: a forked lineup shares its prefix memoization keys
    Given a trail A, B, C with prefix ids P1, P2, P3
    When a second trail A, B, D is encoded
    Then its first two prefix ids equal P1, P2

  Scenario: a path outside the kasten version refuses to encode
    Given the current kasten manifest
    When a trail containing an unknown note id is encoded
    Then encoding fails naming the offending id and the kasten version

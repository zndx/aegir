Feature: EMA scan backends are numerically equivalent
  DeChunkLayer reconstructs a full-length sequence via an EMA scan. The default
  SSD-kernel backend must match the sequential reference within bf16 precision
  for both forward values and backward gradients, under the clamp/pad
  conventions that the real DeChunkLayer.forward imposes on its inputs.

  @tier-0 @gpu
  Scenario: SSD and sequential EMA scans agree on forward values
    Given a random (B,L,D)=(4,128,64) input with decay clamped to [1e-4, 1-1e-4]
    And the first-token decay forced to 1-1e-4 (matches RoutingModule pad)
    When I compute both sequential and SSD EMA scans
    Then their max relative error is below 2%

  @tier-0 @gpu
  Scenario: SSD and sequential EMA scans agree on input gradients
    Given a random (B,L,D)=(4,128,64) input with decay clamped to [1e-4, 1-1e-4]
    And the first-token decay forced to 1-1e-4 (matches RoutingModule pad)
    When I compute both sequential and SSD EMA scan input gradients
    Then their max relative gradient error is below 3%

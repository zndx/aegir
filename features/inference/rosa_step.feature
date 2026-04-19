Feature: ROSA step-mode decoding fails loud
  The ROSA suffix automaton has no well-defined single-token step semantics.
  Calling step() on a ROSA block used to silently return zero — the kind of
  latent trap that corrupts generation without warning. It should now raise
  NotImplementedError, pointing the user at RWKV-7 (w/W) as the supported
  step-mode mixer.

  @tier-0
  Scenario: RWKV_ROSA.step raises NotImplementedError
    Given a freshly constructed RWKV_ROSA block with d_model=64
    When I call step() on a fake (B=2, L=1, D=64) input with allocated state
    Then the call raises NotImplementedError
    And the error message mentions "w/W"

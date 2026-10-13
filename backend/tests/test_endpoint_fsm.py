"""Unit tests for EndpointFSM state transitions and pause resilience."""

import pytest
from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState


def test_fsm_signing_lifecycle_and_pause_resilience():
    # Setup FSM with specific durations for deterministic testing
    fsm = EndpointFSM(
        motion_threshold=0.015,
        min_sign_duration_ms=400,     # 0.4s to confirm signing
        possible_end_delay_ms=600,    # 0.6s of inactivity to enter POSSIBLE_END
        silence_ms=1200,              # 1.2s total silence to trigger END_OF_UTTERANCE
        cooldown_ms=500,
    )

    t = 100.0  # Base timestamp in seconds

    # 1. Start in IDLE
    state, commit = fsm.update(motion_energy=0.0, has_hands=True, blank_ratio=1.0, live_gloss=[], now=t)
    assert state == FSMState.IDLE
    assert not commit

    # 2. Motion begins -> STARTING
    t += 0.1
    state, commit = fsm.update(motion_energy=0.05, has_hands=True, blank_ratio=0.5, live_gloss=["TÔI"], now=t)
    assert state == FSMState.STARTING

    # 3. Motion continues beyond min_sign_duration (0.4s) -> SIGNING
    t += 0.45
    state, commit = fsm.update(motion_energy=0.04, has_hands=True, blank_ratio=0.2, live_gloss=["TÔI", "MUỐN"], now=t)
    assert state == FSMState.SIGNING

    # 4. Short pause (0.5s of low motion) -> Still in SIGNING (under 0.6s delay threshold)
    t += 0.5
    state, commit = fsm.update(motion_energy=0.005, has_hands=True, blank_ratio=0.9, live_gloss=["TÔI", "MUỐN"], now=t)
    assert state == FSMState.SIGNING, "Short pause should not leave SIGNING immediately"

    # 5. Pause extends beyond 0.6s -> Transitions to POSSIBLE_END
    t += 0.2  # Total pause = 0.7s
    state, commit = fsm.update(motion_energy=0.005, has_hands=True, blank_ratio=0.9, live_gloss=["TÔI", "MUỐN"], now=t)
    assert state == FSMState.POSSIBLE_END
    assert not commit, "POSSIBLE_END must NOT finalize utterance yet!"

    # 6. Motion resumes! (Inter-sign transition resumed) -> Returns to SIGNING
    t += 0.1
    state, commit = fsm.update(motion_energy=0.06, has_hands=True, blank_ratio=0.3, live_gloss=["TÔI", "MUỐN", "UỐNG"], now=t)
    assert state == FSMState.SIGNING, "Resumed signing should restore SIGNING state"

    # 7. Final rest: motion stops completely for 1.3s (> silence_ms 1.2s)
    t += 0.7  # Enters POSSIBLE_END
    state, commit = fsm.update(motion_energy=0.001, has_hands=True, blank_ratio=0.98, live_gloss=["TÔI", "MUỐN", "UỐNG"], now=t)
    assert state == FSMState.POSSIBLE_END

    t += 0.6  # Total pause = 1.3s (> 1.2s silence)
    state, commit = fsm.update(motion_energy=0.001, has_hands=True, blank_ratio=1.0, live_gloss=["TÔI", "MUỐN", "UỐNG"], now=t)
    assert state == FSMState.END_OF_UTTERANCE
    assert commit is True, "Utterance must be committed when silence duration is reached!"
    assert fsm.last_final_gloss == ["TÔI", "MUỐN", "UỐNG"]


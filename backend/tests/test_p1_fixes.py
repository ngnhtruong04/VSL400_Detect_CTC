"""Unit tests for P1 fixes (FIX-04, FIX-05, FIX-06)."""

import numpy as np

from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.services.continuous_session import ContinuousRecognitionSession
from backend.app.services.isolated_session import IsolatedRecognitionSession


def _make_pose(sh_l_xy, sh_r_xy):
    pose = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95, "presence": 0.95} for _ in range(33)]
    pose[11] = {"x": sh_l_xy[0], "y": sh_l_xy[1], "z": 0.0, "visibility": 0.99, "presence": 0.99}
    pose[12] = {"x": sh_r_xy[0], "y": sh_r_xy[1], "z": 0.0, "visibility": 0.99, "presence": 0.99}
    return pose


def test_fix04_single_frame_jitter_aborts_starting_to_idle():
    """FIX-04: A brief single-frame jitter burst must abort STARTING back to IDLE, never entering SIGNING."""
    fsm = EndpointFSM(
        motion_threshold=0.015,
        min_sign_duration_ms=300,
        starting_abort_ms=220,
    )
    t = 100.0
    # 1. Single strong spike triggers STARTING
    state, _ = fsm.update(motion_energy=0.04, has_hands=True, blank_ratio=1.0, live_gloss=[], now=t)
    assert state == FSMState.STARTING

    # 2. Followed immediately by silence (0.0) across 0.35s -> must abort back to IDLE, never reach SIGNING
    ever_signed = False
    for _ in range(10):
        t += 0.04
        state, _ = fsm.update(motion_energy=0.0, has_hands=True, blank_ratio=1.0, live_gloss=[], now=t)
        if state == FSMState.SIGNING:
            ever_signed = True
    assert not ever_signed, "Single-frame jitter must never promote STARTING -> SIGNING"
    assert fsm.state == FSMState.IDLE


def test_fix04_sustained_motion_confirms_signing():
    """FIX-04: Genuine sustained motion (including slow signs) confirms STARTING -> SIGNING."""
    fsm = EndpointFSM(
        motion_threshold=0.012,
        min_sign_duration_ms=300,
        starting_abort_ms=220,
    )
    t = 200.0
    for _ in range(10):  # 0.40s of steady gentle motion
        state, _ = fsm.update(motion_energy=0.018, has_hands=True, blank_ratio=0.8, live_gloss=[], now=t)
        t += 0.04
    assert state == FSMState.SIGNING


def test_fix05_signing_space_scale_invariant_near_and_far_camera():
    """FIX-05: Signing space must scale with shoulder width at both near (large) and far (small) distances."""
    # Far camera: narrow shoulders (scale=0.15) at y=0.30
    pose_far = _make_pose((0.425, 0.30), (0.575, 0.30))
    # Near camera: wide shoulders (scale=0.40) at y=0.30
    pose_near = _make_pose((0.30, 0.30), (0.70, 0.30))

    # 1. Resting hands at 1.60 shoulder widths below shoulders -> MUST be False at BOTH far and near!
    #    (Legacy fixed +0.35 failed at far distance because 0.30 + 1.6*0.15 = 0.54 <= 0.65!)
    prep_far = VSLPreprocessor()
    hand_far_rest = [{"x": 0.5, "y": 0.30 + 1.60 * 0.15, "z": 0.0} for _ in range(21)]
    assert prep_far.check_signing_space(None, hand_far_rest, pose_far) is False

    prep_near = VSLPreprocessor()
    hand_near_rest = [{"x": 0.5, "y": 0.30 + 1.60 * 0.40, "z": 0.0} for _ in range(21)]
    assert prep_near.check_signing_space(None, hand_near_rest, pose_near) is False

    # 2. Low-torso active sign at 1.05 shoulder widths below shoulders -> MUST be True at BOTH near and far!
    #    (Legacy fixed +0.35 failed at near distance because 0.30 + 1.05*0.40 = 0.72 > 0.65!)
    hand_near_low = [{"x": 0.5, "y": 0.30 + 1.05 * 0.40, "z": 0.0} for _ in range(21)]
    assert prep_near.check_signing_space(None, hand_near_low, pose_near) is True

    hand_far_low = [{"x": 0.5, "y": 0.30 + 1.05 * 0.15, "z": 0.0} for _ in range(21)]
    assert prep_far.check_signing_space(None, hand_far_low, pose_far) is True


def test_fix05_hysteresis_and_motion_assisted_lower_torso():
    """FIX-05: Hysteresis holds active state across 1.25..1.35 and strong motion prevents blind rejection at 1.40."""
    pose = _make_pose((0.35, 0.30), (0.65, 0.30))  # scale = 0.30
    prep = VSLPreprocessor()

    # Band 1.30: outside enter threshold (1.25) when idle, inside exit threshold (1.35) when already signing
    hand_130 = [{"x": 0.5, "y": 0.30 + 1.30 * 0.30, "z": 0.0} for _ in range(21)]
    prep.in_signing_space = False
    assert prep.check_signing_space(None, hand_130, pose, motion_energy=0.0) is False
    prep.in_signing_space = True
    assert prep.check_signing_space(None, hand_130, pose, motion_energy=0.0) is True

    # Lower torso (1.42) with strong intentional motion (>= 0.025) is NOT blindly rejected
    hand_142 = [{"x": 0.5, "y": 0.30 + 1.42 * 0.30, "z": 0.0} for _ in range(21)]
    prep.in_signing_space = False
    assert prep.check_signing_space(None, hand_142, pose, motion_energy=0.0) is False
    assert prep.check_signing_space(None, hand_142, pose, motion_energy=0.035) is True


def test_fix06_boundary_trim_removes_static_tails_and_preserves_internal_hold():
    """FIX-06: Leading/trailing dead static frames are trimmed while internal static holds are 100% preserved."""
    # Construct synthetic sequence:
    # [0..19]: 20 leading static frames (marker = -1.0)
    # [20..34]: 15 active motion frames (marker = 1.0)
    # [35..49]: 15 INTERNAL static hold frames (marker = 2.0, motion = 0.0!)
    # [50..64]: 15 active motion frames (marker = 3.0)
    # [65..114]: 50 trailing static frames (marker = -2.0)
    frames, motions, spaces = [], [], []
    for _ in range(20):
        frames.append(np.full(294, -1.0, dtype=np.float32))
        motions.append(0.001)
        spaces.append(False)
    for _ in range(15):
        frames.append(np.full(294, 1.0, dtype=np.float32))
        motions.append(0.030)
        spaces.append(True)
    for _ in range(15):
        frames.append(np.full(294, 2.0, dtype=np.float32))
        motions.append(0.000)
        spaces.append(True)
    for _ in range(15):
        frames.append(np.full(294, 3.0, dtype=np.float32))
        motions.append(0.030)
        spaces.append(True)
    for _ in range(50):
        frames.append(np.full(294, -2.0, dtype=np.float32))
        motions.append(0.001)
        spaces.append(False)

    # 1. Verify Mode A (IsolatedRecognitionSession)
    iso = IsolatedRecognitionSession()
    try:
        iso.frames_buffer = list(frames)
        iso.motion_buffer = list(motions)
        iso.in_space_buffer = list(spaces)
        iso.last_hand_buffer_idx = len(frames)
        trimmed_iso = iso._get_trimmed_segment()
        # Expected span: [20-8 : 64+1+6] = [12 : 71] -> 59 frames (vs 115 raw!)
        assert len(trimmed_iso) == 59
        vals = [float(f[0]) for f in trimmed_iso]
        assert vals.count(1.0) == 15
        assert vals.count(2.0) == 15, "Internal static hold must be 100% preserved!"
        assert vals.count(3.0) == 15
        assert vals.count(-2.0) == 6, "Excess trailing static tail must be trimmed!"
    finally:
        iso.close()

    # 2. Verify Mode B (ContinuousRecognitionSession)
    cont = ContinuousRecognitionSession()
    try:
        cont.utterance_buffer = list(frames)
        cont.utterance_motion_buffer = list(motions)
        cont.utterance_space_buffer = list(spaces)
        trimmed_cont = cont._get_trimmed_utterance_array()
        assert trimmed_cont is not None
        assert trimmed_cont.shape == (59, 294)
        vals_c = trimmed_cont[:, 0].tolist()
        assert vals_c.count(1.0) == 15
        assert vals_c.count(2.0) == 15, "Internal static hold / inter-sign pause must be 100% preserved!"
        assert vals_c.count(3.0) == 15
        assert vals_c.count(-2.0) == 6
    finally:
        cont.close()

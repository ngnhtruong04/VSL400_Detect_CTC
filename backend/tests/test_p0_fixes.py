"""Regression unit tests for P0 fixes (FIX-01, FIX-03) and 3D model feature contract."""

import numpy as np

from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState
from backend.app.ml.preprocessing import VSLPreprocessor


def _make_pose(sh_l_xy, sh_r_xy):
    pose = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95, "presence": 0.95} for _ in range(33)]
    pose[11] = {"x": sh_l_xy[0], "y": sh_l_xy[1], "z": 0.0, "visibility": 0.99, "presence": 0.99}
    pose[12] = {"x": sh_r_xy[0], "y": sh_r_xy[1], "z": 0.0, "visibility": 0.99, "presence": 0.99}
    return pose


def test_fix01_isotropic_aspect_invariance_4x3_and_16x9():
    """Padding a 1:1 person frame horizontally to 4:3 or 16:9 must yield identical normalized (x, y, z)."""
    # Original 1:1 square frame (640x640): shoulders at x=[0.35, 0.65], y=0.40; hand at x=0.55, y=0.60, z=-0.05
    pose_1x1 = _make_pose((0.35, 0.40), (0.65, 0.40))
    hand_1x1 = [{"x": 0.55, "y": 0.60, "z": -0.05} for _ in range(21)]

    prep_1x1 = VSLPreprocessor()
    feat_1x1, _ = prep_1x1.process_frame(None, hand_1x1, pose_1x1, frame_aspect=1.0)

    for aspect_w_over_h in [4.0 / 3.0, 16.0 / 9.0]:
        # Horizontal padding shrinks x-offsets & z by (1 / aspect_w_over_h), leaves y unchanged
        s = 1.0 / aspect_w_over_h
        pose_wide = _make_pose((0.5 + (0.35 - 0.5) * s, 0.40), (0.5 + (0.65 - 0.5) * s, 0.40))
        hand_wide = [{"x": 0.5 + (0.55 - 0.5) * s, "y": 0.60, "z": -0.05 * s} for _ in range(21)]

        prep_wide = VSLPreprocessor()
        feat_wide, _ = prep_wide.process_frame(None, hand_wide, pose_wide, frame_aspect=s)
        np.testing.assert_allclose(feat_wide, feat_1x1, atol=1e-5)


def test_fix03_preserves_3d_dxyz_model_features():
    """FIX-03 must NEVER strip z or dz from the 294-dim feature vector fed to the model."""
    prep = VSLPreprocessor()
    pose = _make_pose((0.35, 0.40), (0.65, 0.40))  # scale = 0.30
    h0 = [{"x": 0.50, "y": 0.55, "z": -0.03} for _ in range(21)]
    h1 = [{"x": 0.50, "y": 0.55, "z": -0.09} for _ in range(21)]  # Pure Z depth change = -0.06 -> dz = -0.20

    f0, _ = prep.process_frame(None, h0, pose)
    f1, me1 = prep.process_frame(None, h1, pose)

    m1 = f1.reshape(42, 7)
    # 3D model features retain exact z (-0.30) and dz (-0.20)
    np.testing.assert_allclose(m1[21, 2], -0.30, atol=1e-5)
    np.testing.assert_allclose(m1[21, 5], -0.20, atol=1e-5)
    # Pure depth jitter does NOT inflate 2D FSM motion_energy
    assert me1 == 0.0


def test_fix03_suppresses_period2_roi_oscillation_but_preserves_slow_motion():
    """Period-2 alternating MediaPipe ROI jitter must stay well below 0.008 while real slow motion exceeds it."""
    pose = _make_pose((0.35, 0.40), (0.65, 0.40))  # scale = 0.30

    # 1. Period-2 oscillation (+/- 0.006 in x -> raw dxy = 0.04/frame) + strong dz jitter
    prep_osc = VSLPreprocessor()
    osc_energies = []
    for t in range(25):
        sign = 1.0 if (t % 2 == 0) else -1.0
        hand = [{"x": 0.50 + 0.006 * sign, "y": 0.55 + 0.003 * sign, "z": 0.015 * sign} for _ in range(21)]
        _, me = prep_osc.process_frame(None, hand, pose)
        osc_energies.append(me)
    # After 6-frame warm-up, period-2 oscillation is cancelled to ~0 (well below 0.008)
    assert max(osc_energies[8:]) < 0.003

    # 2. Slow directed signing motion (0.0045 image units/frame = 0.015 normalized/frame)
    prep_slow = VSLPreprocessor()
    slow_energies = []
    for t in range(15):
        hand = [{"x": 0.40 + 0.0045 * t, "y": 0.55, "z": 0.0} for _ in range(21)]
        _, me = prep_slow.process_frame(None, hand, pose)
        slow_energies.append(me)
    assert max(slow_energies[6:]) > 0.014


def test_fix03_fsm_uses_smoothed_motion():
    """EndpointFSM must use smoothed_motion rather than raw unsmoothed motion_energy."""
    fsm = EndpointFSM(motion_threshold=0.015, min_sign_duration_ms=200, possible_end_delay_ms=400, silence_ms=800)
    t = 10.0
    fsm.update(0.05, True, 0.5, ["Bố"], now=t)
    t += 0.25
    state, _ = fsm.update(0.05, True, 0.5, ["Bố"], now=t)
    assert state == FSMState.SIGNING
    # Decay smoothed_motion below 0.015 (~0.25s) and wait past possible_end_delay_ms (0.40s)
    for _ in range(14):
        t += 0.05
        fsm.update(0.0, True, 1.0, ["Bố"], now=t)
    assert fsm.state == FSMState.POSSIBLE_END
    # A tiny 1-frame spike of 0.016 (just over 0.015) is smoothed to 0.6*0.016 = 0.0096 < 0.015 -> must NOT kick back to SIGNING!
    t += 0.04
    state, _ = fsm.update(0.016, True, 1.0, ["Bố"], now=t)
    assert state == FSMState.POSSIBLE_END

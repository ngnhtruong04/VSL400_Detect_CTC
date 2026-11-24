"""Unit tests for P2 fixes (FIX-07, FIX-08, FIX-09, FIX-10)."""

import asyncio
import numpy as np

from backend.app.config.settings import settings
from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState
from backend.app.ml.landmark_extractor import LandmarkExtractor
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.services.continuous_session import ContinuousRecognitionSession


def _make_pose(sh_l_xy, sh_r_xy, wr_l_xy=(0.35, 0.80), wr_r_xy=(0.65, 0.80)):
    pose = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95, "presence": 0.95} for _ in range(33)]
    pose[11] = {"x": sh_l_xy[0], "y": sh_l_xy[1], "z": 0.0, "visibility": 0.99, "presence": 0.99}
    pose[12] = {"x": sh_r_xy[0], "y": sh_r_xy[1], "z": 0.0, "visibility": 0.99, "presence": 0.99}
    pose[15] = {"x": wr_l_xy[0], "y": wr_l_xy[1], "z": 0.0, "visibility": 0.95, "presence": 0.95}
    pose[16] = {"x": wr_r_xy[0], "y": wr_r_xy[1], "z": 0.0, "visibility": 0.95, "presence": 0.95}
    return pose


def test_fix07_roi_remapping_preserves_exact_3d_normalized_contract():
    """FIX-07: Remapping ROI-cropped hand landmarks back to full-frame must yield exact 3D parity."""
    W, H = 640, 480
    aspect = H / float(W)
    # Square 240x240 ROI starting at (x0=200, y0=80)
    x0, y0, side = 200, 80, 240
    # True full-frame pose shoulders and hand coordinates
    pose = _make_pose((260 / W, 160 / H), (380 / W, 160 / H))
    full_hand = [{"x": 340 / W, "y": 210 / H, "z": -0.04} for _ in range(21)]

    # Simulate exact coordinates MediaPipe Hands produces inside the 240x240 ROI crop
    roi_hand = [
        {
            "x": (340 - x0) / float(side),
            "y": (210 - y0) / float(side),
            "z": -0.04 * (W / float(side)),  # MediaPipe z scales inversely with crop width
        }
        for _ in range(21)
    ]

    remapped = LandmarkExtractor._remap_hand_from_roi(roi_hand, x0, y0, side, side, W, H)
    assert remapped is not None

    prep_full = VSLPreprocessor()
    f_full, _ = prep_full.process_frame(None, full_hand, pose, frame_aspect=aspect)

    prep_roi = VSLPreprocessor()
    f_roi, _ = prep_roi.process_frame(None, remapped, pose, frame_aspect=aspect)

    np.testing.assert_allclose(f_roi, f_full, atol=1e-5)


def test_fix07_closed_loop_missed_active_wrist_detection():
    """FIX-07: Missed elevated signing wrist triggers True; detected hand near wrist triggers False."""
    W, H = 640, 480
    # Far signer: shoulders 70px wide (sh_ratio = 0.109), right wrist elevated at y=0.38
    pose = _make_pose((285 / W, 160 / H), (355 / W, 160 / H), wr_l_xy=(0.45, 0.70), wr_r_xy=(0.56, 0.38))

    class _DummyLm:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _DummyHand:
        def __init__(self, x, y):
            self.landmark = [_DummyLm(x, y)]

    class _DummyRes:
        def __init__(self, hands):
            self.multi_hand_landmarks = hands

    # Case 1: Only resting left hand detected -> elevated right wrist is MISSED -> True
    res_only_rest = _DummyRes([_DummyHand(0.45, 0.70)])
    assert LandmarkExtractor._has_missed_active_wrist(pose, res_only_rest, W, H) is True

    # Case 2: Elevated right hand IS detected near Pose wrist (0.56, 0.38) -> False (no ROI needed!)
    res_with_active = _DummyRes([_DummyHand(0.55, 0.39)])
    assert LandmarkExtractor._has_missed_active_wrist(pose, res_with_active, W, H) is False


def test_fix08_runtime_config_synchronization():
    """FIX-08: Runtime settings and ContinuousRecognitionSession must stay synchronized with runtime.yaml."""
    assert settings.CAMERA_WIDTH == 640
    assert settings.CAMERA_HEIGHT == 480
    assert settings.TARGET_FPS == 30
    assert abs(settings.CAMERA_JPEG_QUALITY - 0.80) < 1e-6
    assert settings.BUFFER_MAX_UTTERANCE_FRAMES == 450
    assert settings.COOLDOWN_MS == 200
    assert settings.INFERENCE_STRIDE == 8
    assert settings.CPU_NUM_THREADS == 2

    session = ContinuousRecognitionSession()
    try:
        assert session.max_utterance_frames == settings.BUFFER_MAX_UTTERANCE_FRAMES
        assert session.min_utterance_frames == settings.BUFFER_MIN_FRAMES
        assert session.inference_stride == settings.INFERENCE_STRIDE
        assert abs(session.fsm.cooldown_sec - settings.COOLDOWN_MS / 1000.0) < 1e-6
        assert abs(session.fsm.starting_abort_sec - settings.STARTING_ABORT_MS / 1000.0) < 1e-6
    finally:
        session.close()


def test_fix09_long_sequence_buffer_cap_auto_commits_cleanly():
    """FIX-09: Reaching max_utterance_frames during SIGNING cleanly finalizes with endpoint_reason='max_buffer_reached'."""
    session = ContinuousRecognitionSession(max_utterance_frames=20, min_utterance_frames=10)
    try:
        # Put FSM directly into SIGNING with 19 frames already buffered
        session.fsm.state = FSMState.SIGNING
        session.fsm.last_motion_time = 1000.0
        session.fsm.last_update_time = 1000.0
        session.fsm.smoothed_motion = 0.05
        for _ in range(19):
            session.utterance_buffer.append(np.zeros(294, dtype=np.float32))
            session.utterance_motion_buffer.append(0.03)
            session.utterance_space_buffer.append(True)
        session._last_active_buf_len = 19

        # Mock extractor to return a valid moving right hand so 20th frame stays in SIGNING & hits cap
        pose = _make_pose((0.35, 0.40), (0.65, 0.40))
        hand = [{"x": 0.55, "y": 0.50, "z": 0.0} for _ in range(21)]
        session.extractor.extract = lambda frame_bgr, draw_landmarks=False: (None, hand, pose, frame_bgr)

        async def _run():
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            msg, _ = await session.process_frame(dummy, draw_landmarks=False)
            return msg

        msg = asyncio.run(_run())
        assert msg.state == FSMState.END_OF_UTTERANCE.value
        assert msg.continuous.endpoint_reason == "max_buffer_reached"
        assert msg.continuous.utterance_frames == 20
    finally:
        session.close()


def test_fix10_short_cooldown_recovers_rapid_repeat_without_false_bounce():
    """FIX-10: 200ms cooldown allows rapid repeated sign onset while FIX-04 still aborts post-commit 1-frame jitter."""
    fsm = EndpointFSM(
        motion_threshold=0.008,
        min_sign_duration_ms=300,
        possible_end_delay_ms=600,
        silence_ms=1200,
        cooldown_ms=200,
        starting_abort_ms=220,
    )
    t = 500.0
    # Enter END_OF_UTTERANCE -> COOLDOWN
    fsm._set_state(FSMState.END_OF_UTTERANCE, t)
    t += 0.04
    st, _ = fsm.update(0.0, True, 1.0, [], now=t)
    assert st == FSMState.COOLDOWN

    # After 0.21s (> 200ms), COOLDOWN resets to IDLE
    t += 0.21
    st, _ = fsm.update(0.0, True, 1.0, [], now=t)
    assert st == FSMState.IDLE

    # A brief post-cooldown jitter spike (0.015 > 0.008) followed by silence must abort STARTING -> IDLE
    t += 0.04
    st, _ = fsm.update(0.015, True, 1.0, [], now=t)
    assert st == FSMState.STARTING
    for _ in range(8):
        t += 0.04
        st, _ = fsm.update(0.0, True, 1.0, [], now=t)
    assert st == FSMState.IDLE

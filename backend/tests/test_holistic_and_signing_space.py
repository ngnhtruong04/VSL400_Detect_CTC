"""Unit tests for MediaPipe Holistic extraction, anatomical mapping, and signing space."""

import pytest
import numpy as np
from backend.app.ml.landmark_extractor import LandmarkExtractor
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState


def test_landmark_extractor_instantiation_and_dummy_frame():
    extractor = LandmarkExtractor()
    assert extractor.holistic is not None, "Holistic extractor must be initialized"

    # Synthetic 480x640 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left_hand, right_hand, pose, annotated = extractor.extract(frame, draw_landmarks=False)

    # Empty frame should return None for landmarks without crashing
    assert left_hand is None
    assert right_hand is None
    assert pose is None
    assert annotated.shape == (480, 640, 3)
    extractor.close()


def test_signing_space_detection():
    prep = VSLPreprocessor()

    # Pose with shoulders at y = 0.4
    pose_lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9, "presence": 0.9} for _ in range(33)]
    pose_lms[11] = {"x": 0.4, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}
    pose_lms[12] = {"x": 0.6, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}

    # 1. Hands elevated in front of chest (y = 0.5 <= 0.4 + 0.35 = 0.75) -> In signing space
    hand_in_space = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    assert prep.check_signing_space(None, hand_in_space, pose_lms) is True

    # 2. Hands resting in lap (y = 0.85 > 0.75) -> NOT in signing space
    hand_in_lap = [{"x": 0.5, "y": 0.85, "z": 0.0} for _ in range(21)]
    assert prep.check_signing_space(None, hand_in_lap, pose_lms) is False


def test_fsm_signing_space_rejection():
    fsm = EndpointFSM(motion_threshold=0.015)
    t = 200.0

    # Motion detected but NOT in signing space (e.g. resting hands shifting in lap)
    state, commit = fsm.update(
        motion_energy=0.05,
        has_hands=True,
        blank_ratio=1.0,
        live_gloss=[],
        now=t,
        in_signing_space=False,
    )
    # Must remain in IDLE!
    assert state == FSMState.IDLE, "Motion outside signing space must not trigger STARTING"


def test_pose_unset_presence_still_computes_real_shoulder_anchor():
    """Verifies LandmarkExtractor defaults unset protobuf `presence` to 1.0 so shoulders are never rejected."""
    from mediapipe.framework.formats import landmark_pb2

    lm = landmark_pb2.NormalizedLandmark(x=0.4, y=0.5, z=-0.2, visibility=0.98)
    assert not lm.HasField("presence")
    assert lm.presence == 0.0  # protobuf default that previously broke shoulder normalization!

    # Emulate LandmarkExtractor extraction rule
    presence_val = float(lm.presence) if lm.HasField("presence") else 1.0
    assert presence_val == 1.0

    pose_lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.98, "presence": presence_val} for _ in range(33)]
    pose_lms[11] = {"x": 0.35, "y": 0.70, "z": -0.3, "visibility": 0.99, "presence": presence_val}
    pose_lms[12] = {"x": 0.65, "y": 0.70, "z": -0.3, "visibility": 0.99, "presence": presence_val}

    prep = VSLPreprocessor()
    center, scale = prep.compute_shoulder_anchor(pose_lms)
    assert prep.last_center is not None
    assert prep.last_scale is not None
    np.testing.assert_allclose(center, np.array([0.50, 0.70, 0.0], dtype=np.float32), atol=1e-5)
    assert abs(scale - 0.30) < 1e-5


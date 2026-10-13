"""Unit tests for VSLPreprocessor parity and feature contracts."""

import pytest
import numpy as np
from backend.app.ml.preprocessing import VSLPreprocessor


def test_preprocessor_shape_and_dtype():
    prep = VSLPreprocessor()

    # Synthetic pose with 33 landmarks
    pose_lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9, "presence": 0.9} for _ in range(33)]
    pose_lms[11] = {"x": 0.4, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}  # Left shoulder
    pose_lms[12] = {"x": 0.6, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}  # Right shoulder

    # Synthetic hands (21 joints each)
    left_hand = [{"x": 0.35 + 0.01 * i, "y": 0.6, "z": 0.0} for i in range(21)]
    right_hand = [{"x": 0.65 - 0.01 * i, "y": 0.6, "z": 0.0} for i in range(21)]

    # Frame 0: First frame
    feat0, motion0 = prep.process_frame(left_hand, right_hand, pose_lms)
    assert feat0.shape == (294,), f"Expected (294,), got {feat0.shape}"
    assert feat0.dtype == np.float32
    assert motion0 == 0.0, "First frame motion energy should be 0.0"

    # Frame 1: Hand moved slightly
    left_hand_moved = [{"x": lm["x"] + 0.02, "y": lm["y"] + 0.01, "z": 0.0} for lm in left_hand]
    feat1, motion1 = prep.process_frame(left_hand_moved, right_hand, pose_lms)

    assert feat1.shape == (294,)
    assert motion1 > 0.0, "Motion energy should be positive after movement"

    # Verify 42 hand joints * 7 features breakdown
    feat_matrix = feat1.reshape(42, 7)
    # Check valid flags
    assert np.all(feat_matrix[:, 6] == 1.0), "All 42 hand joints should be marked valid"
    # Check delta velocity on left hand (first 21 joints)
    assert np.all(feat_matrix[:21, 3] > 0.0), "Left hand dx should reflect movement"
    # Check delta velocity on right hand (stationary)
    assert np.all(feat_matrix[21:, 3] == 0.0), "Right hand dx should be 0"


def test_missing_hand_handling():
    prep = VSLPreprocessor()
    pose_lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9, "presence": 0.9} for _ in range(33)]
    pose_lms[11] = {"x": 0.4, "y": 0.4, "z": 0.0, "visibility": 0.9, "presence": 0.9}
    pose_lms[12] = {"x": 0.6, "y": 0.4, "z": 0.0, "visibility": 0.9, "presence": 0.9}

    # Only right hand present, left hand missing
    right_hand = [{"x": 0.6, "y": 0.6, "z": 0.0} for _ in range(21)]
    feat, _ = prep.process_frame(None, right_hand, pose_lms)

    feat_matrix = feat.reshape(42, 7)
    # Left hand (0:21) should be all zeros including valid flag
    assert np.all(feat_matrix[:21] == 0.0)
    # Right hand (21:42) should have valid flag 1.0
    assert np.all(feat_matrix[21:, 6] == 1.0)


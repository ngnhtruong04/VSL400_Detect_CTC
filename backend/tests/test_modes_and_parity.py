"""Comprehensive tests for Mode A (Isolated), Mode B (Continuous), and Strict Parity."""

import numpy as np
import pytest
import torch

from backend.app.config.settings import settings
from backend.app.ml.ctc_decoder import CTCDecoder
from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState
from backend.app.ml.model_loader import ModelLoader
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.services.isolated_session import IsolatedRecognitionSession, IsolatedState
from backend.app.services.continuous_session import ContinuousRecognitionSession


def test_isolated_session_can_emit_nonblank_gloss():
    session = IsolatedRecognitionSession()
    assert session.state == IsolatedState.IDLE

    # Create synthetic sign segment with active features (T=30 frames, 294 features)
    features_list = []
    for t in range(30):
        feat = np.zeros(294, dtype=np.float32)
        # Mark right hand joints (indices 21..41) valid
        for j in range(21, 42):
            feat[j * 7 + 0] = 0.5 + 0.1 * np.sin(t / 5.0)  # x
            feat[j * 7 + 1] = -0.2 + 0.05 * np.cos(t / 5.0)  # y
            feat[j * 7 + 3] = 0.02  # dx
            feat[j * 7 + 6] = 1.0  # valid
        features_list.append(feat)

    pred = session.infer_segment(features_list)
    assert pred is not None
    assert len(pred.top5) == 5
    assert pred.top5[0].rank == 1
    assert pred.top5[0].diagnostic_confidence >= pred.top5[1].diagnostic_confidence
    assert pred.gloss != ""
    assert pred.label_id >= 0


def test_isolated_reset_clears_buffer():
    session = IsolatedRecognitionSession()
    session.frames_buffer.append(np.ones(294, dtype=np.float32))
    session.state = IsolatedState.RECORDING

    session.reset()
    assert session.state == IsolatedState.IDLE
    assert len(session.frames_buffer) == 0
    assert session.last_prediction is None


def test_isolated_manual_record_auto_stops_when_hands_disappear(monkeypatch):
    """Manual Start Record waits for hands, tolerates brief flicker, and auto-stops & trims when both hands disappear."""
    import asyncio
    import backend.app.services.isolated_session as iso_mod

    session = IsolatedRecognitionSession(min_frames=10, max_frames=180, hands_absent_stop_sec=0.45)

    pose = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95, "presence": 0.95} for _ in range(33)]
    pose[11] = {"x": 0.35, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}
    pose[12] = {"x": 0.65, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}

    def make_hand(step: int):
        return [{"x": 0.5 + 0.01 * step, "y": 0.45, "z": 0.0} for _ in range(21)]

    fake_now = [1000.0]
    monkeypatch.setattr(iso_mod.time, "time", lambda: fake_now[0])

    # Current mocked extractor output: (left_hand, right_hand, pose)
    current_lms = [None, None, pose]
    monkeypatch.setattr(
        session.extractor,
        "extract",
        lambda frame, draw=False: (current_lms[0], current_lms[1], current_lms[2], frame),
    )

    dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    async def run_scenario():
        session.start_recording()
        assert session.state == IsolatedState.RECORDING

        # 1. Hands not raised yet (user moving off mouse for 1.0s) -> stays RECORDING, 0 buffered frames
        for _ in range(5):
            fake_now[0] += 0.2
            msg, _ = await session.process_frame(dummy_frame)
            assert msg.state == "RECORDING"
            assert msg.model.buffer_frames == 0

        # 2. Accidental brief hand flash (3 frames < min_frames=10), then hands disappear > 0.45s -> resets buffer & keeps waiting
        for i in range(3):
            fake_now[0] += 0.033
            current_lms[1] = make_hand(i)
            msg, _ = await session.process_frame(dummy_frame)
            assert msg.state == "RECORDING"
        current_lms[1] = None
        fake_now[0] += 0.033
        await session.process_frame(dummy_frame)
        fake_now[0] += 0.5
        msg, _ = await session.process_frame(dummy_frame)
        assert msg.state == "RECORDING"
        assert msg.model.buffer_frames == 0

        # 3. Actual fast sign: 6 hand frames -> 1 brief flicker frame (0.1s < 0.45s) -> 6 more hand frames = 13 frames total (12 hand frames)
        for i in range(6):
            fake_now[0] += 0.033
            current_lms[1] = make_hand(i)
            msg, _ = await session.process_frame(dummy_frame)
            assert msg.state == "RECORDING"

        current_lms[1] = None
        fake_now[0] += 0.10
        msg, _ = await session.process_frame(dummy_frame)
        assert msg.state == "RECORDING", "Brief tracking dropout (< 0.45s) must not prematurely stop recording"

        for i in range(6, 12):
            fake_now[0] += 0.033
            current_lms[1] = make_hand(i)
            msg, _ = await session.process_frame(dummy_frame)
            assert msg.state == "RECORDING"

        # 4. User lowers both hands (no hand skeleton detected) -> starts debounce timer
        current_lms[0] = None
        current_lms[1] = None
        fake_now[0] += 0.05
        msg, _ = await session.process_frame(dummy_frame)
        assert msg.state == "RECORDING"

        # Still under 0.45s -> keeps recording
        fake_now[0] += 0.30
        msg, _ = await session.process_frame(dummy_frame)
        assert msg.state == "RECORDING"

        # Crosses 0.45s without hands -> auto-stops immediately (well before 180 frames!) & trims trailing no-hand frames
        fake_now[0] += 0.20
        msg, _ = await session.process_frame(dummy_frame)
        assert msg.state == "RESULT"
        assert msg.prediction is not None
        assert msg.prediction.gloss != ""
        assert len(msg.prediction.top5) == 5
        # 6 hand + 1 flicker + 6 hand = 13 frames; trailing 3 no-hand debounce frames were trimmed
        assert msg.model.buffer_frames == 13
        # Short 13-frame clip is resampled to min_inference_frames (45) + 2*edge_hold_frames (8) = 53
        assert msg.model.input_shape == "(1, 53, 294)"

    try:
        asyncio.run(run_scenario())
    finally:
        session.close()


def test_isolated_temporal_prepare_preserves_missing_hand_contract():
    """Short-clip resampling + endpoint_hold must never leak non-zero coords/vel into missing hands."""
    session = IsolatedRecognitionSession(min_inference_frames=45, edge_hold_frames=4)
    try:
        raw_frames = []
        for t in range(18):
            mat = np.zeros((42, 7), dtype=np.float32)
            # Only right hand (21..41) present; left hand (0..20) completely missing
            mat[21:, 0] = -0.3 + 0.02 * t
            mat[21:, 1] = 0.2 - 0.01 * t
            mat[21:, 6] = 1.0
            raw_frames.append(mat.reshape(294))

        prepared = session._prepare_inference_sequence(raw_frames)
        assert prepared.shape == (53, 294)  # 45 resampled + 4 start_hold + 4 end_hold
        p_mat = prepared.reshape(53, 42, 7)

        # Missing left hand stays strictly 0 everywhere
        assert np.all(p_mat[:, :21, :] == 0.0)
        # Start-hold frames (0..4) and end-hold frames (-4:) have zero velocity (dxyz == 0)
        assert np.all(p_mat[:5, :, 3:6] == 0.0)
        assert np.all(p_mat[-4:, :, 3:6] == 0.0)
        # Active middle frames have smooth recomputed positive dx on right hand
        assert np.all(p_mat[6:-5, 21:, 3] > 0.0)
    finally:
        session.close()


def test_continuous_accepts_same_features_as_isolated():
    iso_session = IsolatedRecognitionSession()
    cont_session = ContinuousRecognitionSession()

    feat = np.random.randn(294).astype(np.float32)
    # Both sessions should accept the exact same 294-dim feature shape without error
    iso_session.frames_buffer.append(feat)
    cont_session.utterance_buffer.append(feat)
    cont_session.recent_window.append(feat)

    assert len(iso_session.frames_buffer) == 1
    assert len(cont_session.utterance_buffer) == 1
    assert len(cont_session.recent_window) == 1


def test_recent_window_does_not_delete_utterance_history():
    session = ContinuousRecognitionSession(recent_window_size=90, max_utterance_frames=300)

    # Simulate 150 frames recorded during active utterance
    for _ in range(150):
        feat = np.zeros(294, dtype=np.float32)
        session.recent_window.append(feat)
        session.utterance_buffer.append(feat)

    # Recent window is capped at 90 frames for real-time display
    assert len(session.recent_window) == 90
    # Utterance buffer retains all 150 frames without truncation
    assert len(session.utterance_buffer) == 150


def test_ctc_repeated_gloss_separation():
    decoder = CTCDecoder(label_map_path=settings.LABEL_MAP_PATH, blank_id=400)

    # 1. Repeated token separated by blank: Token 5 -> Blank (400) -> Token 5
    logits_sep = torch.zeros(1, 6, 401)
    logits_sep[0, 0:2, 5] = 10.0   # Token 5
    logits_sep[0, 2:4, 400] = 10.0 # Blank
    logits_sep[0, 4:6, 5] = 10.0   # Token 5
    glosses, collapsed, raw, _ = decoder.decode_greedy(logits_sep)
    assert collapsed == [5, 5], f"Expected repeated gloss [5, 5], got {collapsed}"
    assert len(glosses) == 2

    # 2. Repeated token without blank: Token 5 continuously active
    logits_cont = torch.zeros(1, 6, 401)
    logits_cont[0, 0:6, 5] = 10.0
    glosses_c, collapsed_c, _, _ = decoder.decode_greedy(logits_cont)
    assert collapsed_c == [5], f"Expected collapsed single token [5], got {collapsed_c}"
    assert len(glosses_c) == 1


def test_fsm_does_not_reset_utterance_on_short_pause():
    fsm = EndpointFSM(possible_end_delay_ms=600, silence_ms=1200)
    t = 100.0

    # Start motion
    fsm.update(motion_energy=0.05, has_hands=True, blank_ratio=0.5, live_gloss=["TÔI"], now=t)
    t += 0.45
    state, commit = fsm.update(motion_energy=0.04, has_hands=True, blank_ratio=0.3, live_gloss=["TÔI"], now=t)
    assert state == FSMState.SIGNING

    # 0.4s pause (short pause during signing)
    t += 0.4
    state, commit = fsm.update(motion_energy=0.005, has_hands=True, blank_ratio=0.9, live_gloss=["TÔI"], now=t)
    assert state == FSMState.SIGNING, "Short natural pause must not abort SIGNING state"
    assert commit is False


def test_fsm_finalizes_after_real_end():
    fsm = EndpointFSM(possible_end_delay_ms=600, silence_ms=1200)
    t = 200.0

    # Start and sustain signing
    fsm.update(motion_energy=0.05, has_hands=True, blank_ratio=0.2, live_gloss=["CẢM_ƠN"], now=t)
    t += 0.45
    fsm.update(motion_energy=0.04, has_hands=True, blank_ratio=0.2, live_gloss=["CẢM_ƠN"], now=t)

    # Inactivity for 0.7s -> POSSIBLE_END
    t += 0.7
    state, commit = fsm.update(motion_energy=0.002, has_hands=True, blank_ratio=0.95, live_gloss=["CẢM_ƠN"], now=t)
    assert state == FSMState.POSSIBLE_END
    assert commit is False

    # Inactivity extends past silence timeout (1.3s total pause > 1.2s)
    t += 0.6
    state, commit = fsm.update(motion_energy=0.001, has_hands=True, blank_ratio=1.0, live_gloss=["CẢM_ƠN"], now=t)
    assert state == FSMState.END_OF_UTTERANCE
    assert commit is True
    assert fsm.last_final_gloss == ["CẢM_ƠN"]


def test_continuous_reset_clears_state():
    session = ContinuousRecognitionSession()
    session.utterance_buffer.append(np.ones(294, dtype=np.float32))
    session.live_tokens = ["XIN_CHÀO"]
    session.final_tokens = ["XIN_CHÀO"]

    session.reset()
    assert len(session.utterance_buffer) == 0
    assert len(session.recent_window) == 0
    assert session.live_tokens == []
    assert session.session_final_tokens == []
    assert session.final_tokens == []
    assert session.fsm.state == FSMState.IDLE


def test_rule_based_sentence_formatting_natural_vietnamese():
    """Rule-based fallback must reorder VSL S-O-V -> Vietnamese S-V-O and keep natural casing without mid-sentence periods."""
    import asyncio
    from backend.app.services.language_service import LanguageService

    svc = LanguageService(enabled=False, provider="none")
    # S-V-O already in Vietnamese order
    assert asyncio.run(svc.to_natural_sentence(["Chú", "Giúp đỡ", "Bố"])) == "Chú giúp đỡ bố."
    # Classic VSL S-O-V -> Vietnamese S-V-O
    assert asyncio.run(svc.to_natural_sentence(["Bố", "Chú", "Giúp đỡ"])) == "Bố giúp đỡ chú."
    assert asyncio.run(svc.to_natural_sentence(["Buổi sáng", "Bố", "Chú", "Giúp đỡ"])) == "Buổi sáng bố giúp đỡ chú."
    assert asyncio.run(svc.to_natural_sentence(["Bố", "Chợ", "Đi"])) == "Bố đi chợ."
    assert asyncio.run(svc.to_natural_sentence(["Chú", "Bệnh viện", "Khám bệnh"])) == "Chú khám bệnh ở bệnh viện."
    assert asyncio.run(svc.to_natural_sentence(["Mẹ", "Chợ", "Đi", "Mua bán"])) == "Mẹ đi chợ mua bán."
    # 2-word: O-V reorders for non-person objects, S-V stays intact for people
    assert asyncio.run(svc.to_natural_sentence(["Phở", "Ăn"])) == "Ăn phở."
    assert asyncio.run(svc.to_natural_sentence(["Bố", "Ăn"])) == "Bố ăn."
    # Proper nouns & parenthetical labels
    assert asyncio.run(svc.to_natural_sentence(["Chú", "Đi", "Việt Nam"])) == "Chú đi Việt Nam."
    assert asyncio.run(svc.to_natural_sentence(["Bố", "Cao (người)"])) == "Bố cao."


def test_continuous_multi_segment_translates_full_accumulated_gloss_sequence(tmp_path):
    """Signing Chú -> pause -> Giúp đỡ -> pause -> Bố must produce ONE sentence 'Chú giúp đỡ bố.', never 'Chú. Giúp đỡ. Bố.'."""
    import asyncio
    from backend.app.services.language_service import LanguageService

    session = ContinuousRecognitionSession(
        language_service=LanguageService(enabled=False, provider="none")
    )
    session.eval_log_path = tmp_path / "multi_seg.jsonl"
    try:
        async def run_flow():
            # Segment 1: Chú
            session.live_tokens = ["Chú"]
            msg1 = await session.commit_utterance_async()
            assert msg1.continuous.final_tokens == ["Chú"]
            assert msg1.continuous.final_sentence == "Chú."

            # Segment 2: Giúp đỡ
            session.live_tokens = ["Giúp đỡ"]
            msg2 = await session.commit_utterance_async()
            assert msg2.continuous.final_tokens == ["Chú", "Giúp đỡ"]
            assert msg2.continuous.final_sentence == "Chú giúp đỡ."

            # Segment 3: Bố
            session.live_tokens = ["Bố"]
            msg3 = await session.commit_utterance_async()
            assert msg3.continuous.final_tokens == ["Chú", "Giúp đỡ", "Bố"]
            assert msg3.continuous.final_sentence == "Chú giúp đỡ bố."

            # Re-translating manually with empty active buffer still translates the full accumulated sequence
            msg4 = await session.commit_utterance_async()
            assert msg4.continuous.final_tokens == ["Chú", "Giúp đỡ", "Bố"]
            assert msg4.continuous.final_sentence == "Chú giúp đỡ bố."

        asyncio.run(run_flow())
    finally:
        session.close()


def test_eval_5_6_1_telemetry_and_wer_summary(tmp_path):
    """Verifies automatic JSONL telemetry logging and Section 5.6.1 WER/ESM/latency calculator."""
    import asyncio
    import json
    from backend.app.services.language_service import LanguageService
    from scripts.evaluate_webcam_5_6_1 import compute_wer_esm, parse_session_log

    # 1. Verify all 15 template glosses exist in VSL400 label_map.json
    with open(settings.LABEL_MAP_PATH, "r", encoding="utf-8") as lf:
        label_to_id = json.load(lf)["label_to_id"]
    with open(settings.PROJECT_ROOT / "scripts" / "webcam_script_template.json", "r", encoding="utf-8") as sf:
        tpl = json.load(sf)
    for trial in tpl["trials"]:
        for g in trial["reference_gloss"]:
            assert g in label_to_id, f"Template gloss {g!r} not in VSL400 label_map.json!"

    # 2. Simulate 2 webcam trials delimited by reset() on a temporary log file
    session = ContinuousRecognitionSession(
        language_service=LanguageService(enabled=False, provider="none")
    )
    session.eval_log_path = tmp_path / "eval_test.jsonl"
    try:
        async def run_trials():
            # Fake frame telemetry
            session._record_eval_event({"event": "frame", "ts": 1.0, "fps": 24.5, "mediapipe_ms": 18.2, "preprocessing_ms": 0.4, "fsm_state": "SIGNING"}, flush=True)
            session._record_eval_event({"event": "frame", "ts": 1.1, "fps": 25.5, "mediapipe_ms": 22.0, "preprocessing_ms": 0.5, "fsm_state": "SIGNING"}, flush=True)

            # Trial 1: exact match ['Bố', 'Chú', 'Giúp đỡ']
            session.live_tokens = ["Bố", "Chú", "Giúp đỡ"]
            await session.commit_utterance_async()
            session.reset()

            # Trial 2: 1 substitution ['Bố', 'Chợ', 'Đi'] vs ref ['Bố', 'Nhà', 'Đi']
            session.live_tokens = ["Bố", "Chợ", "Đi"]
            await session.commit_utterance_async()
            session.reset()

        asyncio.run(run_trials())
    finally:
        session.close()

    parsed = parse_session_log(tmp_path / "eval_test.jsonl")
    assert len(parsed["trials"]) == 2
    assert parsed["trials"][0]["predicted_gloss"] == ["Bố", "Chú", "Giúp đỡ"]
    assert parsed["trials"][0]["predicted_sentence"] == "Bố giúp đỡ chú."
    assert parsed["trials"][1]["predicted_gloss"] == ["Bố", "Chợ", "Đi"]
    assert len(parsed["mediapipe_ms"]) == 2

    refs = [["Bố", "Chú", "Giúp đỡ"], ["Bố", "Nhà", "Đi"]]
    hyps = [t["predicted_gloss"] for t in parsed["trials"]]
    metrics = compute_wer_esm(refs, hyps)
    assert metrics["S"] == 1 and metrics["D"] == 0 and metrics["I"] == 0
    assert metrics["exact_count"] == 1
    assert abs(metrics["wer_pct"] - 16.67) < 0.05
    assert abs(metrics["esm_pct"] - 50.0) < 0.05


def test_sessions_are_isolated():
    session1 = ContinuousRecognitionSession()
    session2 = ContinuousRecognitionSession()

    session1.utterance_buffer.append(np.ones(294, dtype=np.float32))
    assert len(session1.utterance_buffer) == 1
    assert len(session2.utterance_buffer) == 0


def test_nan_inf_never_reaches_model():
    prep = VSLPreprocessor()

    # Create dummy landmarks containing NaN and Inf
    pose = [{"x": float("nan"), "y": float("inf"), "z": 0.0, "visibility": 0.9, "presence": 0.9} for _ in range(33)]
    left = [{"x": float("nan"), "y": 0.5, "z": 0.0} for _ in range(21)]
    right = [{"x": 0.5, "y": float("inf"), "z": 0.0} for _ in range(21)]

    feat, _ = prep.process_frame(left, right, pose)
    assert not np.isnan(feat).any(), "NaN values must be cleaned up before reaching model"
    assert not np.isinf(feat).any(), "Inf values must be cleaned up before reaching model"


def test_missing_hand_preserves_valid_mask():
    prep = VSLPreprocessor()
    pose = [{"x": 0.5, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95} for _ in range(33)]
    pose[11] = {"x": 0.4, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}
    pose[12] = {"x": 0.6, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}

    # Left hand is missing (None), right hand is present
    right = [{"x": 0.6, "y": 0.5, "z": 0.0} for _ in range(21)]
    feat, _ = prep.process_frame(None, right, pose)

    matrix = feat.reshape(42, 7)
    # Left hand (0..20) must be 100% zeros including valid mask
    assert np.all(matrix[:21] == 0.0), "Missing left hand must have all 0 features and valid=0"
    # Right hand (21..41) valid column must be 1.0
    assert np.all(matrix[21:, 6] == 1.0), "Present right hand must have valid=1"


def test_golden_parity_mathematical_contract():
    """Validates mathematical parity of shoulder anchor normalization and dxyz."""
    prep = VSLPreprocessor(min_scale=0.0001, shoulder_scale_multiplier=1.0, clip_value=6.0)

    # Frame 0
    pose = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95, "presence": 0.95} for _ in range(33)]
    pose[11] = {"x": 0.35, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}
    pose[12] = {"x": 0.65, "y": 0.4, "z": 0.0, "visibility": 0.95, "presence": 0.95}

    # Expected Center & Scale
    expected_center = np.array([0.5, 0.4, 0.0], dtype=np.float32)
    expected_dist = float(np.linalg.norm(np.array([0.35, 0.4]) - np.array([0.65, 0.4]))) # 0.30
    expected_scale = expected_dist

    center, scale = prep.compute_shoulder_anchor(pose)
    np.testing.assert_allclose(center, expected_center, rtol=1e-5)
    assert abs(scale - expected_scale) < 1e-5

    # Hand point
    right_raw = [{"x": 0.5, "y": 0.55, "z": 0.0} for _ in range(21)]
    feat0, motion0 = prep.process_frame(None, right_raw, pose)

    expected_norm_y = (0.55 - 0.40) / 0.30  # 0.15 / 0.30 = 0.5
    matrix0 = feat0.reshape(42, 7)
    np.testing.assert_allclose(matrix0[21, 1], expected_norm_y, rtol=1e-5)
    # First frame delta dxyz must be 0.0
    assert np.all(matrix0[:, 3:6] == 0.0)
    assert motion0 == 0.0

    # Frame 1: Hand moved by dy = +0.03
    right_raw_moved = [{"x": 0.5, "y": 0.58, "z": 0.0} for _ in range(21)]
    feat1, motion1 = prep.process_frame(None, right_raw_moved, pose)

    expected_norm_y_1 = (0.58 - 0.40) / 0.30  # 0.18 / 0.30 = 0.6
    expected_dy = expected_norm_y_1 - expected_norm_y  # 0.10
    matrix1 = feat1.reshape(42, 7)
    np.testing.assert_allclose(matrix1[21, 1], expected_norm_y_1, rtol=1e-5)
    np.testing.assert_allclose(matrix1[21, 4], expected_dy, rtol=1e-5)
    assert motion1 > 0.0


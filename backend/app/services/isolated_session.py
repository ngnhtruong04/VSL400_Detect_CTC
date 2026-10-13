"""Isolated Recognition Session (Mode A) for Single-Gloss Recognition.

Flow:
Webcam -> MediaPipe -> Strict Preprocessing -> Isolated Segment Buffer
-> Detect Signing Interval -> Collect Complete Sign Segment
-> Model Inference on Full Segment -> Single-Gloss Prediction + Top-5 -> Output.
"""

import logging
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from backend.app.config.settings import settings
from backend.app.ml.landmark_extractor import LandmarkExtractor
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.ml.ctc_decoder import CTCDecoder
from backend.app.ml.model_loader import ModelLoader
from backend.app.schemas.api_models import (
    IsolatedRecognitionMessage,
    IsolatedPrediction,
    TopCandidate,
    LandmarksDiagnostic,
    ModelDiagnostic,
    NormalizedPoint,
)

logger = logging.getLogger("vsl.isolated_session")


class IsolatedState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    INFERRING = "INFERRING"
    RESULT = "RESULT"


class IsolatedRecognitionSession:
    """Manages an isolated single-gloss recognition session independent of continuous FSM."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        decoder: Optional[CTCDecoder] = None,
        min_frames: int = 15,
        max_frames: int = 120,
        motion_start_threshold: float = 0.015,
        motion_continue_threshold: float = 0.008,
        pause_timeout_sec: float = 0.55,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.extractor = LandmarkExtractor()
        self.preprocessor = VSLPreprocessor(
            pose_visibility_threshold=settings.POSE_VISIBILITY_THRESHOLD,
            pose_presence_threshold=settings.POSE_PRESENCE_THRESHOLD,
            min_scale=settings.MIN_SCALE,
            shoulder_scale_multiplier=settings.SHOULDER_SCALE_MULTIPLIER,
            clip_value=settings.CLIP_VALUE,
        )
        self.decoder = decoder or CTCDecoder(
            label_map_path=settings.LABEL_MAP_PATH,
            blank_id=settings.BLANK_ID,
        )
        self.model, self.device = ModelLoader.get_loaded_model()

        # Constraints
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.motion_start_threshold = motion_start_threshold
        self.motion_continue_threshold = motion_continue_threshold
        self.pause_timeout_sec = pause_timeout_sec

        # State machine
        self.state: IsolatedState = IsolatedState.IDLE
        self.frames_buffer: List[np.ndarray] = []
        self.recording_start_time: Optional[float] = None
        self.last_motion_time: Optional[float] = None
        self.last_prediction: Optional[IsolatedPrediction] = None

        # Timing & Performance
        self.frame_times: List[float] = []
        self.last_forward_ms: float = 0.0
        self.last_decode_ms: float = 0.0

    def reset(self):
        """Clears buffers and returns to IDLE state."""
        self.state = IsolatedState.IDLE
        self.frames_buffer.clear()
        self.preprocessor.reset_state()
        self.recording_start_time = None
        self.last_motion_time = None
        self.last_prediction = None
        self.last_forward_ms = 0.0
        self.last_decode_ms = 0.0

    def start_recording(self):
        """Manually trigger recording for isolated sign."""
        self.state = IsolatedState.RECORDING
        self.frames_buffer.clear()
        self.recording_start_time = time.time()
        self.last_motion_time = time.time()
        self.last_prediction = None

    def stop_recording(self):
        """Manually finish recording and trigger inference."""
        if self.state == IsolatedState.RECORDING:
            self.state = IsolatedState.INFERRING

    def finalize_and_infer(self) -> IsolatedRecognitionMessage:
        """Forces immediate inference on current buffer and returns message."""
        self.last_prediction = self.infer_segment(self.frames_buffer)
        self.state = IsolatedState.RESULT

        model_diag = ModelDiagnostic(
            buffer_frames=len(self.frames_buffer),
            input_shape=f"(1, {len(self.frames_buffer)}, 294)" if len(self.frames_buffer) > 0 else "(1, 0, 294)",
            blank_ratio=self.last_prediction.blank_ratio if self.last_prediction else 0.0,
            forward_ms=round(self.last_forward_ms, 2),
            decode_ms=round(self.last_decode_ms, 2),
            nan_count=self.preprocessor.last_nan_count,
            inf_count=self.preprocessor.last_inf_count,
        )

        landmarks_diag = LandmarksDiagnostic(
            left_valid_ratio=round(self.preprocessor.last_left_valid_ratio, 2),
            right_valid_ratio=round(self.preprocessor.last_right_valid_ratio, 2),
            overall_valid_ratio=round(self.preprocessor.last_overall_valid_ratio, 2),
            has_left_hand=self.preprocessor.last_left_valid_ratio > 0.0,
            has_right_hand=self.preprocessor.last_right_valid_ratio > 0.0,
            in_signing_space=self.preprocessor.in_signing_space,
        )

        return IsolatedRecognitionMessage(
            type="isolated_recognition",
            mode="isolated",
            timestamp=time.time(),
            session_id=self.session_id,
            state=self.state.value,
            fps=30.0,
            total_ms=round(self.last_forward_ms + self.last_decode_ms, 2),
            motion_energy=0.0,
            live_gloss=[self.last_prediction.gloss] if (self.last_prediction and self.last_prediction.gloss) else [],
            landmarks=landmarks_diag,
            model=model_diag,
            prediction=self.last_prediction,
        )

    def infer_segment(self, segment_features: List[np.ndarray]) -> IsolatedPrediction:
        """Runs full sequence inference on a collected sign segment."""
        if len(segment_features) < self.min_frames:
            return IsolatedPrediction(
                gloss="",
                label_id=-1,
                top5=[],
                greedy_tokens=[],
                blank_ratio=1.0,
            )

        seq_np = np.stack(segment_features, axis=0).astype(np.float32)  # (T, 294)
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)  # (1, T, 294)

        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(seq_tensor)  # (1, T, 401)
        self.last_forward_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        # 1. Greedy CTC decode across entire sequence
        gloss_sequence, collapsed_tokens, raw_tokens, blank_ratio = self.decoder.decode_greedy(logits)

        # 2. Batched single-token CTC scoring for all 400 hypotheses (Top-5)
        top5_raw = self.decoder.decode_single_gloss_topk(logits, k=5)
        self.last_decode_ms = (time.perf_counter() - t0) * 1000.0

        top5_candidates = [
            TopCandidate(
                rank=item["rank"],
                label_id=item["label_id"],
                gloss=item["gloss"],
                log_score=item["log_score"],
                diagnostic_confidence=item["diagnostic_confidence"],
            )
            for item in top5_raw
        ]

        # Primary predicted gloss
        predicted_gloss = ""
        predicted_label_id = -1

        if len(gloss_sequence) > 0 and len(collapsed_tokens) > 0:
            predicted_gloss = gloss_sequence[0]
            predicted_label_id = collapsed_tokens[0]
        elif len(top5_candidates) > 0:
            # Fallback to Top-1 hypothesis if greedy produced pure blanks
            predicted_gloss = top5_candidates[0].gloss
            predicted_label_id = top5_candidates[0].label_id

        return IsolatedPrediction(
            gloss=predicted_gloss,
            label_id=predicted_label_id,
            top5=top5_candidates,
            greedy_tokens=collapsed_tokens,
            blank_ratio=round(blank_ratio, 4),
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        client_action: Optional[str] = None,
        draw_landmarks: bool = False,
    ) -> Tuple[IsolatedRecognitionMessage, np.ndarray]:
        """Processes a single frame in Isolated Mode."""
        t_start = time.perf_counter()
        now = time.time()

        # Update FPS tracking
        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        elapsed = self.frame_times[-1] - self.frame_times[0] if len(self.frame_times) > 1 else 0.0
        fps = round(len(self.frame_times) / elapsed, 1) if elapsed > 1e-4 else 0.0

        # Handle explicit client commands
        if client_action == "reset":
            self.reset()
        elif client_action == "start_record" and self.state in [IsolatedState.IDLE, IsolatedState.RESULT]:
            self.state = IsolatedState.RECORDING
            self.frames_buffer.clear()
            self.recording_start_time = now
            self.last_motion_time = now
            self.last_prediction = None
        elif client_action == "stop_record" and self.state == IsolatedState.RECORDING:
            self.state = IsolatedState.INFERRING

        # 1. Landmark extraction
        left_hand, right_hand, pose, annotated_frame = self.extractor.extract(
            frame_bgr, draw_landmarks=draw_landmarks
        )

        # 2. Strict Preprocessing
        frame_features, motion_energy = self.preprocessor.process_frame(
            left_hand_lms=left_hand,
            right_hand_lms=right_hand,
            pose_lms=pose,
        )

        # 3. State machine transitions
        if self.state == IsolatedState.IDLE:
            # Trigger recording when hand enters signing space with active motion
            if self.preprocessor.in_signing_space and motion_energy >= self.motion_start_threshold:
                self.state = IsolatedState.RECORDING
                self.frames_buffer = [frame_features]
                self.recording_start_time = now
                self.last_motion_time = now
                self.last_prediction = None

        elif self.state == IsolatedState.RECORDING:
            self.frames_buffer.append(frame_features)
            if motion_energy >= self.motion_continue_threshold and self.preprocessor.in_signing_space:
                self.last_motion_time = now

            # Check completion criteria:
            # - Minimum duration reached AND hands pause
            # - OR hands lowered outside signing space
            # - OR buffer reached maximum length
            if len(self.frames_buffer) >= self.min_frames:
                paused = (self.last_motion_time is not None) and ((now - self.last_motion_time) >= self.pause_timeout_sec)
                hands_lowered = (not self.preprocessor.in_signing_space) and (self.last_motion_time is not None) and ((now - self.last_motion_time) >= 0.35)
                buffer_full = len(self.frames_buffer) >= self.max_frames

                if paused or hands_lowered or buffer_full:
                    self.state = IsolatedState.INFERRING
            elif len(self.frames_buffer) < self.min_frames:
                # If motion dropped prematurely before minimum frames, abort back to IDLE
                if (self.last_motion_time is not None) and ((now - self.last_motion_time) > 1.0):
                    self.reset()

        if self.state == IsolatedState.INFERRING:
            self.last_prediction = self.infer_segment(self.frames_buffer)
            self.state = IsolatedState.RESULT

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Construct diagnostic payloads
        landmarks_diag = LandmarksDiagnostic(
            left_valid_ratio=round(self.preprocessor.last_left_valid_ratio, 2),
            right_valid_ratio=round(self.preprocessor.last_right_valid_ratio, 2),
            overall_valid_ratio=round(self.preprocessor.last_overall_valid_ratio, 2),
            has_left_hand=bool(left_hand and len(left_hand) == 21),
            has_right_hand=bool(right_hand and len(right_hand) == 21),
            in_signing_space=self.preprocessor.in_signing_space,
            left_hand=[NormalizedPoint(x=round(p["x"], 4), y=round(p["y"], 4), z=round(p.get("z", 0.0), 4)) for p in left_hand] if left_hand else None,
            right_hand=[NormalizedPoint(x=round(p["x"], 4), y=round(p["y"], 4), z=round(p.get("z", 0.0), 4)) for p in right_hand] if right_hand else None,
            pose=[NormalizedPoint(x=round(p["x"], 4), y=round(p["y"], 4), z=round(p.get("z", 0.0), 4)) for p in pose] if pose else None,
        )

        model_diag = ModelDiagnostic(
            buffer_frames=len(self.frames_buffer),
            input_shape=f"(1, {len(self.frames_buffer)}, 294)" if len(self.frames_buffer) > 0 else "(1, 0, 294)",
            blank_ratio=self.last_prediction.blank_ratio if self.last_prediction else 0.0,
            forward_ms=round(self.last_forward_ms, 2),
            decode_ms=round(self.last_decode_ms, 2),
            nan_count=self.preprocessor.last_nan_count,
            inf_count=self.preprocessor.last_inf_count,
        )

        message = IsolatedRecognitionMessage(
            type="isolated_recognition",
            mode="isolated",
            timestamp=now,
            session_id=self.session_id,
            state=self.state.value,
            fps=fps,
            total_ms=round(total_ms, 2),
            motion_energy=round(motion_energy, 4),
            live_gloss=[self.last_prediction.gloss] if (self.last_prediction and self.last_prediction.gloss) else [],
            landmarks=landmarks_diag,
            model=model_diag,
            prediction=self.last_prediction,
        )

        return message, annotated_frame

    def close(self):
        self.extractor.close()

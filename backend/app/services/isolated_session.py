"""Isolated Recognition Session (Mode A) for Single-Gloss Recognition.

Flow:
Webcam -> MediaPipe (off event loop) -> Strict Preprocessing -> Isolated Segment Buffer
-> Manual or Auto Sign Interval -> Model Inference on Full Segment (in thread pool)
-> Single-Gloss Prediction + Top-5 -> Output.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import logging
import time
import uuid
from typing import List, Optional, Tuple
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

# Dedicated single-worker executor for CPU-bound PyTorch inference
_isolated_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vsl_isolated")


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
        min_frames: int = 10,
        max_frames: int = 180,
        motion_start_threshold: float = 0.025,
        motion_continue_threshold: float = 0.008,
        pause_timeout_sec: float = 1.6,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.extractor = LandmarkExtractor()
        self._mp_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iso_mp")
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
        # True when recording was manually triggered by user (never auto-stop on pause/hands lowered)
        self.manual_recording: bool = False

        # Timing & Performance
        self.frame_times: List[float] = []
        self._current_fps: float = 0.0
        self.last_mediapipe_ms: float = 0.0
        self.last_preprocessing_ms: float = 0.0
        self.last_forward_ms: float = 0.0
        self.last_decode_ms: float = 0.0
        self._last_landmarks_diag: LandmarksDiagnostic = LandmarksDiagnostic()

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
        self.manual_recording = False

    def start_recording(self):
        """Manually trigger recording for isolated sign (manual mode: no auto-stop)."""
        self.state = IsolatedState.RECORDING
        self.frames_buffer.clear()
        now = time.time()
        self.recording_start_time = now
        self.last_motion_time = now
        self.last_prediction = None
        self.manual_recording = True

    def stop_recording(self):
        """Manually finish recording and trigger inference."""
        if self.state == IsolatedState.RECORDING:
            self.state = IsolatedState.INFERRING

    def _build_finalize_message(self) -> IsolatedRecognitionMessage:
        model_diag = ModelDiagnostic(
            buffer_frames=len(self.frames_buffer),
            input_shape=f"(1, {len(self.frames_buffer)}, 294)" if len(self.frames_buffer) > 0 else "(1, 0, 294)",
            blank_ratio=self.last_prediction.blank_ratio if self.last_prediction else 0.0,
            mediapipe_ms=round(self.last_mediapipe_ms, 2),
            preprocessing_ms=round(self.last_preprocessing_ms, 2),
            forward_ms=round(self.last_forward_ms, 2),
            decode_ms=round(self.last_decode_ms, 2),
            nan_count=self.preprocessor.last_nan_count,
            inf_count=self.preprocessor.last_inf_count,
        )

        return IsolatedRecognitionMessage(
            type="isolated_recognition",
            mode="isolated",
            timestamp=time.time(),
            session_id=self.session_id,
            state=self.state.value,
            fps=self._current_fps,
            total_ms=round(self.last_forward_ms + self.last_decode_ms, 2),
            motion_energy=0.0,
            live_gloss=[self.last_prediction.gloss] if (self.last_prediction and self.last_prediction.gloss) else [],
            landmarks=self._last_landmarks_diag,
            model=model_diag,
            prediction=self.last_prediction,
        )

    def finalize_and_infer(self) -> IsolatedRecognitionMessage:
        """Synchronously forces immediate inference on current buffer and returns message."""
        self.manual_recording = False
        self.last_prediction = self.infer_segment(list(self.frames_buffer))
        self.state = IsolatedState.RESULT
        return self._build_finalize_message()

    async def finalize_and_infer_async(self) -> IsolatedRecognitionMessage:
        """Asynchronously runs inference in executor without blocking the WebSocket event loop."""
        self.manual_recording = False
        segment_copy = list(self.frames_buffer)
        loop = asyncio.get_running_loop()
        self.last_prediction = await loop.run_in_executor(
            _isolated_executor,
            self.infer_segment,
            segment_copy,
        )
        self.state = IsolatedState.RESULT
        return self._build_finalize_message()

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

    async def process_frame(
        self,
        frame_bgr: np.ndarray,
        client_action: Optional[str] = None,
        draw_landmarks: bool = False,
    ) -> Tuple[IsolatedRecognitionMessage, np.ndarray]:
        """Processes a single frame in Isolated Mode (async - MediaPipe & PyTorch off event loop)."""
        t_start = time.perf_counter()
        now = time.time()

        # Exact sliding-window FPS tracking
        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        elapsed = self.frame_times[-1] - self.frame_times[0] if len(self.frame_times) > 1 else 0.0
        fps = round((len(self.frame_times) - 1) / elapsed, 1) if elapsed > 1e-4 else 0.0
        self._current_fps = fps

        # Handle explicit client commands
        if client_action == "reset":
            self.reset()
        elif client_action == "start_record" and self.state in [IsolatedState.IDLE, IsolatedState.RESULT]:
            self.start_recording()
        elif client_action == "stop_record" and self.state == IsolatedState.RECORDING:
            self.state = IsolatedState.INFERRING

        # 1. Landmark extraction (off event loop so WebSocket receiver never stalls)
        loop = asyncio.get_running_loop()
        t_mp = time.perf_counter()
        left_hand, right_hand, pose, annotated_frame = await loop.run_in_executor(
            self._mp_executor,
            self.extractor.extract,
            frame_bgr,
            draw_landmarks,
        )
        self.last_mediapipe_ms = (time.perf_counter() - t_mp) * 1000.0

        # 2. Strict Preprocessing
        t_prep = time.perf_counter()
        frame_features, motion_energy = self.preprocessor.process_frame(
            left_hand_lms=left_hand,
            right_hand_lms=right_hand,
            pose_lms=pose,
        )
        self.last_preprocessing_ms = (time.perf_counter() - t_prep) * 1000.0

        # 3. State machine transitions
        if self.state == IsolatedState.IDLE:
            # Auto-trigger recording only on clear active motion in signing space
            if self.preprocessor.in_signing_space and motion_energy >= self.motion_start_threshold:
                self.state = IsolatedState.RECORDING
                self.frames_buffer = [frame_features]
                self.recording_start_time = now
                self.last_motion_time = now
                self.last_prediction = None
                self.manual_recording = False

        elif self.state == IsolatedState.RECORDING:
            self.frames_buffer.append(frame_features)
            if motion_energy >= self.motion_continue_threshold and self.preprocessor.in_signing_space:
                self.last_motion_time = now

            buffer_full = len(self.frames_buffer) >= self.max_frames

            if self.manual_recording:
                # Manual mode: ONLY stop when user clicks Stop or buffer reaches max_frames (6s @ 30fps)
                if buffer_full:
                    self.state = IsolatedState.INFERRING
            else:
                # Auto mode: stop on sustained pause, hands lowered, or buffer full
                if len(self.frames_buffer) >= self.min_frames:
                    paused = (self.last_motion_time is not None) and ((now - self.last_motion_time) >= self.pause_timeout_sec)
                    hands_lowered = (not self.preprocessor.in_signing_space) and (self.last_motion_time is not None) and ((now - self.last_motion_time) >= 0.9)
                    if paused or hands_lowered or buffer_full:
                        self.state = IsolatedState.INFERRING
                elif (self.last_motion_time is not None) and ((now - self.last_motion_time) > 2.0):
                    self.reset()

        if self.state == IsolatedState.INFERRING:
            self.manual_recording = False
            self.last_prediction = await loop.run_in_executor(
                _isolated_executor,
                self.infer_segment,
                list(self.frames_buffer),
            )
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
        self._last_landmarks_diag = landmarks_diag

        model_diag = ModelDiagnostic(
            buffer_frames=len(self.frames_buffer),
            input_shape=f"(1, {len(self.frames_buffer)}, 294)" if len(self.frames_buffer) > 0 else "(1, 0, 294)",
            blank_ratio=self.last_prediction.blank_ratio if self.last_prediction else 0.0,
            mediapipe_ms=round(self.last_mediapipe_ms, 2),
            preprocessing_ms=round(self.last_preprocessing_ms, 2),
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
        self._mp_executor.shutdown(wait=False)

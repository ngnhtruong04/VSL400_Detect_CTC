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
        motion_start_threshold: float = 0.018,
        motion_continue_threshold: float = 0.008,
        pause_timeout_sec: float = 1.6,
        hands_absent_stop_sec: float = 0.45,
        min_inference_frames: int = 45,
        edge_hold_frames: int = 4,
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
        self.hands_absent_stop_sec = hands_absent_stop_sec
        # Training-parity temporal normalization (VSL400 clips: min=34, median=63 + EDGE_HOLD_RANGE=(2,5))
        self.min_inference_frames = min_inference_frames
        self.edge_hold_frames = edge_hold_frames

        # State machine
        self.state: IsolatedState = IsolatedState.IDLE
        self.frames_buffer: List[np.ndarray] = []
        self.motion_buffer: List[float] = []
        self.in_space_buffer: List[bool] = []
        self.recording_start_time: Optional[float] = None
        self.last_motion_time: Optional[float] = None
        self.last_prediction: Optional[IsolatedPrediction] = None
        # True when recording was manually triggered by user (auto-stops when both hands disappear after signing)
        self.manual_recording: bool = False
        # Hand-presence tracking for hands-lowered / skeleton-absent auto-stop & trailing-silence trimming
        self.has_seen_hands: bool = False
        self.hand_frames_count: int = 0
        self.hands_lost_time: Optional[float] = None
        self.last_hand_buffer_idx: int = 0
        self.last_model_input_frames: int = 0

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
        self.motion_buffer.clear()
        self.in_space_buffer.clear()
        self.preprocessor.reset_state()
        self.recording_start_time = None
        self.last_motion_time = None
        self.last_prediction = None
        self.last_forward_ms = 0.0
        self.last_decode_ms = 0.0
        self.manual_recording = False
        self.has_seen_hands = False
        self.hand_frames_count = 0
        self.hands_lost_time = None
        self.last_hand_buffer_idx = 0
        self.last_model_input_frames = 0

    def start_recording(self):
        """Manually trigger recording for isolated sign; auto-stops once hands appear and are then lowered out of frame."""
        self.state = IsolatedState.RECORDING
        self.frames_buffer.clear()
        self.motion_buffer.clear()
        self.in_space_buffer.clear()
        self.preprocessor.prev_hand_coords = None
        self.preprocessor.prev_hand_valid = None
        self.preprocessor.reset_motion_state()
        now = time.time()
        self.recording_start_time = now
        self.last_motion_time = now
        self.last_prediction = None
        self.manual_recording = True
        self.has_seen_hands = False
        self.hand_frames_count = 0
        self.hands_lost_time = None
        self.last_hand_buffer_idx = 0
        self.last_model_input_frames = 0

    def stop_recording(self):
        """Manually finish recording and trigger inference."""
        if self.state == IsolatedState.RECORDING:
            self.state = IsolatedState.INFERRING

    def _get_trimmed_segment(self) -> List[np.ndarray]:
        """Returns recorded frames trimmed of leading/trailing dead static tails while
        strictly preserving the full active signing span (including any internal static holds)."""
        base_end = self.last_hand_buffer_idx if self.last_hand_buffer_idx > 0 else len(self.frames_buffer)
        buf = self.frames_buffer[:base_end]
        n = len(buf)
        if n < self.min_frames or len(self.motion_buffer) < n or len(self.in_space_buffer) < n:
            return list(buf)

        motions = self.motion_buffer[:n]
        spaces = self.in_space_buffer[:n]
        active_idx = [i for i, m in enumerate(motions) if m >= self.motion_continue_threshold]
        # Static-posture sign fallback: preserve frames inside signing space if motion is minimal
        if len(active_idx) < 3:
            active_idx = [i for i, sp in enumerate(spaces) if sp]
        if not active_idx:
            return list(buf)

        # Pad start by 8 frames (compensates 5-frame motion filter lag) and end by 6 frames (endpoint hold).
        # Everything between active_idx[0] and active_idx[-1] (internal holds) is kept intact.
        start_idx = max(0, active_idx[0] - 8)
        end_idx = min(n, active_idx[-1] + 1 + 6)
        if (end_idx - start_idx) < self.min_frames:
            deficit = self.min_frames - (end_idx - start_idx)
            end_idx = min(n, end_idx + deficit)
            start_idx = max(0, end_idx - self.min_frames)
        return list(buf[start_idx:end_idx])

    def _prepare_inference_sequence(self, segment_features: List[np.ndarray]) -> np.ndarray:
        """Aligns isolated segment length & edge holds to VSL400 training distribution.

        1. If recorded clip is shorter than `min_inference_frames` (e.g. fast 15-30f sign vs
           VSL400 training min=34, median=63, TCN receptive field=33), performs validity-aware
           linear temporal resampling of hand xyz (strictly preserving missing-hand zeros) and
           recomputes frame-to-frame velocity dxyz at the normalized rate.
        2. Prepends & appends `edge_hold_frames` static endpoint holds (dxyz=0), matching
           EDGE_HOLD_RANGE=(2, 5) used in VSL400 training notebooks (NB05/NB08).
        """
        raw = np.stack(segment_features, axis=0).astype(np.float32)  # (T, 294)
        T = raw.shape[0]
        mat = raw.reshape(T, 42, 7)
        xyz = mat[:, :, :3].copy()
        valid = mat[:, :, 6] > 0.5

        if T < self.min_inference_frames and T >= 2:
            target_t = self.min_inference_frames
            u = np.linspace(0.0, float(T - 1), target_t, dtype=np.float32)
            i0 = np.floor(u).astype(np.int64)
            i1 = np.minimum(i0 + 1, T - 1)
            frac = u - i0
            alpha = frac[:, None, None]  # (target_t, 1, 1)

            v0 = valid[i0]  # (target_t, 42)
            v1 = valid[i1]
            both = v0 & v1

            near_idx = np.where(frac[:, None] < 0.5, i0[:, None], i1[:, None])
            j_idx = np.arange(42)[None, :]
            near_xyz = xyz[near_idx, j_idx]
            near_v = valid[near_idx, j_idx]

            interp_xyz = (1.0 - alpha) * xyz[i0] + alpha * xyz[i1]
            xyz = np.where(
                both[:, :, None],
                interp_xyz,
                np.where(near_v[:, :, None], near_xyz, 0.0),
            ).astype(np.float32)
            valid = np.where(both, True, near_v)

        if self.edge_hold_frames > 0:
            n_hold = self.edge_hold_frames
            xyz = np.concatenate(
                [
                    np.repeat(xyz[:1], n_hold, axis=0),
                    xyz,
                    np.repeat(xyz[-1:], n_hold, axis=0),
                ],
                axis=0,
            )
            valid = np.concatenate(
                [
                    np.repeat(valid[:1], n_hold, axis=0),
                    valid,
                    np.repeat(valid[-1:], n_hold, axis=0),
                ],
                axis=0,
            )

        # Strict NB08 velocity contract: vel[0]=0, vel[t]=xyz[t]-xyz[t-1] iff valid at both t and t-1
        vel = np.zeros_like(xyz, dtype=np.float32)
        if len(xyz) > 1:
            vel[1:] = xyz[1:] - xyz[:-1]
            pair_valid = valid[1:] & valid[:-1]
            vel[1:][~pair_valid] = 0.0

        xyz[~valid] = 0.0
        feat = np.concatenate(
            [xyz, vel, valid[:, :, None].astype(np.float32)],
            axis=-1,
        ).astype(np.float32)
        return feat.reshape(len(xyz), 294)

    def _build_finalize_message(self) -> IsolatedRecognitionMessage:
        input_t = self.last_model_input_frames if self.last_model_input_frames > 0 else len(self.frames_buffer)
        model_diag = ModelDiagnostic(
            buffer_frames=len(self.frames_buffer),
            input_shape=f"(1, {input_t}, 294)" if input_t > 0 else "(1, 0, 294)",
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
        if self.state == IsolatedState.RESULT and self.last_prediction is not None:
            return self._build_finalize_message()
        self.manual_recording = False
        segment = self._get_trimmed_segment()
        self.frames_buffer = segment
        self.last_prediction = self.infer_segment(segment)
        self.state = IsolatedState.RESULT
        return self._build_finalize_message()

    async def finalize_and_infer_async(self) -> IsolatedRecognitionMessage:
        """Asynchronously runs inference in executor without blocking the WebSocket event loop."""
        if self.state == IsolatedState.RESULT and self.last_prediction is not None:
            return self._build_finalize_message()
        self.manual_recording = False
        segment_copy = self._get_trimmed_segment()
        self.frames_buffer = segment_copy
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
            self.last_model_input_frames = 0
            return IsolatedPrediction(
                gloss="",
                label_id=-1,
                top5=[],
                greedy_tokens=[],
                blank_ratio=1.0,
            )

        seq_np = self._prepare_inference_sequence(segment_features)  # (T_eff, 294)
        self.last_model_input_frames = int(seq_np.shape[0])
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)  # (1, T_eff, 294)

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

        # 2. Strict Preprocessing (isotropic aspect-ratio correction for non-square webcams)
        t_prep = time.perf_counter()
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_aspect = (frame_h / float(frame_w)) if frame_w > 0 else 1.0
        frame_features, motion_energy = self.preprocessor.process_frame(
            left_hand_lms=left_hand,
            right_hand_lms=right_hand,
            pose_lms=pose,
            frame_aspect=frame_aspect,
        )
        self.last_preprocessing_ms = (time.perf_counter() - t_prep) * 1000.0

        has_left_hand = bool(left_hand and len(left_hand) == 21)
        has_right_hand = bool(right_hand and len(right_hand) == 21)
        has_any_hand = has_left_hand or has_right_hand

        # 3. State machine transitions
        if self.state == IsolatedState.IDLE:
            # Auto-trigger recording only on clear active motion in signing space
            if has_any_hand and self.preprocessor.in_signing_space and motion_energy >= self.motion_start_threshold:
                self.state = IsolatedState.RECORDING
                self.frames_buffer = [frame_features]
                self.motion_buffer = [motion_energy]
                self.in_space_buffer = [self.preprocessor.in_signing_space]
                self.recording_start_time = now
                self.last_motion_time = now
                self.last_prediction = None
                self.manual_recording = False
                self.has_seen_hands = True
                self.hand_frames_count = 1
                self.hands_lost_time = None
                self.last_hand_buffer_idx = 1

        elif self.state == IsolatedState.RECORDING:
            if has_any_hand:
                self.has_seen_hands = True
                self.hands_lost_time = None
                self.frames_buffer.append(frame_features)
                self.motion_buffer.append(motion_energy)
                self.in_space_buffer.append(self.preprocessor.in_signing_space)
                self.hand_frames_count += 1
                self.last_hand_buffer_idx = len(self.frames_buffer)
                if motion_energy >= self.motion_continue_threshold and self.preprocessor.in_signing_space:
                    self.last_motion_time = now
            elif self.has_seen_hands:
                # Hands lowered or temporarily lost after signing started
                self.frames_buffer.append(frame_features)
                self.motion_buffer.append(motion_energy)
                self.in_space_buffer.append(self.preprocessor.in_signing_space)
                if self.hands_lost_time is None:
                    self.hands_lost_time = now
            # If hands have not appeared yet after clicking Start Record, wait without buffering empty frames

            buffer_full = len(self.frames_buffer) >= self.max_frames
            hands_absent_elapsed = (now - self.hands_lost_time) if self.hands_lost_time is not None else 0.0
            hands_gone = (
                self.has_seen_hands
                and (not has_any_hand)
                and (hands_absent_elapsed >= self.hands_absent_stop_sec)
            )

            if self.manual_recording:
                # Manual mode: auto-stop once both hands disappear (no hand skeleton) after signing,
                # or when user clicks Stop, or safety cap max_frames (180 frames) is reached.
                if buffer_full:
                    self.state = IsolatedState.INFERRING
                elif hands_gone:
                    if self.hand_frames_count >= self.min_frames:
                        self.state = IsolatedState.INFERRING
                    else:
                        # Ignore brief accidental hand flash (< min_frames) while moving off mouse; keep waiting
                        self.frames_buffer.clear()
                        self.motion_buffer.clear()
                        self.in_space_buffer.clear()
                        self.has_seen_hands = False
                        self.hand_frames_count = 0
                        self.hands_lost_time = None
                        self.last_hand_buffer_idx = 0
            else:
                # Auto mode: stop when hands disappear, hands lowered outside signing space, sustained pause, or buffer full
                if self.hand_frames_count >= self.min_frames:
                    paused = (self.last_motion_time is not None) and ((now - self.last_motion_time) >= self.pause_timeout_sec)
                    hands_lowered = (not self.preprocessor.in_signing_space) and (self.last_motion_time is not None) and ((now - self.last_motion_time) >= 0.9)
                    if hands_gone or paused or hands_lowered or buffer_full:
                        self.state = IsolatedState.INFERRING
                elif hands_gone or ((self.last_motion_time is not None) and ((now - self.last_motion_time) > 2.0)):
                    self.reset()

        if self.state == IsolatedState.INFERRING:
            self.manual_recording = False
            segment_to_infer = self._get_trimmed_segment()
            self.frames_buffer = segment_to_infer
            self.last_prediction = await loop.run_in_executor(
                _isolated_executor,
                self.infer_segment,
                segment_to_infer,
            )
            self.state = IsolatedState.RESULT

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Construct diagnostic payloads
        landmarks_diag = LandmarksDiagnostic(
            left_valid_ratio=round(self.preprocessor.last_left_valid_ratio, 2),
            right_valid_ratio=round(self.preprocessor.last_right_valid_ratio, 2),
            overall_valid_ratio=round(self.preprocessor.last_overall_valid_ratio, 2),
            has_left_hand=has_left_hand,
            has_right_hand=has_right_hand,
            in_signing_space=self.preprocessor.in_signing_space,
            left_hand=[NormalizedPoint(x=round(p["x"], 4), y=round(p["y"], 4), z=round(p.get("z", 0.0), 4)) for p in left_hand] if left_hand else None,
            right_hand=[NormalizedPoint(x=round(p["x"], 4), y=round(p["y"], 4), z=round(p.get("z", 0.0), 4)) for p in right_hand] if right_hand else None,
            pose=[NormalizedPoint(x=round(p["x"], 4), y=round(p["y"], 4), z=round(p.get("z", 0.0), 4)) for p in pose] if pose else None,
        )
        self._last_landmarks_diag = landmarks_diag

        input_t = (
            self.last_model_input_frames
            if (self.state == IsolatedState.RESULT and self.last_model_input_frames > 0)
            else len(self.frames_buffer)
        )
        model_diag = ModelDiagnostic(
            buffer_frames=len(self.frames_buffer),
            input_shape=f"(1, {input_t}, 294)" if input_t > 0 else "(1, 0, 294)",
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

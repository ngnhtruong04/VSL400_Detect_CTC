"""Continuous Recognition Session (Mode B) for Multi-Gloss Continuous Signing.

Key Architecture (30 FPS Lock):
1. Buffer Separation:
   - recent_window: rolling 90 frames for real-time motion energy and pre-roll.
   - utterance_buffer: complete contiguous sequence from start of sign to end of utterance.
2. Strategy 1 (Full Active Utterance Re-Decode, Non-Blocking):
   - Live periodic PyTorch inference runs asynchronously in a background task so the 30 FPS
     landmark stream NEVER stalls waiting for model forward pass.
   - Final utterance commitment awaits exact full-utterance inference & language post-processing.
3. Stable Token Prefix Tracking:
   - Tokens confirmed across multiple updates are marked stable.
4. Token Duplication:
   - Preserves repeating identical glosses separated by CTC blanks.
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import time
import uuid
from typing import List, Optional, Tuple
import numpy as np
import torch

from backend.app.config.settings import settings
from backend.app.ml.landmark_extractor import LandmarkExtractor
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.ml.ctc_decoder import CTCDecoder
from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState
from backend.app.ml.model_loader import ModelLoader
from backend.app.services.language_service import LanguageService
from backend.app.schemas.api_models import (
    ContinuousRecognitionMessage,
    ContinuousStatePayload,
    LandmarksDiagnostic,
    ModelDiagnostic,
    NormalizedPoint,
)

logger = logging.getLogger("vsl.continuous_session")

# Dedicated single-worker executor for CPU-bound PyTorch inference
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vsl_infer")


class ContinuousRecognitionSession:
    """Manages an isolated continuous sign language recognition session."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        decoder: Optional[CTCDecoder] = None,
        language_service: Optional[LanguageService] = None,
        recent_window_size: Optional[int] = None,
        min_utterance_frames: Optional[int] = None,
        max_utterance_frames: Optional[int] = None,
        inference_stride: Optional[int] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.extractor = LandmarkExtractor()
        self._mp_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cont_mp")
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
        self.fsm = EndpointFSM(
            motion_threshold=settings.MOTION_THRESHOLD,
            min_sign_duration_ms=settings.MIN_SIGN_DURATION_MS,
            possible_end_delay_ms=settings.POSSIBLE_END_DELAY_MS,
            silence_ms=settings.SILENCE_MS,
            cooldown_ms=settings.COOLDOWN_MS,
            hands_lowered_possible_end_ms=settings.HANDS_LOWERED_POSSIBLE_END_MS,
            hands_lowered_commit_ms=settings.HANDS_LOWERED_COMMIT_MS,
            starting_abort_ms=settings.STARTING_ABORT_MS,
        )
        self.language_service = language_service or LanguageService()
        self.model, self.device = ModelLoader.get_loaded_model()
        if (
            self.device.type == "cpu"
            and settings.CPU_NUM_THREADS > 0
            and torch.get_num_threads() != settings.CPU_NUM_THREADS
        ):
            torch.set_num_threads(settings.CPU_NUM_THREADS)

        # Buffer separation (synchronized with config/runtime.yaml via settings)
        eff_recent_window = int(recent_window_size if recent_window_size is not None else settings.BUFFER_MAX_FRAMES)
        self.recent_window: deque = deque(maxlen=eff_recent_window)
        self.recent_motion_window: deque = deque(maxlen=eff_recent_window)
        self.recent_space_window: deque = deque(maxlen=eff_recent_window)
        self.utterance_buffer: List[np.ndarray] = []
        self.utterance_motion_buffer: List[float] = []
        self.utterance_space_buffer: List[bool] = []
        self._last_active_buf_len: int = 0
        self.min_utterance_frames = int(min_utterance_frames if min_utterance_frames is not None else settings.BUFFER_MIN_FRAMES)
        self.max_utterance_frames = int(max_utterance_frames if max_utterance_frames is not None else settings.BUFFER_MAX_UTTERANCE_FRAMES)
        self.inference_stride = int(inference_stride if inference_stride is not None else settings.INFERENCE_STRIDE)

        # Predictions & Token stability
        self.live_tokens: List[str] = []
        self.stable_tokens: List[str] = []
        self.final_tokens: List[str] = []
        self.final_sentence: str = ""
        self.last_raw_tokens: List[int] = []
        self.last_collapsed_tokens: List[int] = []
        self.last_blank_ratio: float = 1.0

        # Background non-blocking live inference state
        self._epoch: int = 0
        self._live_infer_task: Optional[asyncio.Task] = None

        # Prefix stability tracker
        self.token_history_window: List[List[str]] = []

        # Diagnostics & FPS
        self.frame_count: int = 0
        self.frame_times: List[float] = []
        self._current_fps: float = 0.0
        self.last_mediapipe_ms: float = 0.0
        self.last_preprocessing_ms: float = 0.0
        self.last_forward_ms: float = 0.0
        self.last_decode_ms: float = 0.0
        self.endpoint_reason: Optional[str] = None
        self._last_landmarks_diag: LandmarksDiagnostic = LandmarksDiagnostic()

        # Structured logging setup
        self.debug_log_path: Optional[Path] = None
        if settings.DEBUG_CONTINUOUS:
            log_dir = settings.PROJECT_ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.debug_log_path = log_dir / f"continuous_debug_{int(time.time())}_{self.session_id[:8]}.jsonl"
            logger.info(f"Continuous debug logging enabled: {self.debug_log_path}")

    def update_sensitivity(self, sensitivity: str):
        """Dynamically adjusts Endpoint FSM sensitivity ('low' | 'medium' | 'high')."""
        lvl = (sensitivity or "medium").lower().strip()
        if lvl == "low":
            self.fsm.possible_end_delay_sec = 1.6
            self.fsm.silence_sec = 3.2
            self.fsm.hands_lowered_possible_end_sec = 0.9
            self.fsm.hands_lowered_commit_sec = 1.8
            self.fsm.motion_threshold = 0.006
        elif lvl == "high":
            self.fsm.possible_end_delay_sec = 0.8
            self.fsm.silence_sec = 1.6
            self.fsm.hands_lowered_possible_end_sec = 0.45
            self.fsm.hands_lowered_commit_sec = 0.85
            self.fsm.motion_threshold = 0.012
        else:
            self.fsm.possible_end_delay_sec = settings.POSSIBLE_END_DELAY_MS / 1000.0
            self.fsm.silence_sec = settings.SILENCE_MS / 1000.0
            self.fsm.hands_lowered_possible_end_sec = settings.HANDS_LOWERED_POSSIBLE_END_MS / 1000.0
            self.fsm.hands_lowered_commit_sec = settings.HANDS_LOWERED_COMMIT_MS / 1000.0
            self.fsm.motion_threshold = settings.MOTION_THRESHOLD

    def update_llm_config(self, enabled: Optional[bool] = None, provider: Optional[str] = None):
        """Dynamically updates LanguageService settings from the web client."""
        if enabled is not None:
            self.language_service.enabled = bool(enabled)
        if provider is not None:
            self.language_service.provider = str(provider).lower().strip()

    def _get_trimmed_utterance_array(self) -> Optional[np.ndarray]:
        """Returns active utterance frames trimmed of leading/trailing dead static tails
        while strictly preserving the complete active span (including internal static holds
        and inter-sign pauses in multi-gloss / repeated utterances)."""
        n = len(self.utterance_buffer)
        if n < self.min_utterance_frames:
            return None

        if len(self.utterance_motion_buffer) == n and len(self.utterance_space_buffer) == n:
            trim_thresh = min(0.008, float(self.fsm.motion_threshold))
            active_idx = [i for i, m in enumerate(self.utterance_motion_buffer) if m >= trim_thresh]
            if len(active_idx) < 3:
                active_idx = [i for i, sp in enumerate(self.utterance_space_buffer) if sp]
            if active_idx:
                start_idx = max(0, active_idx[0] - 8)
                end_idx = min(n, active_idx[-1] + 1 + 6)
                if (end_idx - start_idx) < self.min_utterance_frames:
                    deficit = self.min_utterance_frames - (end_idx - start_idx)
                    end_idx = min(n, end_idx + deficit)
                    start_idx = max(0, end_idx - self.min_utterance_frames)
                return np.stack(self.utterance_buffer[start_idx:end_idx], axis=0).astype(np.float32)

        # Fallback if buffer was populated externally without parallel motion tracking
        if self._last_active_buf_len >= self.min_utterance_frames:
            end_idx = min(n, self._last_active_buf_len + 10)
        else:
            end_idx = n
        return np.stack(self.utterance_buffer[:end_idx], axis=0).astype(np.float32)

    async def commit_utterance_async(self) -> ContinuousRecognitionMessage:
        """Manually finalizes and translates the active utterance immediately."""
        loop = asyncio.get_running_loop()
        now = time.time()
        self.endpoint_reason = "manual_commit"

        seq_np = self._get_trimmed_utterance_array()
        if seq_np is not None:
            (
                (gloss_sequence, collapsed_tokens, raw_tokens, blank_ratio),
                forward_ms,
                decode_ms,
            ) = await loop.run_in_executor(_inference_executor, self._infer_and_decode_sync, seq_np)
            self.last_forward_ms = forward_ms
            self.last_decode_ms = decode_ms
            self.last_raw_tokens = list(raw_tokens)
            self.last_collapsed_tokens = list(collapsed_tokens)
            self.last_blank_ratio = blank_ratio
            self.final_tokens = list(gloss_sequence) if len(gloss_sequence) > 0 else list(self.live_tokens)
        else:
            self.final_tokens = list(self.live_tokens)

        self.final_sentence = await self.language_service.to_natural_sentence(self.final_tokens)
        logger.info(f"Utterance manually finalized: tokens={self.final_tokens}, sentence='{self.final_sentence}'")

        # Reset active utterance buffer & FSM to IDLE for next continuous sentence
        self._epoch += 1
        if self._live_infer_task and not self._live_infer_task.done():
            self._live_infer_task.cancel()
        self._live_infer_task = None
        buf_len = len(self.utterance_buffer)
        self.utterance_buffer.clear()
        self.utterance_motion_buffer.clear()
        self.utterance_space_buffer.clear()
        self._last_active_buf_len = 0
        self.live_tokens = []
        self.stable_tokens = []
        self.token_history_window.clear()
        self.fsm.reset()

        model_diag = ModelDiagnostic(
            buffer_frames=buf_len,
            input_shape=f"(1, {buf_len}, 294)" if buf_len > 0 else "(1, 0, 294)",
            blank_ratio=round(self.last_blank_ratio, 4),
            mediapipe_ms=round(self.last_mediapipe_ms, 2),
            preprocessing_ms=round(self.last_preprocessing_ms, 2),
            forward_ms=round(self.last_forward_ms, 2),
            decode_ms=round(self.last_decode_ms, 2),
            nan_count=self.preprocessor.last_nan_count,
            inf_count=self.preprocessor.last_inf_count,
        )
        cont_payload = ContinuousStatePayload(
            fsm_state=FSMState.END_OF_UTTERANCE.value,
            live_tokens=[],
            stable_tokens=[],
            final_tokens=self.final_tokens,
            raw_token_ids=self.last_raw_tokens,
            collapsed_token_ids=self.last_collapsed_tokens,
            utterance_frames=buf_len,
            recent_frames=len(self.recent_window),
            endpoint_reason=self.endpoint_reason,
            final_sentence=self.final_sentence,
        )
        return ContinuousRecognitionMessage(
            type="continuous_recognition",
            mode="continuous",
            timestamp=now,
            session_id=self.session_id,
            state=FSMState.END_OF_UTTERANCE.value,
            fps=self._current_fps,
            total_ms=round(self.last_forward_ms + self.last_decode_ms, 2),
            motion_energy=0.0,
            live_gloss=[],
            landmarks=self._last_landmarks_diag,
            model=model_diag,
            continuous=cont_payload,
        )

    def reset(self):
        """Resets all session state, buffers, and FSM."""
        self._epoch += 1
        if self._live_infer_task and not self._live_infer_task.done():
            self._live_infer_task.cancel()
        self._live_infer_task = None
        self.preprocessor.reset_state()
        self.recent_window.clear()
        self.recent_motion_window.clear()
        self.recent_space_window.clear()
        self.utterance_buffer.clear()
        self.utterance_motion_buffer.clear()
        self.utterance_space_buffer.clear()
        self._last_active_buf_len = 0
        self.fsm.reset()
        self.live_tokens = []
        self.stable_tokens = []
        self.final_tokens = []
        self.final_sentence = ""
        self.last_raw_tokens = []
        self.last_collapsed_tokens = []
        self.last_blank_ratio = 1.0
        self.token_history_window.clear()
        self.endpoint_reason = None
        self.last_forward_ms = 0.0
        self.last_decode_ms = 0.0

    def _update_stable_prefix(self, current_live: List[str]):
        """Tracks tokens that remain consistent across multiple decoding updates."""
        self.token_history_window.append(list(current_live))
        if len(self.token_history_window) > 3:
            self.token_history_window.pop(0)

        if len(self.token_history_window) >= 2:
            common_prefix = []
            min_len = min(len(hist) for hist in self.token_history_window)
            for i in range(min_len):
                tokens_at_i = [hist[i] for hist in self.token_history_window]
                if all(t == tokens_at_i[0] for t in tokens_at_i):
                    common_prefix.append(tokens_at_i[0])
                else:
                    break
            self.stable_tokens = common_prefix
        else:
            self.stable_tokens = list(current_live)

    def _infer_and_decode_sync(self, seq_np: np.ndarray):
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(seq_tensor)
        forward_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        result = self.decoder.decode_greedy(logits)
        decode_ms = (time.perf_counter() - t1) * 1000.0
        return result, forward_ms, decode_ms

    async def _run_live_inference_bg(self, seq_np: np.ndarray, epoch: int):
        """Non-blocking background task for periodic live decoding (keeps landmarks at 30 FPS)."""
        try:
            loop = asyncio.get_running_loop()
            (
                (gloss_sequence, collapsed_tokens, raw_tokens, blank_ratio),
                forward_ms,
                decode_ms,
            ) = await loop.run_in_executor(_inference_executor, self._infer_and_decode_sync, seq_np)

            # Discard result if session was reset or utterance ended while inferring
            if epoch != self._epoch:
                return

            self.last_forward_ms = forward_ms
            self.last_decode_ms = decode_ms
            self.live_tokens = list(gloss_sequence)
            self.last_raw_tokens = list(raw_tokens)
            self.last_collapsed_tokens = list(collapsed_tokens)
            self.last_blank_ratio = blank_ratio
            self._update_stable_prefix(self.live_tokens)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning(f"Background live inference error: {e}")

    async def process_frame(
        self,
        frame_bgr: np.ndarray,
        client_action: Optional[str] = None,
        draw_landmarks: bool = False,
    ) -> Tuple[ContinuousRecognitionMessage, np.ndarray]:
        """Processes a single frame in Continuous Mode (Mode B) without stalling landmark FPS."""
        t_start = time.perf_counter()
        now = time.time()
        self.frame_count += 1

        # Exact sliding-window FPS tracking
        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        elapsed = self.frame_times[-1] - self.frame_times[0] if len(self.frame_times) > 1 else 0.0
        fps = round((len(self.frame_times) - 1) / elapsed, 1) if elapsed > 1e-4 else 0.0
        self._current_fps = fps

        # Handle explicit client reset
        if client_action == "reset":
            self.reset()

        # 1. MediaPipe landmark extraction (off event loop)
        loop = asyncio.get_running_loop()
        t_mp = time.perf_counter()
        left_hand, right_hand, pose, annotated_frame = await loop.run_in_executor(
            self._mp_executor,
            self.extractor.extract,
            frame_bgr,
            draw_landmarks,
        )
        self.last_mediapipe_ms = (time.perf_counter() - t_mp) * 1000.0

        has_hands = (left_hand is not None and len(left_hand) == 21) or (
            right_hand is not None and len(right_hand) == 21
        )

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

        # Append to recent rolling window (for diagnostics & pre-roll)
        self.recent_window.append(frame_features)
        self.recent_motion_window.append(motion_energy)
        self.recent_space_window.append(self.preprocessor.in_signing_space)

        # 3. Endpoint FSM update
        prev_fsm_state = self.fsm.state
        current_state, is_final_commit = self.fsm.update(
            motion_energy=motion_energy,
            has_hands=has_hands,
            blank_ratio=self.last_blank_ratio,
            live_gloss=self.live_tokens,
            now=now,
            in_signing_space=self.preprocessor.in_signing_space,
        )

        # 4. Utterance Buffer Management
        if prev_fsm_state == FSMState.IDLE and current_state == FSMState.STARTING:
            self._epoch += 1
            # 12-frame pre-roll (~0.4-0.48s) preserves gentle sign onset after 5-frame motion filter lag
            pre_roll = list(self.recent_window)[-12:]
            pre_roll_m = list(self.recent_motion_window)[-12:]
            pre_roll_sp = list(self.recent_space_window)[-12:]
            self.utterance_buffer = list(pre_roll)
            self.utterance_motion_buffer = list(pre_roll_m)
            self.utterance_space_buffer = list(pre_roll_sp)
            self._last_active_buf_len = len(self.utterance_buffer)
            self.live_tokens = []
            self.stable_tokens = []
            self.token_history_window.clear()
            self.final_tokens = []
            self.final_sentence = ""
            self.endpoint_reason = None

        elif current_state in [FSMState.STARTING, FSMState.SIGNING, FSMState.POSSIBLE_END]:
            if len(self.utterance_buffer) < self.max_utterance_frames:
                self.utterance_buffer.append(frame_features)
                self.utterance_motion_buffer.append(motion_energy)
                self.utterance_space_buffer.append(self.preprocessor.in_signing_space)
            if current_state in [FSMState.STARTING, FSMState.SIGNING]:
                self._last_active_buf_len = len(self.utterance_buffer)
            # Safe buffer-cap handling (FIX-09): cleanly finalize utterance at cap rather than silently dropping frames
            if (
                current_state in [FSMState.SIGNING, FSMState.POSSIBLE_END]
                and len(self.utterance_buffer) >= self.max_utterance_frames
                and not is_final_commit
            ):
                is_final_commit = True
                current_state = FSMState.END_OF_UTTERANCE
                self.fsm._set_state(FSMState.END_OF_UTTERANCE, now)
                self.endpoint_reason = "max_buffer_reached"

        elif current_state == FSMState.IDLE:
            if prev_fsm_state in [FSMState.COOLDOWN, FSMState.STARTING]:
                self.utterance_buffer.clear()
                self.utterance_motion_buffer.clear()
                self.utterance_space_buffer.clear()
                self._last_active_buf_len = 0

        # 5. Full Active Utterance Inference (Non-Blocking Background Task, trimmed of trailing silence)
        should_run_inference = (
            current_state in [FSMState.SIGNING, FSMState.POSSIBLE_END]
            and len(self.utterance_buffer) >= self.min_utterance_frames
            and (self.frame_count % self.inference_stride == 0)
            and (self._live_infer_task is None or self._live_infer_task.done())
        )

        if should_run_inference:
            seq_np = self._get_trimmed_utterance_array()
            if seq_np is not None:
                self._live_infer_task = asyncio.create_task(
                    self._run_live_inference_bg(seq_np, self._epoch)
                )

        # 6. Final Utterance Commitment (uses trimmed active utterance — drops excess POSSIBLE_END silence)
        if is_final_commit:
            if self.endpoint_reason != "max_buffer_reached":
                self.endpoint_reason = "silence_timeout_reached"
            seq_np = self._get_trimmed_utterance_array()
            if seq_np is not None:
                (
                    (gloss_sequence, collapsed_tokens, raw_tokens, blank_ratio),
                    forward_ms,
                    decode_ms,
                ) = await loop.run_in_executor(_inference_executor, self._infer_and_decode_sync, seq_np)
                self.last_forward_ms = forward_ms
                self.last_decode_ms = decode_ms
                self.last_raw_tokens = list(raw_tokens)
                self.last_collapsed_tokens = list(collapsed_tokens)
                self.last_blank_ratio = blank_ratio
                self.final_tokens = list(gloss_sequence) if len(gloss_sequence) > 0 else list(self.live_tokens)
            else:
                self.final_tokens = list(self.live_tokens)

            # Optional natural language translation
            self.final_sentence = await self.language_service.to_natural_sentence(self.final_tokens)
            logger.info(f"Utterance finalized: tokens={self.final_tokens}, sentence='{self.final_sentence}'")

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Structured diagnostic logging to file if configured
        if self.debug_log_path:
            log_entry = {
                "timestamp": now,
                "frame_count": self.frame_count,
                "utterance_size": len(self.utterance_buffer),
                "recent_size": len(self.recent_window),
                "motion_energy": round(motion_energy, 4),
                "fsm_state": current_state.value,
                "blank_ratio": round(self.last_blank_ratio, 4),
                "live_tokens": self.live_tokens,
                "stable_tokens": self.stable_tokens,
                "final_tokens": self.final_tokens,
                "raw_tokens_sample": self.last_raw_tokens[:20] if self.last_raw_tokens else [],
                "collapsed_tokens": self.last_collapsed_tokens,
                "endpoint_reason": self.endpoint_reason,
            }
            try:
                with open(self.debug_log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write continuous debug log: {e}")

        # Construct message payload
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
            buffer_frames=len(self.utterance_buffer),
            input_shape=f"(1, {len(self.utterance_buffer)}, 294)" if len(self.utterance_buffer) > 0 else "(1, 0, 294)",
            blank_ratio=round(self.last_blank_ratio, 4),
            mediapipe_ms=round(self.last_mediapipe_ms, 2),
            preprocessing_ms=round(self.last_preprocessing_ms, 2),
            forward_ms=round(self.last_forward_ms, 2),
            decode_ms=round(self.last_decode_ms, 2),
            nan_count=self.preprocessor.last_nan_count,
            inf_count=self.preprocessor.last_inf_count,
        )

        cont_payload = ContinuousStatePayload(
            fsm_state=current_state.value,
            live_tokens=self.live_tokens,
            stable_tokens=self.stable_tokens,
            final_tokens=self.final_tokens,
            raw_token_ids=self.last_raw_tokens,
            collapsed_token_ids=self.last_collapsed_tokens,
            utterance_frames=len(self.utterance_buffer),
            recent_frames=len(self.recent_window),
            endpoint_reason=self.endpoint_reason,
            final_sentence=self.final_sentence,
        )

        message = ContinuousRecognitionMessage(
            type="continuous_recognition",
            mode="continuous",
            timestamp=now,
            session_id=self.session_id,
            state=current_state.value,
            fps=fps,
            total_ms=round(total_ms, 2),
            motion_energy=round(motion_energy, 4),
            live_gloss=list(self.live_tokens),
            landmarks=landmarks_diag,
            model=model_diag,
            continuous=cont_payload,
        )

        return message, annotated_frame

    def close(self):
        self._epoch += 1
        if self._live_infer_task and not self._live_infer_task.done():
            self._live_infer_task.cancel()
        self.extractor.close()
        self._mp_executor.shutdown(wait=False)

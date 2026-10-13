"""Session manager ensuring strict per-client state isolation for realtime recognition."""

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from backend.app.config.settings import settings
from backend.app.ml.landmark_extractor import LandmarkExtractor
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.ml.sequence_buffer import RealtimeSequenceBuffer
from backend.app.ml.ctc_decoder import CTCDecoder
from backend.app.ml.endpoint_fsm import EndpointFSM, FSMState
from backend.app.ml.model_loader import ModelLoader
from backend.app.services.language_service import LanguageService
from backend.app.schemas.api_models import (
    MetricsData,
    RecognitionMessage,
    UtteranceCompleteMessage,
    LandmarksPayload,
    NormalizedPoint,
)

logger = logging.getLogger("vsl.realtime_session")


class RealtimeSession:
    """Encapsulates isolated state for a single WebSocket client connection."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        decoder: Optional[CTCDecoder] = None,
        language_service: Optional[LanguageService] = None,
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
        self.buffer = RealtimeSequenceBuffer(
            max_frames=settings.BUFFER_MAX_FRAMES,
            min_frames=settings.BUFFER_MIN_FRAMES,
            inference_stride=settings.INFERENCE_STRIDE,
            feature_dim=settings.INPUT_DIM,
        )
        self.fsm = EndpointFSM(
            motion_threshold=settings.MOTION_THRESHOLD,
            min_sign_duration_ms=settings.MIN_SIGN_DURATION_MS,
            possible_end_delay_ms=settings.POSSIBLE_END_DELAY_MS,
            silence_ms=settings.SILENCE_MS,
            cooldown_ms=settings.COOLDOWN_MS,
        )

        self.decoder = decoder or CTCDecoder(
            label_map_path=settings.LABEL_MAP_PATH,
            blank_id=settings.BLANK_ID,
        )
        self.language_service = language_service or LanguageService()

        self.model, self.device = ModelLoader.get_loaded_model()

        # Session metrics and prediction cache
        self.last_live_gloss: List[str] = []
        self.last_raw_tokens: List[int] = []
        self.last_collapsed_tokens: List[int] = []
        self.last_blank_ratio: float = 1.0

        # Frame rate tracking
        self.frame_times = []

    def reset(self):
        """Resets temporal session buffers without tearing down components."""
        self.preprocessor.reset_state()
        self.buffer.clear()
        self.fsm.reset()
        self.last_live_gloss = []
        self.last_raw_tokens = []
        self.last_collapsed_tokens = []
        self.last_blank_ratio = 1.0

    async def process_frame(
        self,
        frame_bgr: np.ndarray,
        draw_landmarks: bool = False,
    ) -> Tuple[Optional[RecognitionMessage], Optional[UtteranceCompleteMessage], np.ndarray]:
        """Runs the complete inference and FSM pipeline on an incoming video frame.
        Returns:
            (live_recognition_msg, final_utterance_msg, annotated_frame)
        """
        t_start = time.perf_counter()

        # 1. MediaPipe landmark extraction
        t0 = time.perf_counter()
        left_hand, right_hand, pose, annotated_frame = self.extractor.extract(
            frame_bgr, draw_landmarks=draw_landmarks
        )
        mediapipe_ms = (time.perf_counter() - t0) * 1000.0

        has_hands = (left_hand is not None) or (right_hand is not None)

        # 2. Body-centric preprocessing and feature construction
        t0 = time.perf_counter()
        frame_features, motion_energy = self.preprocessor.process_frame(
            left_hand_lms=left_hand,
            right_hand_lms=right_hand,
            pose_lms=pose,
        )
        preprocessing_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Temporal rolling sequence buffer
        self.buffer.append(frame_features)

        # 4. Model inference & CTC decode
        inference_ms = 0.0
        decode_ms = 0.0

        if self.buffer.should_infer():
            t0 = time.perf_counter()
            seq_tensor = self.buffer.get_sequence_tensor(self.device)  # (1, T, 294)

            with torch.inference_mode():
                logits = self.model(seq_tensor)  # (1, T, 401)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # 5. Greedy CTC decoding
            t0 = time.perf_counter()
            (
                self.last_live_gloss,
                self.last_collapsed_tokens,
                self.last_raw_tokens,
                self.last_blank_ratio,
            ) = self.decoder.decode_greedy(logits)
            decode_ms = (time.perf_counter() - t0) * 1000.0

        # 6. Endpoint FSM update
        now = time.time()
        prev_fsm_state = self.fsm.state
        current_state, is_final_commit = self.fsm.update(
            motion_energy=motion_energy,
            has_hands=has_hands,
            blank_ratio=self.last_blank_ratio,
            live_gloss=self.last_live_gloss,
            now=now,
            in_signing_space=self.preprocessor.in_signing_space,
        )

        # When transitioning from IDLE to STARTING, clear stale idle zeros so sequence buffer focuses on active sign
        if prev_fsm_state == FSMState.IDLE and current_state == FSMState.STARTING:
            self.buffer.trim_to_tail(5)

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Compute smoothed FPS
        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        elapsed = self.frame_times[-1] - self.frame_times[0] if len(self.frame_times) > 1 else 0.0
        fps = len(self.frame_times) / elapsed if elapsed > 1e-4 else 0.0

        metrics = MetricsData(
            fps=round(fps, 1),
            mediapipe_ms=round(mediapipe_ms, 2),
            preprocessing_ms=round(preprocessing_ms, 2),
            inference_ms=round(inference_ms, 2),
            decode_ms=round(decode_ms, 2),
            total_ms=round(total_ms, 2),
            motion_energy=round(motion_energy, 4),
            blank_ratio=round(self.last_blank_ratio, 3),
            buffer_frames=len(self.buffer),
        )

        # 7. Construct landmarks payload for real-time skeleton visualization
        landmarks_payload = LandmarksPayload(
            left_hand=[
                NormalizedPoint(x=round(lm["x"], 4), y=round(lm["y"], 4), z=round(lm.get("z", 0.0), 4))
                for lm in left_hand
            ] if left_hand else None,
            right_hand=[
                NormalizedPoint(x=round(lm["x"], 4), y=round(lm["y"], 4), z=round(lm.get("z", 0.0), 4))
                for lm in right_hand
            ] if right_hand else None,
            pose=[
                NormalizedPoint(x=round(lm["x"], 4), y=round(lm["y"], 4), z=round(lm.get("z", 0.0), 4))
                for lm in pose
            ] if pose else None,
            has_left_hand=(left_hand is not None and len(left_hand) == 21),
            has_right_hand=(right_hand is not None and len(right_hand) == 21),
            in_signing_space=self.preprocessor.in_signing_space,
        )

        # Build live recognition response
        recognition_msg = RecognitionMessage(
            type="recognition",
            state=current_state.value,
            live_gloss=list(self.last_live_gloss),
            raw_tokens=list(self.last_raw_tokens),
            collapsed_tokens=list(self.last_collapsed_tokens),
            final=False,
            landmarks=landmarks_payload,
            metrics=metrics,
        )

        utterance_msg: Optional[UtteranceCompleteMessage] = None

        # 7. Utterance commitment & optional LLM post-processing
        if is_final_commit:
            final_gloss_list = list(self.fsm.last_final_gloss)
            natural_sentence = await self.language_service.to_natural_sentence(final_gloss_list)

            utterance_msg = UtteranceCompleteMessage(
                type="utterance_complete",
                state=FSMState.END_OF_UTTERANCE.value,
                gloss=final_gloss_list,
                sentence=natural_sentence,
                final=True,
                metrics=metrics,
            )
            logger.info(f"Utterance completed: Gloss={final_gloss_list} -> Sentence='{natural_sentence}'")

        return recognition_msg, utterance_msg, annotated_frame

    def close(self):
        self.extractor.close()

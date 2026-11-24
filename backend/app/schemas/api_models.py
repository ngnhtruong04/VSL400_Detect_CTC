"""Pydantic schemas for API requests, responses, and WebSocket messages across Isolated and Continuous modes."""

import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True
    device: str = "cuda"
    labels: int = 400


class ModelInfoResponse(BaseModel):
    model_name: str = "HandsTCNTransformerCTC"
    architecture: Dict[str, Any]
    checkpoint_file: str
    input_contract: str = "(B, T, 294)"
    flattened_dim: int = 294
    num_hand_joints: int = 42
    features_per_joint: int = 7
    gloss_classes: int = 400
    ctc_classes: int = 401
    blank_id: int = 400
    device: str
    trainable_params: int = 5442961


class MetricsData(BaseModel):
    fps: float = 0.0
    mediapipe_ms: float = 0.0
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    motion_energy: float = 0.0
    blank_ratio: float = 0.0
    buffer_frames: int = 0


class NormalizedPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0


class LandmarksPayload(BaseModel):
    left_hand: Optional[List[NormalizedPoint]] = None
    right_hand: Optional[List[NormalizedPoint]] = None
    pose: Optional[List[NormalizedPoint]] = None
    has_left_hand: bool = False
    has_right_hand: bool = False
    in_signing_space: bool = False


# ============================================================
# DIAGNOSTIC AND MODE-SPECIFIC SCHEMAS
# ============================================================

class TopCandidate(BaseModel):
    rank: int
    label_id: int
    gloss: str
    log_score: float
    diagnostic_confidence: float


class IsolatedPrediction(BaseModel):
    gloss: str = ""
    label_id: int = -1
    top5: List[TopCandidate] = Field(default_factory=list)
    greedy_tokens: List[int] = Field(default_factory=list)
    blank_ratio: float = 0.0


class LandmarksDiagnostic(BaseModel):
    left_valid_ratio: float = 0.0
    right_valid_ratio: float = 0.0
    overall_valid_ratio: float = 0.0
    has_left_hand: bool = False
    has_right_hand: bool = False
    in_signing_space: bool = False
    left_hand: Optional[List[NormalizedPoint]] = None
    right_hand: Optional[List[NormalizedPoint]] = None
    pose: Optional[List[NormalizedPoint]] = None


class ModelDiagnostic(BaseModel):
    buffer_frames: int = 0
    input_shape: str = ""
    blank_ratio: float = 0.0
    mediapipe_ms: float = 0.0
    preprocessing_ms: float = 0.0
    forward_ms: float = 0.0
    decode_ms: float = 0.0
    nan_count: int = 0
    inf_count: int = 0


class IsolatedRecognitionMessage(BaseModel):
    type: str = "isolated_recognition"
    mode: str = "isolated"
    timestamp: float = Field(default_factory=time.time)
    session_id: str = ""
    state: str = "IDLE"  # IDLE, RECORDING, INFERRING, RESULT
    fps: float = 0.0
    total_ms: float = 0.0
    motion_energy: float = 0.0
    live_gloss: List[str] = Field(default_factory=list)
    landmarks: LandmarksDiagnostic = Field(default_factory=LandmarksDiagnostic)
    model: ModelDiagnostic = Field(default_factory=ModelDiagnostic)
    prediction: Optional[IsolatedPrediction] = None


class ContinuousStatePayload(BaseModel):
    fsm_state: str = "IDLE"
    live_tokens: List[str] = Field(default_factory=list)
    stable_tokens: List[str] = Field(default_factory=list)
    final_tokens: List[str] = Field(default_factory=list)
    raw_token_ids: List[int] = Field(default_factory=list)
    collapsed_token_ids: List[int] = Field(default_factory=list)
    utterance_frames: int = 0
    recent_frames: int = 0
    endpoint_reason: Optional[str] = None
    final_sentence: str = ""
    translation_status: str = "rule_based"
    translation_error: Optional[str] = None


class ContinuousRecognitionMessage(BaseModel):
    type: str = "continuous_recognition"
    mode: str = "continuous"
    timestamp: float = Field(default_factory=time.time)
    session_id: str = ""
    state: str = "IDLE"
    fps: float = 0.0
    total_ms: float = 0.0
    motion_energy: float = 0.0
    live_gloss: List[str] = Field(default_factory=list)
    landmarks: LandmarksDiagnostic = Field(default_factory=LandmarksDiagnostic)
    model: ModelDiagnostic = Field(default_factory=ModelDiagnostic)
    continuous: ContinuousStatePayload = Field(default_factory=ContinuousStatePayload)


# ============================================================
# LEGACY SCHEMAS (Maintained for Backward Compatibility)
# ============================================================

class RecognitionMessage(BaseModel):
    type: str = "recognition"
    state: str = "SIGNING"
    live_gloss: List[str] = Field(default_factory=list)
    raw_tokens: List[int] = Field(default_factory=list)
    collapsed_tokens: List[int] = Field(default_factory=list)
    final: bool = False
    landmarks: Optional[LandmarksPayload] = None
    metrics: MetricsData = Field(default_factory=MetricsData)


class UtteranceCompleteMessage(BaseModel):
    type: str = "utterance_complete"
    state: str = "END_OF_UTTERANCE"
    gloss: List[str] = Field(default_factory=list)
    sentence: str = ""
    final: bool = True
    metrics: MetricsData = Field(default_factory=MetricsData)


class ErrorMessage(BaseModel):
    type: str = "error"
    message: str
    code: Optional[str] = None

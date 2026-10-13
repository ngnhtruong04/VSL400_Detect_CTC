export type FSMStatus =
  | "IDLE"
  | "STARTING"
  | "SIGNING"
  | "POSSIBLE_END"
  | "END_OF_UTTERANCE"
  | "COOLDOWN";

export type SystemStatus = "loading" | "ready" | "error" | "disconnected";

export type RecognitionMode = "isolated" | "continuous";

export interface NormalizedPoint {
  x: number;
  y: number;
  z?: number;
}

export interface LandmarksData {
  left_hand?: NormalizedPoint[] | null;
  right_hand?: NormalizedPoint[] | null;
  pose?: NormalizedPoint[] | null;
  has_left_hand: boolean;
  has_right_hand: boolean;
  in_signing_space: boolean;
  left_valid_ratio?: number;
  right_valid_ratio?: number;
  overall_valid_ratio?: number;
}

export interface MetricsData {
  fps: number;
  mediapipe_ms: number;
  preprocessing_ms: number;
  inference_ms: number;
  decode_ms: number;
  total_ms: number;
  motion_energy: number;
  blank_ratio: number;
  buffer_frames: number;
}

export interface TopCandidate {
  rank: number;
  label_id: number;
  gloss: string;
  log_score: number;
  diagnostic_confidence: number;
}

export interface IsolatedPrediction {
  gloss: string;
  label_id: number;
  top5: TopCandidate[];
  greedy_tokens: number[];
  blank_ratio: number;
}

export interface LandmarksDiagnostic {
  left_valid_ratio: number;
  right_valid_ratio: number;
  overall_valid_ratio: number;
  has_left_hand: boolean;
  has_right_hand: boolean;
  in_signing_space: boolean;
  left_hand?: NormalizedPoint[] | null;
  right_hand?: NormalizedPoint[] | null;
  pose?: NormalizedPoint[] | null;
}

export interface ModelDiagnostic {
  buffer_frames: number;
  input_shape: string;
  blank_ratio: number;
  forward_ms: number;
  decode_ms: number;
  nan_count: number;
  inf_count: number;
}

export interface ContinuousStatePayload {
  fsm_state: string;
  live_tokens: string[];
  stable_tokens: string[];
  final_tokens: string[];
  raw_token_ids: number[];
  collapsed_token_ids: number[];
  utterance_frames: number;
  recent_frames: number;
  endpoint_reason?: string | null;
  final_sentence: string;
}

export interface IsolatedRecognitionMessage {
  type: "isolated_recognition";
  mode: "isolated";
  timestamp: number;
  session_id: string;
  state: "IDLE" | "RECORDING" | "INFERRING" | "RESULT";
  fps: number;
  total_ms: number;
  motion_energy: number;
  live_gloss: string[];
  landmarks: LandmarksDiagnostic;
  model: ModelDiagnostic;
  prediction?: IsolatedPrediction | null;
}

export interface ContinuousRecognitionMessage {
  type: "continuous_recognition";
  mode: "continuous";
  timestamp: number;
  session_id: string;
  state: FSMStatus;
  fps: number;
  total_ms: number;
  motion_energy: number;
  live_gloss: string[];
  landmarks: LandmarksDiagnostic;
  model: ModelDiagnostic;
  continuous: ContinuousStatePayload;
}

// Backward compatibility messages
export interface RecognitionMessage {
  type: "recognition";
  state: FSMStatus;
  live_gloss: string[];
  raw_tokens?: number[];
  collapsed_tokens?: number[];
  final: boolean;
  landmarks?: LandmarksData | null;
  metrics: MetricsData;
}

export interface UtteranceCompleteMessage {
  type: "utterance_complete";
  state: FSMStatus;
  gloss: string[];
  sentence: string;
  final: boolean;
  metrics: MetricsData;
}

export interface AppSettings {
  mirrorPreview: boolean;
  showLandmarks: boolean;
  drawHands: boolean;
  drawPose: boolean;
  showFps: boolean;
  sensitivity: "low" | "medium" | "high";
  llmEnabled: boolean;
  llmProvider: "none" | "gemini" | "ollama";
}

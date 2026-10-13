"""Centralized settings and path management for VSL Realtime Recognition System."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


class Settings:
    def __init__(self):
        # Base paths
        # backend/app/config/settings.py -> backend/app/config -> backend/app -> backend -> VSL_System
        self.PROJECT_ROOT = Path(os.getenv("VSL_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent))).resolve()
        self.ASSET_ROOT = Path(os.getenv("VSL_ASSET_ROOT", str(self.PROJECT_ROOT / "VSL400_System_Assets"))).resolve()

        # Load runtime.yaml
        config_path = self.PROJECT_ROOT / "config" / "runtime.yaml"
        self.raw_config: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.raw_config = yaml.safe_load(f) or {}

        # Resolve asset file paths
        paths_cfg = self.raw_config.get("paths", {})
        self.CHECKPOINT_PATH = (self.PROJECT_ROOT / paths_cfg.get(
            "checkpoint_path",
            "VSL400_System_Assets/VSL400_11_MULTISEED_TCN_TRANSFORMER/checkpoints/best_seed_20260827.pt"
        )).resolve()

        self.LABEL_MAP_PATH = (self.PROJECT_ROOT / paths_cfg.get(
            "label_map_path",
            "VSL400_System_Assets/VSL400_COMBINE_5GROUPS_Shared_Keypoints/label_map.json"
        )).resolve()

        self.EXTRACTOR_CONFIG_PATH = (self.PROJECT_ROOT / paths_cfg.get(
            "extractor_config_path",
            "VSL400_System_Assets/VSL400_COMBINE_5GROUPS_Shared_Keypoints/extractor_config.json"
        )).resolve()

        # Model configuration
        model_cfg = self.raw_config.get("model", {})
        self.DEVICE_SETTING = os.getenv("VSL_DEVICE", model_cfg.get("device", "auto"))
        self.INPUT_DIM = int(model_cfg.get("input_dim", 294))
        self.MODEL_DIM = int(model_cfg.get("model_dim", 256))
        self.TCN_LAYERS = int(model_cfg.get("tcn_layers", 4))
        self.TCN_KERNEL_SIZE = int(model_cfg.get("tcn_kernel_size", 5))
        self.NHEAD = int(model_cfg.get("nhead", 8))
        self.TRANSFORMER_LAYERS = int(model_cfg.get("transformer_layers", 4))
        self.FF_DIM = int(model_cfg.get("ff_dim", 768))
        self.NUM_CLASSES = int(model_cfg.get("num_classes", 401))
        self.BLANK_ID = int(model_cfg.get("blank_id", 400))
        self.BLANK_BIAS_INIT = float(model_cfg.get("blank_bias_init", -1.5))

        # Preprocessing configuration
        prep_cfg = self.raw_config.get("preprocessing", {})
        self.POSE_VISIBILITY_THRESHOLD = float(prep_cfg.get("pose_visibility_threshold", 0.4))
        self.POSE_PRESENCE_THRESHOLD = float(prep_cfg.get("pose_presence_threshold", 0.2))
        self.MIN_SCALE = float(prep_cfg.get("min_scale", 0.0001))
        self.SHOULDER_SCALE_MULTIPLIER = float(prep_cfg.get("shoulder_scale_multiplier", 1.0))
        self.CLIP_VALUE = float(prep_cfg.get("clip_value", 6.0))
        self.N_JOINTS = int(prep_cfg.get("n_joints", 42))
        self.FEATURES_PER_JOINT = int(prep_cfg.get("features_per_joint", 7))

        # Sequence Buffer
        buf_cfg = self.raw_config.get("buffer", {})
        self.BUFFER_MAX_FRAMES = int(buf_cfg.get("max_frames", 90))
        self.BUFFER_MIN_FRAMES = int(buf_cfg.get("min_frames", 15))
        self.INFERENCE_STRIDE = int(buf_cfg.get("inference_stride", 2))

        # Endpoint FSM
        fsm_cfg = self.raw_config.get("endpoint", {})
        self.ENDPOINT_ENABLED = bool(fsm_cfg.get("enabled", True))
        self.MOTION_THRESHOLD = float(fsm_cfg.get("motion_threshold", 0.015))
        self.MIN_SIGN_DURATION_MS = int(fsm_cfg.get("min_sign_duration_ms", 400))
        self.POSSIBLE_END_DELAY_MS = int(fsm_cfg.get("possible_end_delay_ms", 600))
        self.SILENCE_MS = int(fsm_cfg.get("silence_ms", 1200))
        self.COOLDOWN_MS = int(fsm_cfg.get("cooldown_ms", 500))

        # Camera & Display
        cam_cfg = self.raw_config.get("camera", {})
        self.CAMERA_WIDTH = int(cam_cfg.get("width", 1280))
        self.CAMERA_HEIGHT = int(cam_cfg.get("height", 720))
        self.TARGET_FPS = int(cam_cfg.get("target_fps", 30))
        self.MIRROR_PREVIEW = bool(cam_cfg.get("mirror_preview", True))

        disp_cfg = self.raw_config.get("display", {})
        self.DRAW_HANDS = bool(disp_cfg.get("draw_hands", True))
        self.DRAW_POSE = bool(disp_cfg.get("draw_pose", False))
        self.SHOW_FPS = bool(disp_cfg.get("show_fps", True))

        # LLM
        llm_cfg = self.raw_config.get("llm", {})
        self.LLM_ENABLED = bool(llm_cfg.get("enabled", False))
        self.LLM_PROVIDER = os.getenv("VSL_LLM_PROVIDER", llm_cfg.get("provider", "none"))
        self.LLM_API_KEY = os.getenv("GEMINI_API_KEY", llm_cfg.get("api_key", ""))
        self.LLM_MODEL = os.getenv("VSL_LLM_MODEL", llm_cfg.get("model_name", "gemini-2.5-flash"))
        self.LLM_ENDPOINT = os.getenv("VSL_LLM_ENDPOINT", llm_cfg.get("endpoint_url", ""))

        # Server
        self.HOST = os.getenv("VSL_HOST", "0.0.0.0")
        self.PORT = int(os.getenv("VSL_PORT", 8000))

        # Diagnostic logging
        self.DEBUG_CONTINUOUS = os.getenv("DEBUG_CONTINUOUS", "false").lower() in ("true", "1", "yes")


settings = Settings()

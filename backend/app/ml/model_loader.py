"""Model loader with strict checkpoint verification and hardware device detection."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from backend.app.config.settings import settings
from backend.app.ml.model import HandsTCNTransformerCTC

logger = logging.getLogger("vsl.model_loader")


class ModelLoader:
    _instance: Optional["ModelLoader"] = None
    _model: Optional[HandsTCNTransformerCTC] = None
    _device: Optional[torch.device] = None

    def __init__(self):
        self.device = self._resolve_device()
        self.model: Optional[HandsTCNTransformerCTC] = None

    def _resolve_device(self) -> torch.device:
        pref = settings.DEVICE_SETTING.lower()
        if pref == "cuda" and torch.cuda.is_available():
            dev = torch.device("cuda")
        elif pref == "cpu":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Resolved inference device: {dev}")
        return dev

    def load_model(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = True,
    ) -> HandsTCNTransformerCTC:
        """Loads and strictly validates the HandsTCNTransformerCTC model."""
        if self.model is not None:
            return self.model

        ckpt_path = Path(checkpoint_path or settings.CHECKPOINT_PATH)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        logger.info(f"Instantiating HandsTCNTransformerCTC with input_dim={settings.INPUT_DIM}, classes={settings.NUM_CLASSES}")
        model = HandsTCNTransformerCTC(
            input_dim=settings.INPUT_DIM,
            model_dim=settings.MODEL_DIM,
            tcn_layers=settings.TCN_LAYERS,
            tcn_kernel_size=settings.TCN_KERNEL_SIZE,
            nhead=settings.NHEAD,
            transformer_layers=settings.TRANSFORMER_LAYERS,
            ff_dim=settings.FF_DIM,
            num_classes=settings.NUM_CLASSES,
            blank_bias_init=settings.BLANK_BIAS_INIT,
        )

        logger.info(f"Loading checkpoint weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(f"Unexpected checkpoint structure type: {type(checkpoint)}")

        # Strict state_dict verification
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded successfully. Keys verification: {incompatible_keys}")

        model.to(self.device)
        model.eval()

        # Startup dummy inference verification
        self._verify_dummy_forward(model)

        self.model = model
        ModelLoader._model = model
        ModelLoader._device = self.device
        return model

    def _verify_dummy_forward(self, model: nn.Module):
        """Runs dummy tensor forward pass to ensure pipeline stability."""
        try:
            with torch.inference_mode():
                dummy_input = torch.zeros((1, 30, settings.INPUT_DIM), device=self.device)
                logits = model(dummy_input)
                assert logits.shape == (1, 30, settings.NUM_CLASSES), (
                    f"Unexpected output shape: {logits.shape}, expected (1, 30, {settings.NUM_CLASSES})"
                )
            logger.info(f"Dummy forward validation PASSED! Output shape: {logits.shape}")
        except Exception as e:
            logger.error(f"Dummy forward validation FAILED: {e}")
            raise e

    @classmethod
    def get_loaded_model(cls) -> Tuple[HandsTCNTransformerCTC, torch.device]:
        if cls._model is None or cls._device is None:
            loader = cls()
            loader.load_model()
        return cls._model, cls._device


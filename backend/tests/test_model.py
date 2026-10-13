"""Unit tests for HandsTCNTransformerCTC model architecture and checkpoint loading."""

import pytest
import torch

from backend.app.config.settings import settings
from backend.app.ml.model import HandsTCNTransformerCTC
from backend.app.ml.model_loader import ModelLoader


def test_model_architecture_parameter_count():
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 5442961, f"Expected 5,442,961 params, got {trainable_params}"


def test_checkpoint_strict_load():
    loader = ModelLoader()
    model = loader.load_model(strict=True)
    assert model is not None
    assert model.classifier.out_features == 401
    assert model.classifier.bias[-1].item() < 0.0  # Blank bias was initialized negative


def test_model_dummy_forward_contract():
    model, device = ModelLoader.get_loaded_model()
    batch_size = 2
    time_steps = 45
    dummy_input = torch.randn(batch_size, time_steps, settings.INPUT_DIM, device=device)

    with torch.inference_mode():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, time_steps, settings.NUM_CLASSES)
    assert not torch.isnan(logits).any()


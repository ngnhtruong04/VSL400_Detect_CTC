"""Model information endpoint."""

from fastapi import APIRouter
from backend.app.config.settings import settings
from backend.app.ml.model_loader import ModelLoader
from backend.app.schemas.api_models import ModelInfoResponse

router = APIRouter(tags=["Model Info"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    model, device = ModelLoader.get_loaded_model()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return ModelInfoResponse(
        model_name="HandsTCNTransformerCTC",
        architecture={
            "model_dim": settings.MODEL_DIM,
            "tcn_layers": settings.TCN_LAYERS,
            "tcn_kernel_size": settings.TCN_KERNEL_SIZE,
            "nhead": settings.NHEAD,
            "transformer_layers": settings.TRANSFORMER_LAYERS,
            "ff_dim": settings.FF_DIM,
            "blank_bias_init": settings.BLANK_BIAS_INIT,
        },
        checkpoint_file=settings.CHECKPOINT_PATH.name,
        input_contract=f"(B, T, {settings.INPUT_DIM})",
        flattened_dim=settings.INPUT_DIM,
        num_hand_joints=settings.N_JOINTS,
        features_per_joint=settings.FEATURES_PER_JOINT,
        gloss_classes=settings.NUM_CLASSES - 1,
        ctc_classes=settings.NUM_CLASSES,
        blank_id=settings.BLANK_ID,
        device=str(device),
        trainable_params=trainable_params,
    )


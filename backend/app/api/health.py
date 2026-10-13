"""Health check endpoint."""

from fastapi import APIRouter
from backend.app.ml.model_loader import ModelLoader
from backend.app.schemas.api_models import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    model, device = ModelLoader.get_loaded_model()
    return HealthResponse(
        status="ok",
        model_loaded=(model is not None),
        device=str(device),
        labels=400,
    )


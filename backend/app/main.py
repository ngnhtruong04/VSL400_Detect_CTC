"""Main FastAPI Application entry point for VSL Realtime Recognition System."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config.settings import settings
from backend.app.ml.model_loader import ModelLoader
from backend.app.api import health, model_info, realtime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("vsl.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Pre-loads the locked model and validates checkpoints at startup."""
    logger.info("=" * 60)
    logger.info("STARTING VSL REALTIME RECOGNITION SYSTEM BACKEND")
    logger.info(f"Project Root: {settings.PROJECT_ROOT}")
    logger.info(f"Asset Root:   {settings.ASSET_ROOT}")
    logger.info(f"Checkpoint:   {settings.CHECKPOINT_PATH.name}")
    logger.info("=" * 60)

    try:
        import torch
        if settings.CPU_NUM_THREADS > 0:
            torch.set_num_threads(settings.CPU_NUM_THREADS)
        model, device = ModelLoader.get_loaded_model()
        logger.info(f"Model successfully loaded into memory on device: {device}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Verified trainable parameters: {trainable_params:,} (Target: 5,442,961)")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize model at startup: {e}", exc_info=True)
        raise e

    yield

    logger.info("SHUTTING DOWN VSL REALTIME RECOGNITION SYSTEM BACKEND")


app = FastAPI(
    title="VSL Realtime Recognition System",
    description="Realtime Vietnamese Sign Language Recognition integrating locked TCN-Transformer-CTC (VSL400).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware to support frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(realtime.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )


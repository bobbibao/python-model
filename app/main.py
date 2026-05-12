"""
FastAPI Application for SDXL Image Generation with ControlNet and LoRA

Production-ready backend with:
- SDXL base model
- ControlNet support (canny, lineart)
- LoRA fine-tuning (dynamic loading)
- Unified v2 API (/api/v2/generate)
- Memory optimizations for Colab Pro
- CUDA memory management and defragmentation
"""

import logging
import gc
import torch
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .api.v2 import router as v2_router
from .pipelines.pipeline_registry import get_registry
from .utils.memory_utils import get_gpu_memory_info, log_memory_stats

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Ensure output directory exists
output_dir = Path(settings.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown events.
    
    Startup:
    - Initialize pipeline registry
    - Log device and configuration
    - Preload pipelines (optional)
    
    Shutdown:
    - Clean up all pipelines
    - Clear CUDA cache
    - Log shutdown message
    """
    # ============================================
    # STARTUP
    # ============================================
    logger.info("=" * 70)
    logger.info("🚀 Starting Vizera Python Model Service (SDXL + ControlNet + LoRA)")
    logger.info("=" * 70)
    
    try:
        # Initialize pipeline registry (singleton)
        registry = get_registry()
        logger.info("✓ Pipeline registry initialized")
        
        # Log device and memory info
        log_memory_stats("startup")
        
        # Get registry status
        status = registry.get_status()
        logger.info(f"Device: {status['device']} | Dtype: {status['dtype']}")
        
        # Log configuration
        logger.info(f"SDXL Model: {settings.sdxl_base_model_id}")
        logger.info(f"ControlNet Canny: {settings.controlnet_canny_model_id}")
        logger.info(f"ControlNet Lineart: {settings.controlnet_lineart_model_id}")
        logger.info(f"Default LoRA: {settings.default_lora_path}")
        logger.info(f"Cache Dir: {settings.cache_dir}")
        logger.info(f"Output Dir: {settings.output_dir}")
        
        logger.info("=" * 70)
        logger.info("✓ Service started successfully!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # ============================================
    # SHUTDOWN
    # ============================================
    logger.info("=" * 70)
    logger.info("🛑 Shutting down Vizera Python Model Service...")
    logger.info("=" * 70)
    
    try:
        # Clean up all pipelines
        registry = get_registry()
        registry.cleanup_all()
        logger.info("✓ Pipelines cleaned up")
        
        # Force garbage collection
        gc.collect()
        logger.info("✓ Python garbage collected")
        
        # Clear CUDA memory and caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("✓ CUDA cache cleared and IPC collected")
        
        # Log final memory state
        log_memory_stats("shutdown")
        
        logger.info("=" * 70)
        logger.info("✓ Shutdown completed successfully")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"⚠ Error during shutdown: {e}", exc_info=True)

app = FastAPI(
    title=settings.service_name,
    version="2.0.0",
    description="Production-grade SDXL image generation with ControlNet and LoRA support",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for image serving
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")

# Include v2 routes
app.include_router(v2_router)


# ============================================
# HEALTH CHECK ENDPOINT
# ============================================


@app.get("/health", tags=["health"])
def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns basic service status.
    """
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": "2.0.0",
    }


@app.get("/debug/memory", tags=["debug"])
def debug_memory():
    """
    Debug endpoint to check GPU memory usage.
    
    ⚠️ Warning: For debugging only, not for production use.
    """
    return get_gpu_memory_info()


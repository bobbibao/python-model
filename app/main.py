"""
FastAPI Application for Stable Diffusion Image Generation and Editing
Production-ready with optimized model loading and resource management
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .generation import build_output_url, ensure_output_dir, process_edit, process_generate
from .models import EditRequest, GenerateRequest, GenerateResponse
from .pipeline import get_pipeline_status

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

ensure_output_dir()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown events
    - Preload models on startup if configured
    - Graceful shutdown
    """
    # Startup
    logger.info("=" * 70)
    logger.info("🚀 Starting Vizera Python Model Service...")
    logger.info("=" * 70)
    
    
    logger.info("=" * 70)
    logger.info("✓ Service started successfully!")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("=" * 70)
    logger.info("🛑 Shutting down Vizera Python Model Service...")
    logger.info("=" * 70)


app = FastAPI(
    title=settings.service_name,
    version="0.3.0",
    description="Production-grade Stable Diffusion API with optimized model loading",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": "0.3.0",
    }


@app.get("/debug/pipeline-status")
def pipeline_status():
    """Debug endpoint to check pipeline status and configuration"""
    return get_pipeline_status()


@app.post("/api/v1/generate", response_model=GenerateResponse)
def generate_image(payload: GenerateRequest):
    """
    Generate images using Stable Diffusion
    
    Supports multiple input types:
    - text-to-image: Generate from prompt
    - image-to-image: Modify existing image
    - image-upscaling: Upscale image resolution
    - line-drawing-to-image: Convert line drawing to detailed image
    """
    try:
        job_id, out_width, out_height = process_generate(
            prompt=payload.prompt,
            width=payload.width,
            height=payload.height,
            seed=payload.seed,
            input_type=payload.input_type,
            image=payload.image,
            strength=payload.strength,
            upscale_factor=payload.upscale_factor,
        )
        return GenerateResponse(
            job_id=job_id,
            status="completed",
            image_url=build_output_url(job_id),
            width=out_width,
            height=out_height,
        )
    except Exception as e:
        logger.error(f"Generation request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )


@app.post("/api/v1/edit", response_model=GenerateResponse)
def edit_image(payload: EditRequest):
    """
    Edit images using various methods:
    - EDIT_OBJECT_REMOVAL: Remove objects from images
    - EDIT_UPSCALING: Upscale image resolution
    - EDIT_FLUX_FILL_INPAINT: Fill masked areas with AI
    - EDIT_FLUX_FILL_EXTEND: Extend canvas with AI generation
    """
    try:
        job_id, out_width, out_height = process_edit(
            method=payload.method,
            image=payload.image,
            prompt=payload.prompt,
            style_prompt=payload.style_prompt,
            mask=payload.mask,
            crop=payload.crop,
            direction=payload.direction,
            pixels=payload.pixels,
            upscale_factor=payload.upscale_factor,
            seed=payload.seed,
        )
        return GenerateResponse(
            job_id=job_id,
            status="completed",
            image_url=build_output_url(job_id),
            width=out_width,
            height=out_height,
        )
    except Exception as e:
        logger.error(f"Edit request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image editing failed: {str(e)}"
        )

"""
Production-grade Stable Diffusion Pipeline Manager
- Singleton pattern with thread-safe lazy loading
- Local caching using HuggingFace cache directories
- Memory optimization (float16, attention slicing, VAE slicing)
- Automatic GPU/CPU detection and fallback
- Comprehensive logging and error handling
"""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Global pipeline instances
_pipelines = {
    "text_to_image": None,
    "image_to_image": None,
    "inpaint": None,
}
_pipeline_lock = Lock()
_initialized = {key: False for key in _pipelines.keys()}


def _get_device() -> str:
    """
    Determine device: CUDA (GPU) or CPU
    Returns: "cuda" if CUDA is available and working, else "cpu"
    """
    if torch.cuda.is_available():
        try:
            # Test CUDA availability
            _ = torch.tensor([1.0]).to("cuda")
            logger.info(f"✓ CUDA available. GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except RuntimeError as e:
            logger.warning(f"⚠ CUDA available but not working: {e}. Falling back to CPU.")
            return "cpu"
    else:
        logger.info("ℹ CUDA not available. Using CPU (inference will be slower)")
        return "cpu"


def _get_torch_dtype(device: str) -> torch.dtype:
    """
    Select appropriate dtype based on device
    - GPU (CUDA): float16 for memory efficiency
    - CPU: float32 (float16 can be unstable on CPU)
    """
    return torch.float16 if device == "cuda" else torch.float32


def _check_cache_exists(model_id: str) -> bool:
    """
    Check if model is already cached locally
    Returns: True if model exists in cache, False otherwise
    """
    cache_dir = Path(settings.cache_dir) / f"models--{model_id.replace('/', '--')}"
    exists = cache_dir.exists()
    if exists:
        logger.info(f"✓ Using cached model: {model_id}")
    else:
        logger.info(f"ℹ Model not cached, will download: {model_id}")
    return exists


def get_text_to_image_pipeline():
    """
    Get or initialize text-to-image pipeline (singleton pattern)
    
    Features:
    - Loads only once at first access
    - Thread-safe using locks
    - Respects HF_HOME and TRANSFORMERS_CACHE environment variables
    - Memory optimizations enabled
    - Automatic GPU/CPU detection
    """
    global _pipelines, _initialized
    
    key = "text_to_image"
    
    if _pipelines[key] is not None:
        return _pipelines[key]
    
    with _pipeline_lock:
        # Double-check after acquiring lock
        if _pipelines[key] is not None:
            return _pipelines[key]
        
        try:
            device = _get_device()
            dtype = _get_torch_dtype(device)
            model_id = settings.model_id
            
            logger.info(f"🔄 Loading text-to-image model (لأول مرة)...")
            logger.info(f"   Model: {model_id}")
            logger.info(f"   Device: {device}")
            logger.info(f"   Dtype: {dtype}")
            logger.info(f"   Cache: {settings.cache_dir}")
            
            _check_cache_exists(model_id)
            
            # Load pipeline with memory optimizations
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,  # Disable safety checker for faster inference
                low_cpu_mem_usage=settings.low_cpu_mem_usage,
                cache_dir=settings.cache_dir,
            ).to(device)
            
            # Enable memory optimizations
            if settings.enable_attention_slicing:
                pipeline.enable_attention_slicing()
                logger.debug("✓ Attention slicing enabled")
            
            if settings.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                logger.debug("✓ VAE slicing enabled")
            
            _pipelines[key] = pipeline
            _initialized[key] = True
            
            logger.info("✓ Text-to-image pipeline loaded successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"✗ Failed to load text-to-image pipeline: {e}", exc_info=True)
            raise


def get_image_to_image_pipeline():
    """
    Get or initialize image-to-image pipeline (singleton pattern)
    """
    global _pipelines, _initialized
    
    key = "image_to_image"
    
    if _pipelines[key] is not None:
        return _pipelines[key]
    
    with _pipeline_lock:
        if _pipelines[key] is not None:
            return _pipelines[key]
        
        try:
            device = _get_device()
            dtype = _get_torch_dtype(device)
            model_id = settings.model_id
            
            logger.info(f"🔄 Loading image-to-image pipeline...")
            logger.info(f"   Model: {model_id}")
            
            _check_cache_exists(model_id)
            
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                low_cpu_mem_usage=settings.low_cpu_mem_usage,
                cache_dir=settings.cache_dir,
            ).to(device)
            
            if settings.enable_attention_slicing:
                pipeline.enable_attention_slicing()
            
            if settings.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            
            _pipelines[key] = pipeline
            _initialized[key] = True
            
            logger.info("✓ Image-to-image pipeline loaded successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"✗ Failed to load image-to-image pipeline: {e}", exc_info=True)
            raise


def get_inpaint_pipeline():
    """
    Get or initialize inpainting pipeline (singleton pattern)
    """
    global _pipelines, _initialized
    
    key = "inpaint"
    
    if _pipelines[key] is not None:
        return _pipelines[key]
    
    with _pipeline_lock:
        if _pipelines[key] is not None:
            return _pipelines[key]
        
        try:
            device = _get_device()
            dtype = _get_torch_dtype(device)
            model_id = settings.inpaint_model_id
            
            logger.info(f"🔄 Loading inpaint pipeline...")
            logger.info(f"   Model: {model_id}")
            
            _check_cache_exists(model_id)
            
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                low_cpu_mem_usage=settings.low_cpu_mem_usage,
                cache_dir=settings.cache_dir,
            ).to(device)
            
            if settings.enable_attention_slicing:
                pipeline.enable_attention_slicing()
            
            if settings.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            
            _pipelines[key] = pipeline
            _initialized[key] = True
            
            logger.info("✓ Inpaint pipeline loaded successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"✗ Failed to load inpaint pipeline: {e}", exc_info=True)
            raise


def preload_all_pipelines():
    """
    Preload all pipelines at application startup (optional)
    Useful for reducing first-request latency
    """
    logger.info("=" * 60)
    logger.info("🚀 PRELOADING ALL PIPELINES (model initialization)")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading text-to-image pipeline...")
        get_text_to_image_pipeline()
        
        logger.info("Loading image-to-image pipeline...")
        get_image_to_image_pipeline()
        
        logger.info("Loading inpaint pipeline...")
        get_inpaint_pipeline()
        
        logger.info("=" * 60)
        logger.info("✓ All pipelines preloaded successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Failed to preload pipelines: {e}", exc_info=True)
        raise


def get_pipeline_status():
    """
    Get status of all pipelines for debugging
    """
    return {
        "text_to_image": {
            "loaded": _pipelines["text_to_image"] is not None,
            "initialized": _initialized["text_to_image"],
        },
        "image_to_image": {
            "loaded": _pipelines["image_to_image"] is not None,
            "initialized": _initialized["image_to_image"],
        },
        "inpaint": {
            "loaded": _pipelines["inpaint"] is not None,
            "initialized": _initialized["inpaint"],
        },
        "device": _get_device(),
        "model_id": settings.model_id,
        "inpaint_model_id": settings.inpaint_model_id,
        "cache_dir": settings.cache_dir,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

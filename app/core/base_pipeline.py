"""
Base pipeline class: shared state and initialization logic.

Provides:
- Device and dtype management
- Common initialization/cleanup patterns
- Generator management for reproducibility
- VRAM profiling hooks
"""

import logging
import torch
from pathlib import Path
from typing import Optional

from ..config import settings
from .device_utils import get_device, get_torch_dtype, log_device_info

logger = logging.getLogger(__name__)


class BasePipeline:
    """
    Base class for all pipeline implementations.
    
    Manages:
    - Device selection and dtype
    - Cache directory handling
    - Generator creation
    - Memory optimization settings
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize base pipeline state.
        
        Args:
            model_name: Descriptive name for logging (e.g., "sdxl_base", "controlnet")
        """
        self.model_name = model_name
        self.device = get_device()
        self.dtype = get_torch_dtype(self.device)
        self.cache_dir = settings.cache_dir
        
        logger.info(f"[{self.model_name}] Initializing on {self.device} with dtype {self.dtype}")
        log_device_info()
    
    def get_generator(self, seed: Optional[int] = None) -> torch.Generator:
        """
        Create a torch Generator for reproducible inference.
        
        Args:
            seed: Optional seed for reproducibility. If None, uses random seed.
        
        Returns:
            torch.Generator configured for the current device.
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
            logger.debug(f"[{self.model_name}] Using seed: {seed}")
        return generator
    
    def apply_memory_optimizations(self, pipeline):
        """
        Apply memory optimization techniques to a pipeline.
        
        Optimizations:
        - Attention slicing (reduces peak memory)
        - VAE slicing (reduces peak memory during VAE decode)
        - Enable xformers if available
        - Low CPU memory usage flag
        
        Args:
            pipeline: The diffusers pipeline to optimize.
        """
        try:
            if settings.enable_attention_slicing:
                pipeline.enable_attention_slicing()
                logger.debug(f"[{self.model_name}] Enabled attention slicing")
        except Exception as e:
            logger.warning(f"[{self.model_name}] Could not enable attention slicing: {e}")
        
        try:
            if settings.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                logger.debug(f"[{self.model_name}] Enabled VAE slicing")
        except Exception as e:
            logger.warning(f"[{self.model_name}] Could not enable VAE slicing: {e}")
        
        try:
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                pipeline.enable_xformers_memory_efficient_attention()
                logger.debug(f"[{self.model_name}] Enabled xformers optimization")
        except Exception as e:
            logger.debug(f"[{self.model_name}] xformers not available: {e}")
    
    def apply_cpu_offload(self, pipeline):
        """
        Enable CPU offloading for better VRAM utilization.
        
        Trades off latency (model moving CPU ↔ GPU) for reduced peak VRAM.
        
        Args:
            pipeline: The diffusers pipeline to optimize.
        """
        try:
            if settings.enable_model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload(device_id=0)
                logger.debug(f"[{self.model_name}] Enabled CPU offload")
        except Exception as e:
            logger.warning(f"[{self.model_name}] Could not enable CPU offload: {e}")
    
    def to_device(self, obj):
        """
        Move object to the configured device.
        
        Args:
            obj: Object with .to() method
        
        Returns:
            Object on the configured device
        """
        return obj.to(self.device)
    
    def check_model_cached(self, model_id: str) -> bool:
        """
        Check if a model is already cached locally.
        
        Args:
            model_id: HuggingFace model ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
        
        Returns:
            True if model exists in cache, False otherwise
        """
        cache_path = Path(self.cache_dir) / f"models--{model_id.replace('/', '--')}"
        exists = cache_path.exists()
        
        if exists:
            logger.debug(f"[{self.model_name}] Model cached: {model_id}")
        else:
            logger.info(f"[{self.model_name}] Model not cached, will download: {model_id}")
        
        return exists
    
    @staticmethod
    def inference_mode():
        """
        Context manager for inference mode (disables gradients).
        
        Usage:
            with pipeline.inference_mode():
                output = pipeline(...)
        """
        return torch.inference_mode()

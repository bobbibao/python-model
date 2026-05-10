"""
SDXL Base Pipeline: Text-to-image generation using SDXL.

Uses StableDiffusionXLPipeline for core functionality with memory optimizations.
"""

import logging
import torch
from typing import Optional
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
)
from PIL import Image

from ..core import BasePipeline
from ..config import settings

logger = logging.getLogger(__name__)


class SDXLBasePipeline(BasePipeline):
    """
    SDXL base pipeline for text-to-image generation.
    
    Manages:
    - Model initialization and caching
    - Memory optimizations
    - Inference execution with proper cleanup
    """
    
    def __init__(self):
        """Initialize SDXL base pipeline."""
        super().__init__(model_name="sdxl_base")
        self.pipeline = None
        self._initialized = False
    
    def setup(self):
        """
        Initialize the SDXL pipeline.
        
        Loads the base model, applies memory optimizations, and prepares for inference.
        Thread-safe: should only be called once, protected by registry lock.
        """
        if self._initialized and self.pipeline is not None:
            logger.debug("[sdxl_base] Pipeline already initialized")
            return
        
        try:
            model_id = settings.sdxl_base_model_id
            logger.info(f"[sdxl_base] Loading SDXL base model: {model_id}")
            self.check_model_cached(model_id)
            
            # Load pipeline with memory optimizations
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if settings.use_fp16 else None,
                cache_dir=settings.cache_dir,
            )
            
            # Move to device
            self.pipeline = self.to_device(self.pipeline)
            
            # Apply memory optimizations
            self.apply_memory_optimizations(self.pipeline)
            self.apply_cpu_offload(self.pipeline)
            
            # Disable safety checker for faster inference
            self.pipeline.safety_checker = None
            
            self._initialized = True
            logger.info("[sdxl_base] ✓ SDXL base pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"[sdxl_base] ✗ Failed to initialize: {e}", exc_info=True)
            raise
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate image from text prompt using SDXL.
        
        Args:
            prompt: The text prompt to generate from
            negative_prompt: Negative prompt to avoid
            height: Output image height (multiple of 8)
            width: Output image width (multiple of 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
        
        Returns:
            PIL Image generated from the prompt
        
        Raises:
            RuntimeError: If pipeline is not initialized
        """
        if not self._initialized or self.pipeline is None:
            raise RuntimeError("[sdxl_base] Pipeline not initialized. Call setup() first.")
        
        # Normalize dimensions to multiples of 8
        height = (height // 8) * 8
        width = (width // 8) * 8
        
        logger.debug(
            f"[sdxl_base] Generating: prompt='{prompt[:60]}...', "
            f"size={width}x{height}, steps={num_inference_steps}, scale={guidance_scale}"
        )
        
        generator = self.get_generator(seed)
        
        try:
            with self.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            
            logger.debug("[sdxl_base] ✓ Generation completed")
            return output.images[0]
            
        except Exception as e:
            logger.error(f"[sdxl_base] Generation failed: {e}", exc_info=True)
            raise
    
    def cleanup(self):
        """
        Clean up pipeline resources.
        
        Moves pipeline to CPU and clears CUDA cache.
        """
        if self.pipeline is not None:
            try:
                self.pipeline = self.pipeline.to("cpu")
                logger.debug("[sdxl_base] Pipeline moved to CPU")
            except Exception as e:
                logger.warning(f"[sdxl_base] Error during cleanup: {e}")
        
        # Clear CUDA cache if available
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("[sdxl_base] CUDA cache cleared")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"[sdxl_base] Error during __del__: {e}")

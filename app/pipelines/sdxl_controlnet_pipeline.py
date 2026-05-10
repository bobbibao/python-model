"""
SDXL + ControlNet Pipeline: Sketch-to-image and conditional generation.

Uses StableDiffusionXLControlNetPipeline for ControlNet integration.
Supports canny edge detection and line art / sketch inputs.
"""

import logging
import torch
from typing import Optional
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
from PIL import Image

from ..core import BasePipeline
from ..models.enums import ControlNetType
from ..config import settings

logger = logging.getLogger(__name__)


class SDXLControlNetPipeline(BasePipeline):
    """
    SDXL pipeline with ControlNet support for conditional generation.
    
    Manages:
    - Base model initialization
    - ControlNet model loading and switching
    - Sketch-to-image inference
    - Image conditioning with canny or lineart guides
    """
    
    # Model ID mappings for ControlNet types
    CONTROLNET_MODELS = {
        ControlNetType.CANNY: settings.controlnet_canny_model_id,
        ControlNetType.LINEART: settings.controlnet_lineart_model_id,
    }
    
    def __init__(self):
        """Initialize SDXL ControlNet pipeline."""
        super().__init__(model_name="sdxl_controlnet")
        self.pipeline = None
        self.controlnet_models = {}  # Cache of loaded ControlNet models
        self.current_controlnet = None  # Currently active ControlNet
        self._initialized = False
    
    def setup(self):
        """
        Initialize the SDXL ControlNet pipeline.
        
        Loads the base model and prepares for ControlNet attachment.
        ControlNet models are loaded lazily on first use.
        """
        if self._initialized and self.pipeline is not None:
            logger.debug("[sdxl_controlnet] Pipeline already initialized")
            return
        
        try:
            model_id = settings.sdxl_base_model_id
            logger.info(f"[sdxl_controlnet] Loading SDXL base model: {model_id}")
            self.check_model_cached(model_id)
            
            # Start with no ControlNet attached (will attach on first use)
            self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_id,
                controlnet=None,  # No ControlNet initially
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
            
            # Disable safety checker
            self.pipeline.safety_checker = None
            
            self._initialized = True
            logger.info("[sdxl_controlnet] ✓ SDXL ControlNet pipeline initialized")
            
        except Exception as e:
            logger.error(f"[sdxl_controlnet] ✗ Failed to initialize: {e}", exc_info=True)
            raise
    
    def set_controlnet(self, controlnet_type: ControlNetType) -> bool:
        """
        Switch to a specific ControlNet type.
        
        Loads the ControlNet model if not already cached, then attaches it to the pipeline.
        
        Args:
            controlnet_type: ControlNetType.CANNY or ControlNetType.LINEART
        
        Returns:
            True if successfully set, False if invalid type
        
        Raises:
            RuntimeError: If pipeline is not initialized
        """
        if not self._initialized or self.pipeline is None:
            raise RuntimeError("[sdxl_controlnet] Pipeline not initialized. Call setup() first.")
        
        if controlnet_type == ControlNetType.NONE:
            self.pipeline.controlnet = None
            self.current_controlnet = None
            logger.debug("[sdxl_controlnet] ControlNet disabled")
            return True
        
        # Check if valid type
        if controlnet_type not in self.CONTROLNET_MODELS:
            logger.error(f"[sdxl_controlnet] Unsupported ControlNet type: {controlnet_type}")
            return False
        
        # Return early if already loaded
        if controlnet_type == self.current_controlnet:
            logger.debug(f"[sdxl_controlnet] ControlNet {controlnet_type} already active")
            return True
        
        try:
            # Load from cache if available, otherwise load from HuggingFace
            if controlnet_type not in self.controlnet_models:
                model_id = self.CONTROLNET_MODELS[controlnet_type]
                logger.info(f"[sdxl_controlnet] Loading ControlNet: {controlnet_type} ({model_id})")
                self.check_model_cached(model_id)
                
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    variant="fp16" if settings.use_fp16 else None,
                    cache_dir=settings.cache_dir,
                )
                
                controlnet = self.to_device(controlnet)
                self.controlnet_models[controlnet_type] = controlnet
                logger.debug(f"[sdxl_controlnet] ControlNet {controlnet_type} loaded")
            
            # Attach to pipeline
            self.pipeline.controlnet = self.controlnet_models[controlnet_type]
            self.current_controlnet = controlnet_type
            logger.info(f"[sdxl_controlnet] ✓ ControlNet {controlnet_type} activated")
            return True
            
        except Exception as e:
            logger.error(f"[sdxl_controlnet] Failed to load ControlNet {controlnet_type}: {e}", exc_info=True)
            return False
    
    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        controlnet_type: ControlNetType,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.9,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate image conditioned on a control image (sketch, canny edges, etc.).
        
        Args:
            prompt: Text prompt to guide generation
            control_image: PIL Image for ControlNet conditioning
            controlnet_type: Type of ControlNet to use (CANNY, LINEART)
            negative_prompt: Negative prompt to avoid
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Text guidance scale
            controlnet_scale: ControlNet conditioning scale
            seed: Random seed for reproducibility
        
        Returns:
            Generated PIL Image
        
        Raises:
            RuntimeError: If pipeline not initialized or ControlNet loading fails
            ValueError: If controlnet_type is invalid
        """
        if not self._initialized or self.pipeline is None:
            raise RuntimeError("[sdxl_controlnet] Pipeline not initialized. Call setup() first.")
        
        if not isinstance(controlnet_type, ControlNetType):
            raise ValueError(f"Invalid ControlNetType: {controlnet_type}")
        
        # Set ControlNet
        if not self.set_controlnet(controlnet_type):
            raise RuntimeError(f"[sdxl_controlnet] Failed to set ControlNet: {controlnet_type}")
        
        # Normalize dimensions
        height = (height // 8) * 8
        width = (width // 8) * 8
        
        # Resize control image to match output dimensions
        control_image_resized = control_image.resize((width, height), Image.Resampling.LANCZOS)
        
        logger.debug(
            f"[sdxl_controlnet] Generating: prompt='{prompt[:60]}...', "
            f"controlnet={controlnet_type}, size={width}x{height}, "
            f"steps={num_inference_steps}, guidance={guidance_scale}, controlnet_scale={controlnet_scale}"
        )
        
        generator = self.get_generator(seed)
        
        try:
            with self.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    image=control_image_resized,
                    controlnet_conditioning_scale=controlnet_scale,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            
            logger.debug("[sdxl_controlnet] ✓ Generation completed")
            return output.images[0]
            
        except Exception as e:
            logger.error(f"[sdxl_controlnet] Generation failed: {e}", exc_info=True)
            raise
    
    def cleanup(self):
        """
        Clean up pipeline and ControlNet resources.
        
        Moves models to CPU and clears CUDA cache.
        """
        if self.pipeline is not None:
            try:
                self.pipeline = self.pipeline.to("cpu")
                logger.debug("[sdxl_controlnet] Pipeline moved to CPU")
            except Exception as e:
                logger.warning(f"[sdxl_controlnet] Error moving pipeline to CPU: {e}")
        
        for controlnet_type, controlnet in self.controlnet_models.items():
            try:
                controlnet = controlnet.to("cpu")
                logger.debug(f"[sdxl_controlnet] ControlNet {controlnet_type} moved to CPU")
            except Exception as e:
                logger.warning(f"[sdxl_controlnet] Error moving ControlNet to CPU: {e}")
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("[sdxl_controlnet] CUDA cache cleared")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"[sdxl_controlnet] Error during __del__: {e}")

"""
SDXL + ControlNet Pipeline: Sketch-to-image and conditional generation.

Uses StableDiffusionXLControlNetPipeline for ControlNet integration.
Supports canny edge detection and line art / sketch inputs.

Memory management features:
- torch.inference_mode() for gradient-free inference
- Post-inference cleanup
- Proper ControlNet model switching with memory cleanup
- Control image cleanup after use
"""

import logging
import gc
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
            
            # Disable safety checker (not needed for inference, saves memory)
            self.pipeline.safety_checker = None
            logger.debug("[sdxl_controlnet] Safety checker disabled")
            
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
        
        generator = None
        output = None
        result_image = None
        
        try:
            # Create generator for reproducibility
            generator = self.get_generator(seed)
            
            # CRITICAL: Use inference_mode() which handles ALL cleanup
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
                
                # Extract image from output IMMEDIATELY
                result_image = output.images[0]
                logger.debug("[sdxl_controlnet] ✓ Image extracted from output")
            
            # Explicitly delete intermediate objects
            del output
            del generator
            del control_image_resized
            logger.debug("[sdxl_controlnet] Output, generator, and control image deleted")
            
            # Clear any remaining references
            self.clear_tensor_references()
            
            return result_image
            
        except Exception as e:
            logger.error(f"[sdxl_controlnet] Generation failed: {e}", exc_info=True)
            # Ensure cleanup even on error
            try:
                del output
                del generator
                del control_image_resized
            except:
                pass
            self.clear_tensor_references()
            raise
    
    def cleanup(self):
        """
        Clean up pipeline and ControlNet resources.
        
        Moves models to CPU, deallocates ControlNets, and clears CUDA cache.
        Called during shutdown to free all VRAM.
        """
        logger.info("[sdxl_controlnet] Starting cleanup...")
        
        # 1. Move base pipeline to CPU
        if self.pipeline is not None:
            try:
                self.pipeline = self.pipeline.to("cpu")
                logger.debug("[sdxl_controlnet] Pipeline moved to CPU")
            except Exception as e:
                logger.warning(f"[sdxl_controlnet] Error moving pipeline to CPU: {e}")
        
        # 2. Move and delete ControlNet models
        for controlnet_type, controlnet in self.controlnet_models.items():
            try:
                if controlnet is not None:
                    controlnet = controlnet.to("cpu")
                    del controlnet
                    logger.debug(f"[sdxl_controlnet] ControlNet {controlnet_type} deleted")
            except Exception as e:
                logger.warning(f"[sdxl_controlnet] Error cleaning up ControlNet {controlnet_type}: {e}")
        
        # 3. Clear the ControlNet cache
        self.controlnet_models.clear()
        self.current_controlnet = None
        logger.debug("[sdxl_controlnet] ControlNet cache cleared")
        
        # 4. Clear CUDA caches
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.debug("[sdxl_controlnet] CUDA cache cleared and IPC collected")
            except Exception as e:
                logger.warning(f"[sdxl_controlnet] Error during CUDA cleanup: {e}")
        
        # 5. Force garbage collection
        gc.collect()
        logger.info("[sdxl_controlnet] ✓ Cleanup completed")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"[sdxl_controlnet] Error during __del__: {e}")

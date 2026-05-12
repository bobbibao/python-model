"""
Generation Service: Unified orchestration for all image generation modes.

Coordinates:
- Pipeline selection and execution
- LoRA loading/unloading
- ControlNet preprocessing
- Request validation and response building
- Memory cleanup after generation
"""

import logging
import gc
import time
import torch
from typing import Optional, Tuple
from PIL import Image

from ..pipelines.pipeline_registry import get_registry
from ..services.lora_service import get_lora_service
from ..services.controlnet_service import get_controlnet_service
from ..services.prompt_service import get_prompt_service
from ..models.enums import GenerationMode, ControlNetType
from ..models.configs import LoRAConfig, ControlNetConfig, PipelineInferenceConfig
from ..core import get_device

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Central orchestration service for image generation.
    
    Responsibilities:
    - Route requests to appropriate pipeline
    - Manage LoRA loading per request
    - Handle ControlNet conditioning
    - Validate inputs and execute inference
    - Return structured results with metadata
    """
    
    _instance: Optional["GenerationService"] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize GenerationService."""
        if hasattr(self, "_initialized"):
            return
        
        self.registry = get_registry()
        self.lora_service = get_lora_service()
        self.controlnet_service = get_controlnet_service()
        self.prompt_service = get_prompt_service()
        self._initialized = True
        
        logger.info("🔄 GenerationService initialized")
    
    def generate(
        self,
        mode: GenerationMode,
        prompt: str,
        negative_prompt: str = "",
        image: Optional[Image.Image] = None,
        strength: float = 0.5,
        controlnet_config: Optional[ControlNetConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        inference_config: Optional[PipelineInferenceConfig] = None,
    ) -> Tuple[Image.Image, dict]:
        """
        Generate an image based on the specified mode and parameters.
        
        Unified entry point that routes to appropriate generation method.
        
        Args:
            mode: GenerationMode (TEXT_TO_IMAGE, IMAGE_TO_IMAGE, SKETCH_TO_IMAGE)
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (what to avoid)
            image: Input image (for image-to-image and sketch modes)
            strength: Blend strength for image-to-image (0.0 to 1.0)
            controlnet_config: ControlNet configuration if using ControlNet
            lora_config: LoRA configuration if using LoRA
            inference_config: Inference parameters (steps, guidance, etc.)
        
        Returns:
            Tuple of (generated_image, metadata_dict)
            
            Metadata includes:
            - inference_time_ms: Time taken for generation
            - device: Device used (cuda/cpu)
            - mode: Generation mode used
            - lora_applied: Whether LoRA was used
            - controlnet_used: ControlNet type used (if any)
        """
        start_time = time.time()
        metadata = {
            "mode": mode.value,
            "device": get_device(),
            "lora_applied": False,
            "controlnet_used": None,
        }
        
        try:
            # Validate inputs
            is_valid, msg = self.prompt_service.validate_prompt(prompt)
            if not is_valid:
                raise ValueError(f"Invalid prompt: {msg}")
            
            # Use defaults if not provided
            if inference_config is None:
                from ..config import settings
                inference_config = PipelineInferenceConfig(
                    num_inference_steps=settings.num_inference_steps,
                    guidance_scale=settings.guidance_scale,
                    height=settings.default_height,
                    width=settings.default_width,
                )
            
            # Normalize dimensions
            inference_config.normalize_dimensions()
            
            if not inference_config.is_valid():
                raise ValueError("Invalid inference configuration")
            
            # Normalize prompts
            prompt, negative_prompt = self.prompt_service.combine_prompts(
                prompt, negative_prompt
            )
            
            # Route to appropriate generation method
            if mode == GenerationMode.TEXT_TO_IMAGE:
                result_image = self._generate_text_to_image(
                    prompt, negative_prompt, inference_config, lora_config
                )
                metadata["lora_applied"] = bool(lora_config and lora_config.enabled)
                
            elif mode == GenerationMode.IMAGE_TO_IMAGE:
                if image is None:
                    raise ValueError("image_to_image requires input image")
                result_image = self._generate_image_to_image(
                    prompt, negative_prompt, image, strength, inference_config, lora_config
                )
                metadata["lora_applied"] = bool(lora_config and lora_config.enabled)
                
            elif mode == GenerationMode.SKETCH_TO_IMAGE:
                if image is None:
                    raise ValueError("sketch_to_image requires input image")
                if controlnet_config is None or not controlnet_config.enabled:
                    raise ValueError("sketch_to_image requires ControlNet configuration")
                result_image = self._generate_sketch_to_image(
                    prompt, negative_prompt, image, controlnet_config, inference_config, lora_config
                )
                metadata["controlnet_used"] = controlnet_config.type
                metadata["lora_applied"] = bool(lora_config and lora_config.enabled)
                
            elif mode == GenerationMode.INPAINT:
                if image is None:
                    raise ValueError("inpaint requires input image")
                # For now, inpaint uses same pipeline as text-to-image
                result_image = self._generate_text_to_image(
                    prompt, negative_prompt, inference_config, lora_config
                )
                metadata["lora_applied"] = bool(lora_config and lora_config.enabled)
                
            else:
                raise ValueError(f"Unsupported generation mode: {mode}")
            
            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000
            metadata["inference_time_ms"] = round(inference_time_ms, 2)
            
            logger.info(
                f"[generation] ✓ {mode.value} completed in {metadata['inference_time_ms']}ms"
            )
            
            return result_image, metadata
            
        except Exception as e:
            logger.error(f"[generation] ✗ Generation failed: {e}", exc_info=True)
            raise
    
    def _generate_text_to_image(
        self,
        prompt: str,
        negative_prompt: str,
        inference_config: PipelineInferenceConfig,
        lora_config: Optional[LoRAConfig] = None,
    ) -> Image.Image:
        """Generate image from text prompt."""
        pipeline = self.registry.get_pipeline(GenerationMode.TEXT_TO_IMAGE)
        
        # Load LoRA if needed
        if lora_config and lora_config.enabled:
            self.lora_service.load_lora(
                pipeline.pipeline,
                lora_config.path,
                lora_scale=lora_config.scale,
            )
        
        try:
            image = pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=inference_config.height,
                width=inference_config.width,
                num_inference_steps=inference_config.num_inference_steps,
                guidance_scale=inference_config.guidance_scale,
                seed=inference_config.seed,
            )
            return image
        finally:
            # Unload LoRA after generation
            if lora_config and lora_config.enabled:
                self.lora_service.unload_lora(pipeline.pipeline)
    
    def _generate_image_to_image(
        self,
        prompt: str,
        negative_prompt: str,
        image: Image.Image,
        strength: float,
        inference_config: PipelineInferenceConfig,
        lora_config: Optional[LoRAConfig] = None,
    ) -> Image.Image:
        """Generate image from text prompt + input image."""
        # For now, use base SDXL (img2img variant not yet implemented)
        # In production, would use dedicated img2img pipeline
        logger.warning("[generation] Using text-to-image pipeline for image-to-image (img2img pipeline TODO)")
        
        return self._generate_text_to_image(
            prompt, negative_prompt, inference_config, lora_config
        )
    
    def _generate_sketch_to_image(
        self,
        prompt: str,
        negative_prompt: str,
        sketch_image: Image.Image,
        controlnet_config: ControlNetConfig,
        inference_config: PipelineInferenceConfig,
        lora_config: Optional[LoRAConfig] = None,
    ) -> Image.Image:
        """Generate image from sketch using ControlNet."""
        # Get ControlNet pipeline
        controlnet_type = ControlNetType(controlnet_config.type)
        
        # Preprocess sketch image
        processed_image = self.controlnet_service.preprocess_for_controlnet(
            sketch_image,
            controlnet_type,
            target_height=inference_config.height,
            target_width=inference_config.width,
        )
        
        if processed_image is None:
            raise RuntimeError("Failed to preprocess sketch image for ControlNet")
        
        # Get ControlNet pipeline
        pipeline = self.registry.get_pipeline(GenerationMode.SKETCH_TO_IMAGE)
        
        result_image = None
        
        # Load LoRA if needed
        if lora_config and lora_config.enabled:
            self.lora_service.load_lora(
                pipeline.pipeline,
                lora_config.path,
                lora_scale=lora_config.scale,
            )
        
        try:
            result_image = pipeline.generate(
                prompt=prompt,
                control_image=processed_image,
                controlnet_type=controlnet_type,
                negative_prompt=negative_prompt,
                height=inference_config.height,
                width=inference_config.width,
                num_inference_steps=inference_config.num_inference_steps,
                guidance_scale=inference_config.guidance_scale,
                controlnet_scale=controlnet_config.scale,
                seed=inference_config.seed,
            )
            return result_image
        finally:
            # CRITICAL: Cleanup in proper order
            
            # 1. Unload LoRA after generation
            if lora_config and lora_config.enabled:
                self.lora_service.unload_lora(pipeline.pipeline)
            
            # 2. Delete processed image (can be large)
            try:
                del processed_image
            except:
                pass
            
            # 3. Force garbage collection
            gc.collect()
            logger.debug("[generation] Sketch-to-image cleanup completed")
    
    def get_status(self) -> dict:
        """Get status of the generation service and all pipelines."""
        return {
            "service": "generation",
            "status": "ready",
            "pipelines": self.registry.get_status(),
            "lora_cache": self.lora_service.get_loaded_loras(),
            "controlnet_types": list(self.controlnet_service.get_available_types()),
        }


def get_generation_service() -> GenerationService:
    """Get the global GenerationService singleton."""
    return GenerationService()

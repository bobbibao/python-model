"""
Configuration dataclasses for pipelines and inference.

These are more flexible than Pydantic for internal use.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    
    enabled: bool = False
    """Whether to use LoRA for this generation."""
    
    path: str = ""
    """Path to LoRA adapter (e.g., 'house_lora_final')."""
    
    scale: float = 1.0
    """LoRA weight scaling (0.0 to 1.0+)."""
    
    def is_valid(self) -> bool:
        """Check if LoRA config is valid."""
        if not self.enabled:
            return True
        return len(self.path) > 0 and 0.0 <= self.scale <= 2.0


@dataclass
class ControlNetConfig:
    """ControlNet configuration."""
    
    enabled: bool = False
    """Whether to use ControlNet."""
    
    type: str = "none"
    """ControlNet type: 'canny', 'lineart', or 'none'."""
    
    scale: float = 1.0
    """ControlNet weight (0.0 to 1.0+)."""
    
    def is_valid(self) -> bool:
        """Check if ControlNet config is valid."""
        if not self.enabled:
            return True
        return self.type in ("canny", "lineart") and 0.0 <= self.scale <= 2.0


@dataclass
class PipelineInferenceConfig:
    """Inference-time configuration for pipelines."""
    
    num_inference_steps: int = 30
    """Number of denoising steps (higher = better quality but slower)."""
    
    guidance_scale: float = 7.5
    """Classifier-free guidance scale (higher = more prompt adherence)."""
    
    height: int = 1024
    """Output image height (must be multiple of 8)."""
    
    width: int = 1024
    """Output image width (must be multiple of 8)."""
    
    seed: Optional[int] = None
    """Random seed for reproducibility."""
    
    negative_prompt: str = ""
    """Negative prompt to avoid in generation."""
    
    num_images_per_prompt: int = 1
    """Number of images to generate per prompt."""
    
    controlnet_guidance_scale: float = 0.9
    """ControlNet guidance scale (if using ControlNet)."""
    
    def normalize_dimensions(self):
        """Ensure dimensions are multiples of 8."""
        self.height = (self.height // 8) * 8
        self.width = (self.width // 8) * 8
    
    def is_valid(self) -> bool:
        """Check if inference config is valid."""
        if self.num_inference_steps < 1 or self.num_inference_steps > 150:
            return False
        if self.guidance_scale < 0.0 or self.guidance_scale > 25.0:
            return False
        if self.height < 256 or self.height > 4096:
            return False
        if self.width < 256 or self.width > 4096:
            return False
        if self.num_images_per_prompt < 1 or self.num_images_per_prompt > 4:
            return False
        return True

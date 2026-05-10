"""
Pydantic schemas for v2 API request/response validation.

Clean, type-safe request/response contracts for /api/v2/generate endpoint.
"""

from typing import Optional
from pydantic import BaseModel, Field
from ..models.enums import GenerationMode, ControlNetType


# Request sub-schemas


class ControlNetRequest(BaseModel):
    """ControlNet configuration for a generation request."""
    
    controlnet_type: ControlNetType = Field(
        ControlNetType.NONE,
        description="ControlNet type: canny, lineart, or none"
    )
    """Type of ControlNet to use."""
    
    controlnet_scale: float = Field(
        0.9,
        ge=0.0,
        le=2.0,
        description="ControlNet guidance scale (0.0 to 2.0)"
    )
    """Strength of ControlNet conditioning."""


class LoRARequest(BaseModel):
    """LoRA configuration for a generation request."""
    
    lora_enabled: bool = Field(
        False,
        description="Whether to use LoRA adapter"
    )
    """Enable LoRA."""
    
    lora_scale: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="LoRA weight scale (0.0 to 2.0)"
    )
    """LoRA weight scaling."""
    
    lora_path: Optional[str] = Field(
        None,
        description="Path to LoRA adapter (uses default if None)"
    )
    """Optional override for LoRA path."""


class InferenceParams(BaseModel):
    """Inference-time parameters for diffusion process."""
    
    num_inference_steps: int = Field(
        30,
        ge=1,
        le=150,
        description="Number of denoising steps"
    )
    """Denoising steps (higher = better quality but slower)."""
    
    guidance_scale: float = Field(
        7.5,
        ge=0.0,
        le=25.0,
        description="Classifier-free guidance scale"
    )
    """Guidance strength (higher = more prompt adherence)."""
    
    height: int = Field(
        1024,
        ge=256,
        le=4096,
        multiple_of=8,
        description="Image height (must be multiple of 8)"
    )
    """Output image height."""
    
    width: int = Field(
        1024,
        ge=256,
        le=4096,
        multiple_of=8,
        description="Image width (must be multiple of 8)"
    )
    """Output image width."""
    
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility"
    )
    """Optional seed for deterministic generation."""
    
    negative_prompt: str = Field(
        "",
        max_length=4000,
        description="Negative prompt (what to avoid)"
    )
    """Negative prompt."""


# Main request schema


class GenerateRequest(BaseModel):
    """Unified request schema for /api/v2/generate endpoint."""
    
    mode: GenerationMode = Field(
        GenerationMode.TEXT_TO_IMAGE,
        description="Generation mode: text_to_image, image_to_image, or sketch_to_image"
    )
    """Generation mode."""
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Text prompt for generation"
    )
    """Main prompt."""
    
    image: Optional[str] = Field(
        None,
        description="Base64-encoded input image (for image_to_image or sketch_to_image modes)"
    )
    """Input image as base64 string."""
    
    strength: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Blend strength for image_to_image (0.0 = no change, 1.0 = full replacement)"
    )
    """Strength for image-to-image mode."""
    
    controlnet: Optional[ControlNetRequest] = Field(
        None,
        description="ControlNet configuration (for sketch_to_image mode)"
    )
    """ControlNet configuration."""
    
    lora: Optional[LoRARequest] = Field(
        None,
        description="LoRA configuration"
    )
    """LoRA configuration."""
    
    inference: InferenceParams = Field(
        default_factory=InferenceParams,
        description="Inference parameters"
    )
    """Inference settings."""
    
    class Config:
        json_schema_extra = {
            "example": {
                "mode": "text_to_image",
                "prompt": "photorealistic modern house, architectural photograph",
                "inference": {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "height": 1024,
                    "width": 1024,
                    "seed": 42,
                    "negative_prompt": "blurry, low quality, distorted"
                }
            }
        }


# Response schemas


class ImageMetadata(BaseModel):
    """Metadata about generated image."""
    
    mode: str = Field(description="Generation mode used")
    inference_time_ms: float = Field(description="Time taken for generation in milliseconds")
    device: str = Field(description="Device used (cuda or cpu)")
    lora_applied: bool = Field(description="Whether LoRA was applied")
    controlnet_used: Optional[str] = Field(None, description="ControlNet type used (if any)")


class GenerateResponse(BaseModel):
    """Response schema for /api/v2/generate endpoint."""
    
    job_id: str = Field(
        description="Unique identifier for this generation job"
    )
    """Job ID (UUID)."""
    
    status: str = Field(
        default="completed",
        description="Job status (completed or error)"
    )
    """Status."""
    
    image_url: str = Field(
        description="URL to download generated image"
    )
    """Image download URL."""
    
    width: int = Field(
        description="Output image width in pixels"
    )
    """Image width."""
    
    height: int = Field(
        description="Output image height in pixels"
    )
    """Image height."""
    
    metadata: ImageMetadata = Field(
        description="Additional metadata about generation"
    )
    """Generation metadata."""


# Error response


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    status: str = Field(default="error")
    """Status."""
    
    detail: str = Field(
        description="Error message"
    )
    """Error detail."""
    
    error_code: Optional[str] = Field(
        None,
        description="Error code for programmatic handling"
    )
    """Error code."""

"""
FastAPI v2 routes: unified /api/v2/generate endpoint.

Single entry point for all generation modes with clean separation of concerns.
"""

import logging
import uuid
import traceback
from fastapi import APIRouter, HTTPException, status

from .models import GenerateRequest, GenerateResponse, ErrorResponse, ImageMetadata
from ...services.generation_service import get_generation_service
from ...utils.image_utils import decode_base64_image, save_generated_image, encode_image_to_base64
from ...utils.memory_utils import log_memory_stats, warn_if_low_memory
from ...models.enums import GenerationMode, ControlNetType
from ...models.configs import LoRAConfig, ControlNetConfig, PipelineInferenceConfig
from ...config import settings
from pathlib import Path

logger = logging.getLogger(__name__)

# Create router with v2 prefix
router = APIRouter(prefix="/api/v2", tags=["v2"])

# Get singleton generation service
generation_service = get_generation_service()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate images using SDXL with optional ControlNet and LoRA",
    description="""
    Unified endpoint for image generation supporting:
    - Text-to-image: Generate from text prompt only
    - Image-to-image: Modify existing image with text guidance
    - Sketch-to-image: Generate from sketch using ControlNet conditioning
    
    Supports optional LoRA fine-tuning for custom styles.
    """,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Image generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """
    Generate image using SDXL with optional ControlNet and LoRA.
    
    Request body should be JSON with the following structure:
    
    ```json
    {
        "mode": "text_to_image",
        "prompt": "photorealistic modern house",
        "image": null,
        "strength": 0.5,
        "controlnet": null,
        "lora": null,
        "inference": {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
            "seed": 42,
            "negative_prompt": "blurry, low quality"
        }
    }
    ```
    """
    job_id = str(uuid.uuid4())
    
    try:
        # Log memory at start
        log_memory_stats(f"start_{request.mode.value}")
        warn_if_low_memory(threshold_mb=1000)
        
        logger.info(f"[{job_id}] Received generation request: mode={request.mode.value}")
        logger.debug(f"[{job_id}] Prompt: {request.prompt[:60]}...")
        
        # ============================================
        # INPUT PREPARATION
        # ============================================
        
        # Decode input image if provided
        input_image = None
        if request.image is not None:
            logger.debug(f"[{job_id}] Decoding input image...")
            input_image = decode_base64_image(request.image)
            if input_image is None:
                raise ValueError("Failed to decode input image")
            logger.debug(f"[{job_id}] Image decoded: {input_image.size}")
        
        # ============================================
        # CONFIG PREPARATION
        # ============================================
        
        # LoRA configuration
        lora_config = None
        if request.lora and request.lora.lora_enabled:
            lora_path = request.lora.lora_path or settings.default_lora_path
            lora_config = LoRAConfig(
                enabled=True,
                path=lora_path,
                scale=request.lora.lora_scale,
            )
            logger.debug(f"[{job_id}] LoRA enabled: {lora_path} (scale={request.lora.lora_scale})")
        
        # ControlNet configuration
        controlnet_config = None
        if request.controlnet and request.controlnet.controlnet_type != ControlNetType.NONE:
            controlnet_config = ControlNetConfig(
                enabled=True,
                type=request.controlnet.controlnet_type.value,
                scale=request.controlnet.controlnet_scale,
            )
            logger.debug(
                f"[{job_id}] ControlNet: {controlnet_config.type} "
                f"(scale={controlnet_config.scale})"
            )
        
        # Inference configuration
        inference_config = PipelineInferenceConfig(
            num_inference_steps=request.inference.num_inference_steps,
            guidance_scale=request.inference.guidance_scale,
            height=request.inference.height,
            width=request.inference.width,
            seed=request.inference.seed,
            negative_prompt=request.inference.negative_prompt,
        )
        
        # ============================================
        # GENERATION
        # ============================================
        
        logger.info(f"[{job_id}] Starting generation...")
        
        result_image, metadata = generation_service.generate(
            mode=request.mode,
            prompt=request.prompt,
            negative_prompt=request.inference.negative_prompt,
            image=input_image,
            strength=request.strength,
            controlnet_config=controlnet_config,
            lora_config=lora_config,
            inference_config=inference_config,
        )
        
        logger.info(f"[{job_id}] ✓ Generation completed in {metadata['inference_time_ms']}ms")
        
        # ============================================
        # OUTPUT PREPARATION
        # ============================================
        
        # Save generated image
        output_dir = Path(settings.output_dir)
        output_path = save_generated_image(result_image, output_dir, job_id)
        
        # Build image URL
        image_url = f"{settings.base_url}/outputs/{job_id}.png"
        
        logger.debug(f"[{job_id}] Image saved: {output_path}")
        
        # ============================================
        # RESPONSE
        # ============================================
        
        response = GenerateResponse(
            job_id=job_id,
            status="completed",
            image_url=image_url,
            width=result_image.width,
            height=result_image.height,
            metadata=ImageMetadata(
                mode=metadata["mode"],
                inference_time_ms=metadata["inference_time_ms"],
                device=metadata["device"],
                lora_applied=metadata["lora_applied"],
                controlnet_used=metadata["controlnet_used"],
            ),
        )
        
        log_memory_stats(f"end_{request.mode.value}")
        logger.info(f"[{job_id}] ✓ Request completed successfully")
        
        return response
    
    except ValueError as e:
        # Validation errors
        logger.warning(f"[{job_id}] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    
    except RuntimeError as e:
        # Pipeline/model loading errors
        logger.error(f"[{job_id}] Runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline initialization failed: {str(e)}",
        )
    
    except Exception as e:
        # Generic errors
        logger.error(f"[{job_id}] Unexpected error: {e}", exc_info=True)
        logger.error(f"[{job_id}] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}",
        )


@router.get(
    "/status",
    summary="Get service status",
    description="Get status of generation service, pipelines, and LoRA cache",
)
async def get_status() -> dict:
    """
    Get current service status including pipeline state and cache info.
    
    Returns information about:
    - Pipelines (loaded, device, dtype)
    - LoRA cache (loaded adapters)
    - Available ControlNet types
    """
    try:
        return generation_service.get_status()
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve status: {str(e)}",
        )

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    """
    Application settings with support for both legacy SD v1.5 and new SDXL + ControlNet architecture.
    
    Environment variable precedence:
    - HF_HOME / TRANSFORMERS_CACHE for cache directory
    - SERVICE_* for service config
    - SDXL_* for SDXL model config
    - CONTROLNET_* for ControlNet config
    - LORA_* for LoRA config
    - Sensible defaults otherwise
    """
    
    # =====================================================================
    # SERVICE CONFIGURATION
    # =====================================================================
    service_name: str = os.getenv("SERVICE_NAME", "vizera-python-model")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    base_url: str = os.getenv("BASE_URL", "http://localhost:8001")
    
    # =====================================================================
    # LEGACY MODEL CONFIGURATION (SD v1.5) - for backward compatibility
    # =====================================================================
    model_id: str = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
    inpaint_model_id: str = os.getenv("INPAINT_MODEL_ID", "runwayml/stable-diffusion-v1-5-inpainting")
    
    # =====================================================================
    # SDXL MODEL CONFIGURATION (New Architecture)
    # =====================================================================
    sdxl_base_model_id: str = os.getenv(
        "SDXL_BASE_MODEL_ID", 
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
    """Base SDXL model for all generation tasks."""
    
    sdxl_refiner_model_id: str = os.getenv(
        "SDXL_REFINER_MODEL_ID",
        "stabilityai/stable-diffusion-xl-refiner-1.0"
    )
    """Optional SDXL refiner for 2-stage generation (not used by default)."""
    
    vae_model_id: str = os.getenv(
        "VAE_MODEL_ID",
        "madebyollin/sdxl-vae-fp16-fix"
    )
    """Custom VAE for SDXL (optional, uses default SDXL VAE if not specified)."""
    
    # =====================================================================
    # CONTROLNET CONFIGURATION
    # =====================================================================
    controlnet_canny_model_id: str = os.getenv(
        "CONTROLNET_CANNY_MODEL_ID",
        "lllyasviel/sd-xl-controlnet-canny"
    )
    """ControlNet model for canny edge detection."""
    
    controlnet_lineart_model_id: str = os.getenv(
        "CONTROLNET_LINEART_MODEL_ID",
        "lllyasviel/sd-xl-controlnet-lineart"
    )
    """ControlNet model for line art / sketch-to-image."""
    
    controlnet_default_guidance_scale: float = float(
        os.getenv("CONTROLNET_GUIDANCE_SCALE", "0.9")
    )
    """Default ControlNet guidance scale."""
    
    # =====================================================================
    # LORA CONFIGURATION
    # =====================================================================
    default_lora_path: str = os.getenv(
        "DEFAULT_LORA_PATH",
        "house_lora_final"
    )
    """Path to default trained LoRA adapter."""
    
    lora_cache_size: int = int(
        os.getenv("LORA_CACHE_SIZE", "3")
    )
    """Maximum number of LoRA adapters to keep in memory (LRU eviction)."""
    
    lora_cache_memory_limit_mb: int = int(
        os.getenv("LORA_CACHE_MEMORY_LIMIT_MB", "2000")
    )
    """Maximum memory for LoRA cache in MB."""
    
    # =====================================================================
    # CACHE CONFIGURATION
    # =====================================================================
    cache_dir: str = os.getenv(
        "HF_HOME",
        os.getenv("TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface"))
    )
    """HuggingFace model cache directory."""
    
    # =====================================================================
    # MEMORY OPTIMIZATION CONFIGURATION
    # =====================================================================
    enable_attention_slicing: bool = os.getenv(
        "ENABLE_ATTENTION_SLICING", "true"
    ).lower() in ("true", "1", "yes")
    """Enable attention slicing to reduce VRAM peak usage."""
    
    enable_vae_slicing: bool = os.getenv(
        "ENABLE_VAE_SLICING", "true"
    ).lower() in ("true", "1", "yes")
    """Enable VAE slicing to reduce VRAM peak usage during VAE decode."""
    
    enable_vae_tiling: bool = os.getenv(
        "ENABLE_VAE_TILING", "false"
    ).lower() in ("true", "1", "yes")
    """Enable VAE tiling (more memory efficient but slower decoding)."""
    
    enable_model_cpu_offload: bool = os.getenv(
        "ENABLE_MODEL_CPU_OFFLOAD", "true"
    ).lower() in ("true", "1", "yes")
    """Enable model CPU offload (move models between GPU/CPU as needed)."""
    
    use_fp16: bool = os.getenv(
        "USE_FP16", "true"
    ).lower() in ("true", "1", "yes")
    """Use float16 dtype for GPU inference (faster, less VRAM)."""
    
    low_cpu_mem_usage: bool = os.getenv(
        "LOW_CPU_MEM_USAGE", "true"
    ).lower() in ("true", "1", "yes")
    """Use low CPU memory mode when loading models."""
    
    max_vram_mb: int = int(
        os.getenv("MAX_VRAM_MB", "11000")
    )
    """Target maximum VRAM usage in MB (for monitoring/limits)."""
    
    # =====================================================================
    # INFERENCE CONFIGURATION (Defaults)
    # =====================================================================
    num_inference_steps: int = int(
        os.getenv("NUM_INFERENCE_STEPS", "30")
    )
    """Default number of diffusion steps (can be overridden per request)."""
    
    guidance_scale: float = float(
        os.getenv("GUIDANCE_SCALE", "7.5")
    )
    """Default classifier-free guidance scale."""
    
    default_height: int = int(
        os.getenv("DEFAULT_HEIGHT", "1024")
    )
    """Default image height."""
    
    default_width: int = int(
        os.getenv("DEFAULT_WIDTH", "1024")
    )
    """Default image width."""
    
    # =====================================================================
    # LOGGING
    # =====================================================================
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    """Logging level (DEBUG, INFO, WARNING, ERROR)."""
    
    class Config:
        # Pydantic v2 config
        extra = "ignore"


settings = Settings()

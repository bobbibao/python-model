import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
class Settings(BaseModel):
    # Service configuration
    service_name: str = os.getenv("SERVICE_NAME", "vizera-python-model")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    base_url: str = os.getenv("BASE_URL", "http://localhost:8001")
    
    # Model configuration
    model_id: str = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
    inpaint_model_id: str = os.getenv("INPAINT_MODEL_ID", "runwayml/stable-diffusion-v1-5-inpainting")
    
    # Cache configuration
    cache_dir: str = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface")))
    
    # Model loading configuration
    preload_model: bool = os.getenv("PRELOAD_MODEL", "false").lower() in ("true", "1", "yes")
    enable_attention_slicing: bool = os.getenv("ENABLE_ATTENTION_SLICING", "true").lower() in ("true", "1", "yes")
    enable_vae_slicing: bool = os.getenv("ENABLE_VAE_SLICING", "true").lower() in ("true", "1", "yes")
    low_cpu_mem_usage: bool = os.getenv("LOW_CPU_MEM_USAGE", "true").lower() in ("true", "1", "yes")
    
    # Inference configuration
    num_inference_steps: int = int(os.getenv("NUM_INFERENCE_STEPS", "50"))
    guidance_scale: float = float(os.getenv("GUIDANCE_SCALE", "7.5"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        # Pydantic v2 config
        extra = "ignore"


settings = Settings()

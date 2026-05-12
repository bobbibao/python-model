"""
Base pipeline class: shared state and initialization logic.

Provides:
- Device and dtype management
- Common initialization/cleanup patterns
- Generator management for reproducibility
- VRAM profiling hooks
- Explicit memory cleanup after inference
"""

import logging
import gc
import torch
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

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
            
        NOTE: Generator will be used in inference_mode() context and cleaned up automatically.
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
            logger.debug(f"[{self.model_name}] Using seed: {seed}")
        return generator
    
    def apply_memory_optimizations(self, pipeline):
        """
        Apply memory optimization techniques to a pipeline.
        
        Optimizations applied (in order of impact on peak VRAM):
        1. Attention slicing: Process attention in slices (major VRAM reduction)
        2. VAE slicing: Process VAE in slices during encode/decode
        3. VAE tiling: Further reduce peak VRAM during VAE decode
        4. xFormers: Memory-efficient attention (requires xformers library)
        5. CPU offload: Move models between GPU/CPU (trades latency for VRAM)
        
        Args:
            pipeline: The diffusers pipeline to optimize.
        """
        # 1. ATTENTION SLICING (Major impact: ~30-40% peak memory reduction)
        try:
            if settings.enable_attention_slicing:
                pipeline.enable_attention_slicing()
                logger.debug(f"[{self.model_name}] ✓ Enabled attention slicing")
        except Exception as e:
            logger.warning(f"[{self.model_name}] Could not enable attention slicing: {e}")
        
        # 2. VAE SLICING (Process VAE in slices)
        try:
            if settings.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                logger.debug(f"[{self.model_name}] ✓ Enabled VAE slicing")
        except Exception as e:
            logger.warning(f"[{self.model_name}] Could not enable VAE slicing: {e}")
        
        # 3. VAE TILING (Process VAE in tiles for even lower peak VRAM)
        try:
            if settings.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
                logger.info(f"[{self.model_name}] ✓ Enabled VAE tiling (VRAM optimized, may be slower)")
        except Exception as e:
            logger.warning(f"[{self.model_name}] Could not enable VAE tiling: {e}")
        
        # 4. XFORMERS OPTIMIZATION (Memory-efficient attention)
        try:
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                pipeline.enable_xformers_memory_efficient_attention()
                logger.debug(f"[{self.model_name}] ✓ Enabled xFormers optimization")
        except Exception as e:
            # This is expected if xformers not installed
            logger.debug(f"[{self.model_name}] xFormers not available (install with: pip install xformers): {e}")
        
        # 5. CPU OFFLOAD (Move models between GPU/CPU as needed)
        # NOTE: Applied separately via apply_cpu_offload()
    
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
    
    @contextmanager
    def inference_mode(self):
        """
        Context manager for inference mode with automatic memory cleanup.
        
        Usage:
            with pipeline.inference_mode():
                output = pipeline(...)
        
        Benefits:
        - Disables gradient computation (saves VRAM)
        - Automatically cleans up after inference:
          * Deletes intermediate tensors
          * Runs garbage collection
          * Clears CUDA cache
          * Collects IPC memory
        
        This is CRITICAL for preventing VRAM fragmentation during long-running inference.
        """
        try:
            # Enter inference mode (no gradients)
            with torch.inference_mode():
                yield
        finally:
            # ============================================
            # POST-INFERENCE CLEANUP (CRITICAL!)
            # ============================================
            
            # 1. Force Python garbage collection
            gc.collect()
            logger.debug(f"[{self.model_name}] Garbage collected")
            
            # 2. Clear CUDA cache if on GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                logger.debug(f"[{self.model_name}] CUDA cache cleared")
                
                # 3. Collect IPC memory (for fragmentation prevention)
                try:
                    torch.cuda.ipc_collect()
                    logger.debug(f"[{self.model_name}] CUDA IPC collected")
                except Exception as e:
                    # ipc_collect might not be available in all torch versions
                    logger.debug(f"[{self.model_name}] IPC collect not available: {e}")
    
    def clear_tensor_references(self):
        """
        Explicitly delete intermediate tensor references.
        
        Called after inference to ensure no lingering tensors consume VRAM.
        Useful when inference leaves dangling references.
        """
        try:
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
                logger.debug(f"[{self.model_name}] Tensor references cleared")
        except Exception as e:
            logger.warning(f"[{self.model_name}] Error clearing tensor references: {e}")

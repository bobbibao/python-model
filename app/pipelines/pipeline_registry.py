"""
Pipeline Registry: Singleton manager for all pipeline instances.

Responsibilities:
- Lazy initialization of pipelines (only when first accessed)
- Thread-safe access (prevents duplicate model loading)
- Centralized lifecycle management (cleanup, status)
- Type-safe pipeline routing
"""

import logging
from threading import Lock
from typing import Dict, Optional

from .sdxl_pipeline import SDXLBasePipeline
from .sdxl_controlnet_pipeline import SDXLControlNetPipeline
from ..core import get_device, get_torch_dtype
from ..models.enums import GenerationMode

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """
    Singleton registry for managing SDXL pipeline instances.
    
    Thread-safe lazy initialization ensures models are loaded exactly once,
    preventing VRAM explosion and duplicate model loading.
    
    Usage:
        registry = PipelineRegistry()
        sdxl = registry.get_pipeline(GenerationMode.TEXT_TO_IMAGE)
        image = sdxl.generate(prompt=...)
    """
    
    _instance: Optional["PipelineRegistry"] = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize registry (called only once)."""
        if self._initialized:
            return
        
        self._pipelines: Dict[str, object] = {}
        self._pipeline_locks: Dict[str, Lock] = {}
        self._initialized = True
        
        logger.info("🔄 PipelineRegistry initialized (singleton)")
    
    def _ensure_lock(self, key: str) -> Lock:
        """Ensure a lock exists for the given pipeline key."""
        if key not in self._pipeline_locks:
            self._pipeline_locks[key] = Lock()
        return self._pipeline_locks[key]
    
    def get_pipeline(self, mode: GenerationMode):
        """
        Get a pipeline for the given generation mode.
        
        Lazy loads pipeline on first access. Subsequent calls return cached instance.
        Thread-safe via per-pipeline locks.
        
        Args:
            mode: GenerationMode enum value
        
        Returns:
            Initialized pipeline instance (SDXLBasePipeline or SDXLControlNetPipeline)
        
        Raises:
            ValueError: If mode is not supported
            RuntimeError: If pipeline initialization fails
        """
        if mode == GenerationMode.TEXT_TO_IMAGE or mode == GenerationMode.IMAGE_TO_IMAGE:
            return self._get_sdxl_base_pipeline()
        elif mode == GenerationMode.SKETCH_TO_IMAGE:
            return self._get_sdxl_controlnet_pipeline()
        elif mode == GenerationMode.INPAINT:
            # For now, inpaint also uses base SDXL
            # Future: could have dedicated inpaint pipeline
            return self._get_sdxl_base_pipeline()
        else:
            raise ValueError(f"Unsupported generation mode: {mode}")
    
    def _get_sdxl_base_pipeline(self) -> SDXLBasePipeline:
        """Get or initialize SDXL base pipeline (thread-safe)."""
        key = "sdxl_base"
        
        if key in self._pipelines:
            return self._pipelines[key]
        
        lock = self._ensure_lock(key)
        with lock:
            # Double-check after acquiring lock
            if key in self._pipelines:
                return self._pipelines[key]
            
            logger.info(f"🚀 Initializing {key} pipeline...")
            pipeline = SDXLBasePipeline()
            pipeline.setup()
            self._pipelines[key] = pipeline
            logger.info(f"✓ {key} pipeline ready")
            return pipeline
    
    def _get_sdxl_controlnet_pipeline(self) -> SDXLControlNetPipeline:
        """Get or initialize SDXL ControlNet pipeline (thread-safe)."""
        key = "sdxl_controlnet"
        
        if key in self._pipelines:
            return self._pipelines[key]
        
        lock = self._ensure_lock(key)
        with lock:
            # Double-check after acquiring lock
            if key in self._pipelines:
                return self._pipelines[key]
            
            logger.info(f"🚀 Initializing {key} pipeline...")
            pipeline = SDXLControlNetPipeline()
            pipeline.setup()
            self._pipelines[key] = pipeline
            logger.info(f"✓ {key} pipeline ready")
            return pipeline
    
    def get_all_pipelines(self) -> Dict[str, object]:
        """
        Get all initialized pipelines.
        
        Returns:
            Dict mapping pipeline names to instances (only initialized ones)
        """
        return dict(self._pipelines)
    
    def get_status(self) -> dict:
        """
        Get status of all pipelines for debugging and monitoring.
        
        Returns:
            Dict with pipeline status, device info, and model details
        """
        device = get_device()
        dtype = get_torch_dtype(device)
        
        status = {
            "device": device,
            "dtype": str(dtype),
            "pipelines": {},
        }
        
        for key, pipeline in self._pipelines.items():
            status["pipelines"][key] = {
                "initialized": True,
                "model_name": getattr(pipeline, "model_name", key),
                "device": getattr(pipeline, "device", device),
            }
        
        return status
    
    def cleanup_all(self):
        """
        Cleanup all pipeline resources.
        
        Called during application shutdown to properly release VRAM.
        """
        logger.info("🛑 Cleaning up all pipelines...")
        
        for key, pipeline in self._pipelines.items():
            try:
                if hasattr(pipeline, "cleanup"):
                    pipeline.cleanup()
                    logger.debug(f"✓ Cleaned up {key}")
            except Exception as e:
                logger.warning(f"⚠ Error cleaning up {key}: {e}")
        
        self._pipelines.clear()
        logger.info("✓ All pipelines cleaned up")
    
    def __del__(self):
        """Cleanup on deletion (safety measure)."""
        try:
            self.cleanup_all()
        except Exception as e:
            logger.warning(f"Error during PipelineRegistry cleanup: {e}")


# Global singleton instance
_registry: Optional[PipelineRegistry] = None


def get_registry() -> PipelineRegistry:
    """
    Get the global PipelineRegistry singleton.
    
    Returns:
        The PipelineRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PipelineRegistry()
    return _registry

"""
LoRA Service: Manages LoRA adapter loading, unloading, and caching.

Handles:
- Loading LoRA adapters from disk
- Fusing/unfusing adapters
- LRU-based memory management
- Per-request adapter switching
"""

import logging
from pathlib import Path
from typing import Optional, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)


class LoRAService:
    """
    Service for managing LoRA (Low-Rank Adaptation) adapters.
    
    Features:
    - Load adapters from local paths
    - Fuse/unfuse adapters into pipelines
    - LRU eviction for memory management
    - Singleton pattern (one service instance)
    """
    
    _instance: Optional["LoRAService"] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize LoRA service."""
        if hasattr(self, "_initialized"):
            return
        
        self._loaded_loras: Dict[str, object] = OrderedDict()  # name -> adapter config
        self._current_lora: Optional[str] = None
        self._max_cache_size = 3  # Max LoRAs in memory
        self._initialized = True
        
        logger.info("🔄 LoRAService initialized")
    
    def load_lora(
        self,
        pipeline,
        lora_path: str,
        lora_scale: float = 1.0,
        lora_name: Optional[str] = None,
    ) -> bool:
        """
        Load and fuse a LoRA adapter into the pipeline.
        
        Args:
            pipeline: The diffusers pipeline to load LoRA into
            lora_path: Path to LoRA adapter (e.g., "house_lora_final")
            lora_scale: Weight scale for the LoRA (0.0 to 2.0+)
            lora_name: Optional name for caching; if None uses lora_path
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            lora_id = lora_name or lora_path
            
            # Check if already loaded
            if lora_id in self._loaded_loras:
                logger.debug(f"[lora] LoRA {lora_id} already loaded")
                self._current_lora = lora_id
                return True
            
            # Validate path exists
            lora_dir = Path(lora_path)
            if not lora_dir.exists():
                logger.error(f"[lora] LoRA path does not exist: {lora_path}")
                return False
            
            logger.info(f"[lora] Loading LoRA from {lora_path}...")
            
            # Load using diffusers/peft
            try:
                # For SDXL with peft-based LoRA
                from peft import PeftConfig
                
                # Check if it's a valid LoRA adapter
                config = PeftConfig.from_pretrained(lora_path)
                logger.debug(f"[lora] LoRA config loaded: {config.peft_type}")
                
                # Use load_lora_weights if available (newer diffusers)
                if hasattr(pipeline, "load_lora_weights"):
                    pipeline.load_lora_weights(lora_path)
                    logger.info(f"[lora] ✓ LoRA loaded via load_lora_weights: {lora_id}")
                elif hasattr(pipeline, "load_adapter"):
                    pipeline.load_adapter(lora_path)
                    logger.info(f"[lora] ✓ LoRA loaded via load_adapter: {lora_id}")
                else:
                    logger.error(f"[lora] Pipeline does not support LoRA loading")
                    return False
                
                # Store in cache
                self._loaded_loras[lora_id] = {
                    "path": lora_path,
                    "scale": lora_scale,
                }
                self._current_lora = lora_id
                
                # Enforce max cache size with LRU eviction
                self._evict_lru()
                
                logger.info(f"[lora] ✓ LoRA {lora_id} loaded with scale={lora_scale}")
                return True
                
            except ImportError:
                logger.error("[lora] peft library not installed. Install via: pip install peft")
                return False
            except Exception as e:
                logger.error(f"[lora] Failed to load LoRA: {e}", exc_info=True)
                return False
        
        except Exception as e:
            logger.error(f"[lora] Unexpected error loading LoRA: {e}", exc_info=True)
            return False
    
    def unload_lora(self, pipeline) -> bool:
        """
        Unload and unfuse LoRA adapter from pipeline.
        
        Args:
            pipeline: The pipeline to unload LoRA from
        
        Returns:
            True if successfully unloaded, False otherwise
        """
        try:
            if self._current_lora is None:
                logger.debug("[lora] No LoRA currently loaded")
                return True
            
            logger.info(f"[lora] Unloading LoRA: {self._current_lora}...")
            
            # Unfuse if available
            if hasattr(pipeline, "unfuse_lora"):
                try:
                    pipeline.unfuse_lora()
                    logger.debug("[lora] LoRA unfused")
                except Exception as e:
                    logger.warning(f"[lora] Could not unfuse LoRA: {e}")
            
            # Disable if available (peft)
            if hasattr(pipeline, "disable_lora"):
                try:
                    pipeline.disable_lora()
                    logger.debug("[lora] LoRA disabled")
                except Exception as e:
                    logger.warning(f"[lora] Could not disable LoRA: {e}")
            
            self._current_lora = None
            logger.info("[lora] ✓ LoRA unloaded")
            return True
            
        except Exception as e:
            logger.error(f"[lora] Error unloading LoRA: {e}", exc_info=True)
            return False
    
    def set_lora_scale(self, lora_scale: float) -> bool:
        """
        Set the scale/weight of the currently loaded LoRA.
        
        Args:
            lora_scale: Scale factor (0.0 to 2.0+)
        
        Returns:
            True if set successfully, False if no LoRA loaded
        """
        if self._current_lora is None:
            logger.warning("[lora] No LoRA loaded, cannot set scale")
            return False
        
        if self._current_lora in self._loaded_loras:
            self._loaded_loras[self._current_lora]["scale"] = lora_scale
            logger.debug(f"[lora] Set scale for {self._current_lora} to {lora_scale}")
            return True
        
        return False
    
    def get_loaded_loras(self) -> Dict[str, dict]:
        """
        Get information about all cached LoRA adapters.
        
        Returns:
            Dict mapping LoRA names to their config
        """
        return dict(self._loaded_loras)
    
    def get_current_lora(self) -> Optional[str]:
        """Get the name of the currently loaded LoRA."""
        return self._current_lora
    
    def _evict_lru(self):
        """
        Evict oldest LoRA if cache exceeds max size.
        
        Uses OrderedDict to track insertion order for LRU.
        """
        from ..config import settings
        
        max_size = settings.lora_cache_size
        
        while len(self._loaded_loras) > max_size:
            # Remove oldest (first) item
            oldest_name, _ = self._loaded_loras.popitem(last=False)
            logger.info(f"[lora] LRU evicted: {oldest_name}")
    
    def clear_cache(self):
        """Clear all cached LoRA adapters."""
        self._loaded_loras.clear()
        self._current_lora = None
        logger.info("[lora] LoRA cache cleared")


def get_lora_service() -> LoRAService:
    """Get the global LoRA service singleton."""
    return LoRAService()

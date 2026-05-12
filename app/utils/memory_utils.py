"""
Memory utilities: VRAM profiling and diagnostics.

Helpers for monitoring GPU memory usage, detecting leaks, and performing
comprehensive memory cleanup to prevent VRAM fragmentation and OOM errors.
"""

import logging
import gc
import torch

logger = logging.getLogger(__name__)


def get_device_info() -> dict:
    """
    Get information about current device.
    
    Returns:
        Dict with:
        - device: "cuda" or "cpu"
        - device_name: Name of GPU (if CUDA) or "CPU"
        - cuda_available: Whether CUDA is available
    """
    from ..core import check_cuda_available, get_device
    
    device = get_device()
    
    info = {
        "device": device,
        "cuda_available": check_cuda_available(),
    }
    
    if device == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
    else:
        info["device_name"] = "CPU"
    
    return info


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory usage statistics.
    
    Returns:
        Dict with memory stats (all zeros if CUDA not available):
        - allocated_mb: Currently allocated memory
        - reserved_mb: Currently reserved memory
        - free_mb: Free memory
        - total_mb: Total GPU memory
        - utilization_percent: Percentage of total memory used
    """
    from ..core import check_cuda_available
    
    if not check_cuda_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
            "utilization_percent": 0.0,
        }
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    free = total - allocated
    utilization = (allocated / total * 100) if total > 0 else 0.0
    
    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "free_mb": round(free, 2),
        "total_mb": round(total, 2),
        "utilization_percent": round(utilization, 2),
    }


def log_memory_stats(stage: str = ""):
    """
    Log current memory statistics.
    
    Args:
        stage: Description of current stage (e.g., "after_generation")
    """
    mem = get_gpu_memory_info()
    stage_str = f"[{stage}] " if stage else ""
    
    logger.info(
        f"{stage_str}VRAM: "
        f"allocated={mem['allocated_mb']}MB / "
        f"reserved={mem['reserved_mb']}MB / "
        f"free={mem['free_mb']}MB / "
        f"total={mem['total_mb']}MB "
        f"({mem['utilization_percent']}%)"
    )


def reset_memory():
    """
    Reset/clear memory caches comprehensively.
    
    Performs full CUDA memory cleanup:
    - Python garbage collection
    - CUDA cache clearing
    - CUDA IPC collection (defragmentation)
    
    Critical for preventing VRAM fragmentation during long-running inference.
    """
    from ..core import check_cuda_available
    
    # 1. Force Python garbage collection
    gc.collect()
    logger.debug("Python garbage collected")
    
    # 2. CUDA cleanup
    if check_cuda_available():
        try:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {e}")
        
        try:
            torch.cuda.ipc_collect()
            logger.debug("CUDA IPC collected (defragmentation)")
        except Exception as e:
            # ipc_collect might not be available in all torch versions
            logger.debug(f"CUDA IPC collect not available: {e}")
    else:
        logger.debug("CUDA not available, skipping cache clear")


def comprehensive_cleanup():
    """
    Perform comprehensive memory cleanup.
    
    More aggressive than reset_memory - use when experiencing memory pressure.
    """
    logger.info("Performing comprehensive memory cleanup...")
    
    # 1. Python garbage collection (multiple passes)
    gc.collect()
    gc.collect()
    logger.debug("Python garbage collected (2 passes)")
    
    # 2. CUDA cleanup
    from ..core import check_cuda_available
    
    if check_cuda_available():
        try:
            # Clear cache multiple times
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared (2 passes)")
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {e}")
        
        try:
            torch.cuda.ipc_collect()
            torch.cuda.ipc_collect()
            logger.debug("CUDA IPC collected (2 passes, defragmentation)")
        except Exception as e:
            logger.debug(f"CUDA IPC collect not fully available: {e}")
    
    logger.info("Comprehensive cleanup completed")


def estimate_model_memory(model_params: int, dtype_bytes: int = 2) -> float:
    """
    Rough estimate of model memory usage.
    
    Args:
        model_params: Number of model parameters
        dtype_bytes: Bytes per parameter (2 for fp16, 4 for fp32)
    
    Returns:
        Estimated memory in MB
    """
    # Parameters + activations (rough estimate)
    param_memory = model_params * dtype_bytes / 1024 / 1024
    # Activations typically use 1-2x parameters
    activation_memory = param_memory * 1.5
    return param_memory + activation_memory


def warn_if_low_memory(threshold_mb: int = 2000) -> bool:
    """
    Warn if available GPU memory is below threshold.
    
    Args:
        threshold_mb: Warning threshold in MB
    
    Returns:
        True if memory is low, False otherwise
    """
    mem = get_gpu_memory_info()
    
    if mem["free_mb"] < threshold_mb:
        logger.warning(
            f"⚠ Low GPU memory: {mem['free_mb']}MB free (threshold: {threshold_mb}MB)"
        )
        return True
    
    return False

"""
Device detection and management utilities.

Provides:
- Device detection (CUDA/CPU)
- Appropriate dtype selection based on device
- CUDA availability validation
- Memory profiling helpers
"""

import logging
import torch

logger = logging.getLogger(__name__)


def check_cuda_available() -> bool:
    """
    Check if CUDA is available and working.
    
    Returns:
        True if CUDA is available and functional, False otherwise.
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA not available (torch.cuda.is_available() == False)")
        return False
    
    try:
        # Test CUDA availability
        _ = torch.tensor([1.0]).to("cuda")
        logger.debug(f"✓ CUDA functional. GPU: {torch.cuda.get_device_name(0)}")
        return True
    except RuntimeError as e:
        logger.warning(f"⚠ CUDA available but not functional: {e}. Falling back to CPU.")
        return False


def get_device() -> str:
    """
    Determine the best device for computation: CUDA (GPU) or CPU.
    
    Returns:
        "cuda" if CUDA is available and working, else "cpu"
    """
    if check_cuda_available():
        return "cuda"
    else:
        logger.info("ℹ Using CPU (inference will be slower)")
        return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """
    Select appropriate dtype based on device.
    
    - GPU (CUDA): float16 for memory efficiency and speed
    - CPU: float32 (float16 can be unstable on CPU)
    
    Args:
        device: "cuda" or "cpu"
    
    Returns:
        torch.float16 if device is cuda, else torch.float32
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.debug(f"Using dtype: {dtype} for device: {device}")
    return dtype


def get_cuda_memory_info() -> dict:
    """
    Get current CUDA memory usage statistics.
    
    Returns:
        Dict with:
        - allocated_mb: Currently allocated memory
        - reserved_mb: Currently reserved memory
        - free_mb: Free memory
        - total_mb: Total memory
        
        Returns all zeros if CUDA is not available.
    """
    if not check_cuda_available():
        return {
            "allocated_mb": 0,
            "reserved_mb": 0,
            "free_mb": 0,
            "total_mb": 0,
        }
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    free = total - allocated
    
    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "free_mb": round(free, 2),
        "total_mb": round(total, 2),
    }


def log_device_info():
    """Log current device configuration and memory info."""
    device = get_device()
    dtype = get_torch_dtype(device)
    logger.info(f"Device: {device} | Dtype: {dtype}")
    
    if device == "cuda":
        mem = get_cuda_memory_info()
        logger.info(
            f"VRAM: allocated={mem['allocated_mb']}MB, "
            f"reserved={mem['reserved_mb']}MB, "
            f"free={mem['free_mb']}MB, "
            f"total={mem['total_mb']}MB"
        )


def clear_cuda_cache():
    """Clear CUDA cache if available."""
    if check_cuda_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")

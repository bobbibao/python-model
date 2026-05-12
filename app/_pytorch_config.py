"""
🔥 CRITICAL: PyTorch CUDA memory configuration.

⚠️ THIS MODULE MUST BE IMPORTED BEFORE ANY OTHER TORCH IMPORTS!

Sets up optimal CUDA memory allocation strategies for inference on Colab T4/L4 GPUs:
- expandable_segments: Allows CUDA to reuse fragmented memory
- roundSmallAllocationsUp: Reduces fragmentation
- max_split_size_mb: Prevents OOM from fragmentation

This prevents CUDA OOM and VRAM fragmentation issues during long-running inference.
Must be configured BEFORE importing torch or diffusers!
"""

import os
import logging

logger = logging.getLogger(__name__)

# ============================================
# CRITICAL: Set BEFORE torch import
# ============================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "roundSmallAllocationsUp:True,"
    "max_split_size_mb:512"
)

logger.info(
    "🔧 PYTORCH_CUDA_ALLOC_CONF configured:\n"
    "   - expandable_segments: True (reuse fragmented VRAM)\n"
    "   - roundSmallAllocationsUp: True (reduce fragmentation)\n"
    "   - max_split_size_mb: 512 (split large allocations)"
)

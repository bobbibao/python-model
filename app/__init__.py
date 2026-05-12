"""
Vizera Python Model - SDXL Image Generation API.

CRITICAL: _pytorch_config must be imported FIRST, before any torch imports!
"""

# ⚠️ MUST BE FIRST: Configure CUDA memory allocation BEFORE torch import
from . import _pytorch_config  # noqa: F401

from .main import app

__all__ = ["app"]


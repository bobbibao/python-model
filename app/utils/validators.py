"""
Validation utilities for inputs.

Centralized validation to ensure consistency across API endpoints.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def validate_prompt(
    prompt: str,
    min_length: int = 1,
    max_length: int = 4000,
) -> Tuple[bool, str]:
    """
    Validate prompt string.
    
    Args:
        prompt: Prompt text
        min_length: Minimum length
        max_length: Maximum length
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not isinstance(prompt, str):
        return False, "Prompt must be a string"
    
    prompt = prompt.strip()
    
    if len(prompt) < min_length:
        return False, f"Prompt is too short (min {min_length} characters)"
    
    if len(prompt) > max_length:
        return False, f"Prompt is too long (max {max_length} characters)"
    
    return True, "Valid"


def validate_dimensions(
    width: int,
    height: int,
    min_size: int = 256,
    max_size: int = 4096,
) -> Tuple[bool, str]:
    """
    Validate image dimensions.
    
    Args:
        width: Image width
        height: Image height
        min_size: Minimum dimension
        max_size: Maximum dimension
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not isinstance(width, int) or not isinstance(height, int):
        return False, "Width and height must be integers"
    
    if width < min_size or height < min_size:
        return False, f"Dimensions too small (min {min_size})"
    
    if width > max_size or height > max_size:
        return False, f"Dimensions too large (max {max_size})"
    
    if width % 8 != 0 or height % 8 != 0:
        return False, "Width and height must be multiples of 8"
    
    return True, "Valid"


def validate_seed(seed: int) -> Tuple[bool, str]:
    """
    Validate random seed.
    
    Args:
        seed: Seed value
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if seed is None:
        return True, "Valid (None)"
    
    if not isinstance(seed, int):
        return False, "Seed must be an integer"
    
    if seed < 0 or seed > 2147483647:
        return False, "Seed must be between 0 and 2147483647"
    
    return True, "Valid"


def validate_inference_steps(steps: int) -> Tuple[bool, str]:
    """
    Validate number of inference steps.
    
    Args:
        steps: Number of steps
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not isinstance(steps, int):
        return False, "Steps must be an integer"
    
    if steps < 1 or steps > 150:
        return False, "Steps must be between 1 and 150"
    
    return True, "Valid"


def validate_guidance_scale(scale: float) -> Tuple[bool, str]:
    """
    Validate guidance scale.
    
    Args:
        scale: Guidance scale value
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not isinstance(scale, (int, float)):
        return False, "Guidance scale must be a number"
    
    if scale < 0.0 or scale > 25.0:
        return False, "Guidance scale must be between 0.0 and 25.0"
    
    return True, "Valid"


def validate_strength(strength: float) -> Tuple[bool, str]:
    """
    Validate image-to-image strength.
    
    Args:
        strength: Strength value
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not isinstance(strength, (int, float)):
        return False, "Strength must be a number"
    
    if strength < 0.0 or strength > 1.0:
        return False, "Strength must be between 0.0 and 1.0"
    
    return True, "Valid"

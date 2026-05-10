"""
Enums for type-safe routing and configuration.
"""

from enum import Enum


class ControlNetType(str, Enum):
    """Supported ControlNet types."""
    
    CANNY = "canny"
    LINEART = "lineart"
    NONE = "none"
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if value is a valid ControlNetType."""
        try:
            cls(value.lower())
            return True
        except ValueError:
            return False


class GenerationMode(str, Enum):
    """Supported generation modes."""
    
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    SKETCH_TO_IMAGE = "sketch_to_image"
    INPAINT = "inpaint"
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if value is a valid GenerationMode."""
        try:
            cls(value.lower())
            return True
        except ValueError:
            return False

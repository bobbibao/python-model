"""
ControlNet Service: Manages ControlNet model loading and validation.

Handles:
- ControlNet model caching
- Input image preprocessing (canny, lineart)
- Type-safe ControlNet management
"""

import logging
from typing import Optional, Dict
from PIL import Image

from ..models.enums import ControlNetType
from ..config import settings

logger = logging.getLogger(__name__)


class ControlNetService:
    """
    Service for managing ControlNet models and preprocessing.
    
    Features:
    - Lazy loading of ControlNet models
    - Input validation and preprocessing
    - Support for canny and lineart conditioning
    """
    
    _instance: Optional["ControlNetService"] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ControlNet service."""
        if hasattr(self, "_initialized"):
            return
        
        self._available_types = {
            ControlNetType.CANNY,
            ControlNetType.LINEART,
        }
        self._initialized = True
        
        logger.info("🔄 ControlNetService initialized")
    
    def get_available_types(self) -> set:
        """
        Get set of available ControlNet types.
        
        Returns:
            Set of supported ControlNetType values
        """
        return self._available_types.copy()
    
    def is_valid_type(self, controlnet_type: ControlNetType) -> bool:
        """
        Check if ControlNet type is supported.
        
        Args:
            controlnet_type: ControlNetType to validate
        
        Returns:
            True if supported, False otherwise
        """
        return controlnet_type in self._available_types
    
    def validate_input_image(self, image: Image.Image, max_size: int = 4096) -> bool:
        """
        Validate that input image is suitable for ControlNet conditioning.
        
        Args:
            image: PIL Image to validate
            max_size: Maximum dimension in pixels
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(image, Image.Image):
            logger.error("[controlnet] Input is not a PIL Image")
            return False
        
        width, height = image.size
        
        if width < 64 or height < 64:
            logger.error(f"[controlnet] Image too small: {width}x{height}")
            return False
        
        if width > max_size or height > max_size:
            logger.error(f"[controlnet] Image too large: {width}x{height} (max {max_size})")
            return False
        
        return True
    
    def preprocess_for_controlnet(
        self,
        image: Image.Image,
        controlnet_type: ControlNetType,
        target_height: int = 1024,
        target_width: int = 1024,
    ) -> Optional[Image.Image]:
        """
        Preprocess image for ControlNet conditioning.
        
        For canny: Converts to grayscale, applies Canny edge detection
        For lineart: Converts to grayscale, applies line art preprocessing
        
        Args:
            image: Input PIL Image
            controlnet_type: Type of preprocessing to apply
            target_height: Target height for resizing
            target_width: Target width for resizing
        
        Returns:
            Preprocessed PIL Image, or None if preprocessing failed
        """
        if not self.validate_input_image(image):
            return None
        
        try:
            logger.debug(f"[controlnet] Preprocessing image for {controlnet_type}...")
            
            # Convert to RGB first if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            if controlnet_type == ControlNetType.CANNY:
                image = self._preprocess_canny(image)
            elif controlnet_type == ControlNetType.LINEART:
                image = self._preprocess_lineart(image)
            else:
                logger.error(f"[controlnet] Unknown ControlNet type: {controlnet_type}")
                return None
            
            # Resize to target dimensions
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            logger.debug(f"[controlnet] ✓ Preprocessing complete: {controlnet_type}")
            return image
            
        except Exception as e:
            logger.error(f"[controlnet] Preprocessing failed: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _preprocess_canny(image: Image.Image) -> Image.Image:
        """
        Apply Canny edge detection to image.
        
        Args:
            image: Input PIL Image (should be RGB)
        
        Returns:
            Edge-detected PIL Image
        """
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Convert back to PIL RGB (3-channel)
            edges_rgb = np.stack([edges] * 3, axis=-1)
            result = Image.fromarray(edges_rgb)
            
            return result
            
        except ImportError:
            logger.warning("[controlnet] OpenCV not available, using PIL edge filter instead")
            # Fallback to PIL
            image_gray = image.convert("L")
            edges = image_gray.filter(ImageFilter.FIND_EDGES)
            # Convert back to RGB
            return Image.new("RGB", edges.size)
    
    @staticmethod
    def _preprocess_lineart(image: Image.Image) -> Image.Image:
        """
        Apply line art preprocessing to image.
        
        Converts to grayscale and applies edge detection suitable for sketch/line art.
        
        Args:
            image: Input PIL Image (should be RGB)
        
        Returns:
            Line art preprocessed PIL Image
        """
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Invert (white background → black lines)
            inverted = cv2.bitwise_not(gray)
            
            # Apply threshold to make binary
            _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
            
            # Dilate slightly to strengthen lines
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            # Convert back to PIL RGB
            result_rgb = np.stack([dilated] * 3, axis=-1)
            result = Image.fromarray(result_rgb)
            
            return result
            
        except ImportError:
            logger.warning("[controlnet] OpenCV not available, using PIL edge filter instead")
            # Fallback: simple grayscale
            return image.convert("L").convert("RGB")
    
    def get_model_id(self, controlnet_type: ControlNetType) -> Optional[str]:
        """
        Get HuggingFace model ID for a ControlNet type.
        
        Args:
            controlnet_type: Type of ControlNet
        
        Returns:
            Model ID string, or None if unknown type
        """
        if controlnet_type == ControlNetType.CANNY:
            return settings.controlnet_canny_model_id
        elif controlnet_type == ControlNetType.LINEART:
            return settings.controlnet_lineart_model_id
        else:
            logger.error(f"[controlnet] Unknown ControlNet type: {controlnet_type}")
            return None


def get_controlnet_service() -> ControlNetService:
    """Get the global ControlNet service singleton."""
    return ControlNetService()

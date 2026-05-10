"""
Image utilities: base64 encoding/decoding, PIL operations, preprocessing.

Centralized image I/O to avoid duplication across services.
"""

import logging
import base64
import io
from pathlib import Path
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str) -> Optional[Image.Image]:
    """
    Decode base64-encoded image string to PIL Image.
    
    Supports both raw base64 and data: URIs.
    
    Args:
        base64_string: Base64 encoded image (e.g., from frontend)
    
    Returns:
        PIL Image object, or None if decoding failed
    """
    try:
        # Handle data: URIs (e.g., from canvas.toDataURL())
        if base64_string.startswith("data:"):
            # Format: "data:image/png;base64,..."
            base64_string = base64_string.split(",", 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        logger.debug(f"✓ Decoded base64 image: {image.size} {image.mode}")
        return image
        
    except Exception as e:
        logger.error(f"✗ Failed to decode base64 image: {e}")
        return None


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image to encode
        format: Image format (PNG, JPEG, WEBP)
    
    Returns:
        Base64 encoded string
    """
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        image_data = buffer.getvalue()
        base64_string = base64.b64encode(image_data).decode("utf-8")
        
        logger.debug(f"✓ Encoded image to base64: {len(base64_string)} chars")
        return base64_string
        
    except Exception as e:
        logger.error(f"✗ Failed to encode image to base64: {e}")
        raise


def resize_image(
    image: Image.Image,
    target_size: tuple,
    preserve_aspect: bool = True,
) -> Image.Image:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input PIL Image
        target_size: Tuple of (width, height)
        preserve_aspect: If True, maintain aspect ratio (pad with background)
    
    Returns:
        Resized PIL Image
    """
    target_width, target_height = target_size
    
    if preserve_aspect:
        # Resize preserving aspect ratio, then pad
        image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        # Create white background
        result = Image.new("RGB", (target_width, target_height), "white")
        # Center the resized image
        x = (target_width - image.width) // 2
        y = (target_height - image.height) // 2
        result.paste(image, (x, y))
        return result
    else:
        # Direct resize (may distort)
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def preprocess_canny(
    image: Image.Image,
    threshold1: int = 100,
    threshold2: int = 200,
) -> Optional[Image.Image]:
    """
    Apply Canny edge detection to image.
    
    Args:
        image: Input PIL Image
        threshold1: Lower Canny threshold
        threshold2: Upper Canny threshold
    
    Returns:
        Edge-detected PIL Image (grayscale as RGB for consistency)
    """
    try:
        import cv2
        import numpy as np
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Convert to RGB (3-channel)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        result = Image.fromarray(edges_rgb)
        
        logger.debug(f"✓ Applied Canny edge detection: {result.size}")
        return result
        
    except ImportError:
        logger.warning("OpenCV not available, using PIL filter instead")
        return preprocess_canny_pil(image)
    except Exception as e:
        logger.error(f"✗ Canny preprocessing failed: {e}")
        return None


def preprocess_canny_pil(image: Image.Image) -> Image.Image:
    """
    Fallback Canny preprocessing using PIL (less accurate than OpenCV).
    
    Args:
        image: Input PIL Image
    
    Returns:
        Edge-detected PIL Image
    """
    from PIL import ImageFilter
    
    try:
        # Convert to grayscale
        gray = image.convert("L")
        # Edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        # Convert to RGB
        result = Image.new("RGB", edges.size)
        result.paste(edges)
        return result
    except Exception as e:
        logger.error(f"✗ PIL Canny fallback failed: {e}")
        return image


def preprocess_lineart(
    image: Image.Image,
    invert: bool = True,
) -> Optional[Image.Image]:
    """
    Apply line art / sketch preprocessing to image.
    
    Converts to binary line drawing suitable for sketch-guided generation.
    
    Args:
        image: Input PIL Image
        invert: If True, invert colors (white background, black lines)
    
    Returns:
        Line art PIL Image
    """
    try:
        import cv2
        import numpy as np
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        if invert:
            gray = cv2.bitwise_not(gray)
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Dilate to strengthen lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Convert to RGB
        result_rgb = np.stack([dilated] * 3, axis=-1)
        result = Image.fromarray(result_rgb)
        
        logger.debug(f"✓ Applied line art preprocessing: {result.size}")
        return result
        
    except ImportError:
        logger.warning("OpenCV not available, using PIL filter instead")
        return preprocess_lineart_pil(image, invert)
    except Exception as e:
        logger.error(f"✗ Line art preprocessing failed: {e}")
        return None


def preprocess_lineart_pil(image: Image.Image, invert: bool = True) -> Image.Image:
    """
    Fallback line art preprocessing using PIL.
    
    Args:
        image: Input PIL Image
        invert: If True, invert colors
    
    Returns:
        Line art PIL Image
    """
    from PIL import ImageFilter, ImageOps
    
    try:
        # Convert to grayscale
        gray = image.convert("L")
        
        if invert:
            gray = ImageOps.invert(gray)
        
        # Edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to RGB
        result = Image.new("RGB", edges.size)
        result.paste(edges)
        return result
    except Exception as e:
        logger.error(f"✗ PIL line art fallback failed: {e}")
        return image


def validate_image_dimensions(
    image: Image.Image,
    min_size: int = 64,
    max_size: int = 4096,
) -> bool:
    """
    Validate image dimensions.
    
    Args:
        image: PIL Image to validate
        min_size: Minimum dimension
        max_size: Maximum dimension
    
    Returns:
        True if dimensions are valid
    """
    width, height = image.size
    
    if width < min_size or height < min_size:
        logger.error(f"Image too small: {width}x{height} (min {min_size})")
        return False
    
    if width > max_size or height > max_size:
        logger.error(f"Image too large: {width}x{height} (max {max_size})")
        return False
    
    return True


def save_generated_image(
    image: Image.Image,
    output_dir: Path,
    job_id: str,
    format: str = "PNG",
) -> Path:
    """
    Save generated image to disk.
    
    Args:
        image: PIL Image to save
        output_dir: Directory to save to
        job_id: Unique job identifier
        format: Image format (PNG, JPEG, WEBP)
    
    Returns:
        Path to saved file
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{job_id}.png"
        image.save(output_path, format=format)
        
        logger.debug(f"✓ Saved image: {output_path} ({image.size})")
        return output_path
        
    except Exception as e:
        logger.error(f"✗ Failed to save image: {e}")
        raise


def load_image_from_file(file_path: str) -> Optional[Image.Image]:
    """
    Load image from file path.
    
    Args:
        file_path: Path to image file
    
    Returns:
        PIL Image, or None if loading failed
    """
    try:
        image = Image.open(file_path)
        logger.debug(f"✓ Loaded image: {file_path} ({image.size})")
        return image
    except Exception as e:
        logger.error(f"✗ Failed to load image from {file_path}: {e}")
        return None

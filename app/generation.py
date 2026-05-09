"""
Image generation and processing using Stable Diffusion
Uses centralized pipeline management for efficient resource usage
"""

import base64
import io
import logging
import urllib.request
import uuid
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, ImageDraw

from .config import settings
from .pipeline import (
    get_inpaint_pipeline,
    get_image_to_image_pipeline,
    get_text_to_image_pipeline,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Get current device (cuda or cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_output_dir() -> Path:
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_output_url(job_id: str) -> str:
    return f"{settings.base_url}/outputs/{job_id}.png"


def process_generate(
    prompt: str,
    width: int,
    height: int,
    seed: Optional[int] = None,
    input_type: str = "text-to-image",
    image: Optional[str] = None,
    strength: Optional[float] = None,
    upscale_factor: Optional[int] = None,
) -> Tuple[str, int, int]:
    """Generate images using Stable Diffusion based on input type"""
    mode = (input_type or "text-to-image").strip().lower()

    try:
        logger.info(f"Processing generate request: mode={mode}, size={width}x{height}")
        
        if mode in {"upscale", "image-upscaling"} and image:
            src = _load_image(image).convert("RGB")
            factor = max(2, min(4, upscale_factor or 2))
            logger.info(f"Upscaling image by {factor}x")
            result = src.resize(
                (src.width * factor, src.height * factor),
                Image.Resampling.LANCZOS,
            )
        elif mode in {"image-to-image", "reference", "3d-model"} and image:
            src = _load_image(image).convert("RGB")
            src = _fit_image(src, (width, height))
            strength_val = max(0.2, min(0.9, strength or 0.45))
            result = _image_to_image(src, prompt, seed, strength_val)
        elif mode in {"line-drawing-to-image", "line-drawing"} and image:
            src = _load_image(image).convert("RGB")
            src = _fit_image(src, (width, height))
            result = _line_drawing_to_image(src, prompt, seed)
        else:
            # Text-to-image mode
            result = _text_to_image(prompt, width, height, seed)

        return _save_image(result)
    except Exception as e:
        logger.error(f"Error in process_generate: {e}", exc_info=True)
        raise


def process_edit(
    method: str,
    image: str,
    prompt: Optional[str] = None,
    style_prompt: Optional[str] = None,
    mask: Optional[str] = None,
    crop: Optional[str] = None,
    direction: Optional[str] = None,
    pixels: Optional[int] = None,
    upscale_factor: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[str, int, int]:
    """Edit images using various methods"""
    normalized_method = (method or "").strip().upper()
    source = _load_image(image).convert("RGB")
    text = style_prompt or prompt or "Edited image"

    try:
        logger.info(f"Processing edit request: method={normalized_method}")
        
        if normalized_method == "EDIT_OBJECT_REMOVAL":
            mask_img = _resolve_mask(mask, source.size, seed)
            result = _remove_object(source, mask_img, text, seed)
        elif normalized_method == "EDIT_UPSCALING":
            factor = max(2, min(4, upscale_factor or 2))
            logger.info(f"Upscaling image by {factor}x")
            result = source.resize(
                (source.width * factor, source.height * factor),
                Image.Resampling.LANCZOS,
            )
        elif normalized_method == "EDIT_FLUX_FILL_INPAINT":
            mask_img = _resolve_mask(mask, source.size, seed)
            result = _inpaint_with_mask(source, mask_img, text, seed)
        elif normalized_method == "EDIT_FLUX_FILL_EXTEND":
            result = _extend_canvas(
                source,
                direction=direction or "all",
                pixels=max(32, min(1536, pixels or 256)),
                prompt=text,
                seed=seed,
            )
        else:
            # Default to image-to-image
            result = _image_to_image(source, text, seed, 0.5)

        return _save_image(result)
    except Exception as e:
        logger.error(f"Error in process_edit: {e}", exc_info=True)
        raise


def _save_image(image: Image.Image) -> Tuple[str, int, int]:
    """Save generated image and return job_id, width, height"""
    out_dir = ensure_output_dir()
    job_id = str(uuid.uuid4())
    output_path = out_dir / f"{job_id}.png"
    image.save(output_path, format="PNG")
    logger.info(f"✓ Image saved: {job_id}.png ({image.width}x{image.height})")
    return job_id, image.width, image.height


def _text_to_image(prompt: str, width: int, height: int, seed: Optional[int]) -> Image.Image:
    """Generate image from text prompt using Stable Diffusion"""
    # Ensure dimensions are multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    pipe = get_text_to_image_pipeline()
    device = _get_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    logger.debug(f"Generating image: prompt='{prompt[:50]}...', size={width}x{height}, seed={seed}")
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            generator=generator,
        )
    
    return output.images[0]


def _image_to_image(image: Image.Image, prompt: str, seed: Optional[int], strength: float) -> Image.Image:
    """Apply style transfer or modifications using image-to-image"""
    pipe = get_image_to_image_pipeline()
    device = _get_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    logger.debug(f"Image-to-image: prompt='{prompt[:50]}...', strength={strength}, seed={seed}")
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=settings.guidance_scale,
            num_inference_steps=settings.num_inference_steps,
            generator=generator,
        )
    
    return output.images[0]


def _line_drawing_to_image(image: Image.Image, prompt: str, seed: Optional[int]) -> Image.Image:
    """Convert line drawing to image"""
    # Convert to edges/line drawing effect first
    edges = image.convert("L")
    
    # Use as mask for inpainting
    mask = Image.new("L", image.size, 0)
    mask.paste(edges, (0, 0))
    
    pipe = get_inpaint_pipeline()
    device = _get_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    logger.debug(f"Line drawing to image: prompt='{prompt[:50]}...', seed={seed}")
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt or "detailed artistic image",
            image=image,
            mask_image=mask,
            guidance_scale=settings.guidance_scale,
            num_inference_steps=settings.num_inference_steps,
            generator=generator,
        )
    
    return output.images[0]


def _remove_object(image: Image.Image, mask: Image.Image, prompt: str, seed: Optional[int]) -> Image.Image:
    """Remove object using inpainting"""
    pipe = get_inpaint_pipeline()
    device = _get_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    logger.debug(f"Object removal: prompt='{prompt[:50]}...', seed={seed}")
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt or "seamless background",
            image=image,
            mask_image=mask,
            guidance_scale=settings.guidance_scale,
            num_inference_steps=settings.num_inference_steps,
            generator=generator,
        )
    
    return output.images[0]


def _inpaint_with_mask(image: Image.Image, mask: Image.Image, prompt: str, seed: Optional[int]) -> Image.Image:
    """Inpaint using mask"""
    pipe = get_inpaint_pipeline()
    device = _get_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    logger.debug(f"Inpainting: prompt='{prompt[:50]}...', seed={seed}")
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=settings.guidance_scale,
            num_inference_steps=settings.num_inference_steps,
            generator=generator,
        )
    
    return output.images[0]


def _extend_canvas(
    image: Image.Image,
    direction: str,
    pixels: int,
    prompt: str,
    seed: Optional[int],
) -> Image.Image:
    """Extend canvas in specified direction"""
    normalized = (direction or "all").lower()
    left = pixels if normalized in {"left", "all"} else 0
    right = pixels if normalized in {"right", "all"} else 0
    top = pixels if normalized in {"up", "top", "all"} else 0
    bottom = pixels if normalized in {"down", "bottom", "all"} else 0

    new_width = image.width + left + right
    new_height = image.height + top + bottom
    
    # Generate new background
    background = _text_to_image(prompt, new_width, new_height, seed)
    
    # Create mask for original image area
    mask = Image.new("L", (new_width, new_height), 255)
    mask.paste(Image.new("L", image.size, 0), (left, top))
    
    # Use inpainting to blend
    background.paste(image, (left, top))
    
    return background


def _resolve_mask(mask_source: Optional[str], size: Tuple[int, int], seed: Optional[int]) -> Image.Image:
    """Load or generate mask for inpainting"""
    if mask_source:
        try:
            mask = _load_image(mask_source).convert("L")
            mask = _fit_image(mask, size)
            return mask.point(lambda px: 255 if px > 100 else 0)
        except Exception:
            pass

    # Generate default mask if not provided
    width, height = size
    mask = Image.new("L", size, 0)
    draw_mask = ImageDraw.ImageDraw(mask)
    
    # Create a circular mask in center
    margin = max(64, min(width, height) // 4)
    left = margin
    top = margin
    right = width - margin
    bottom = height - margin
    
    draw_mask.ellipse((left, top, right, bottom), fill=255)
    return mask


def _load_image(source: str) -> Image.Image:
    """Load image from various sources"""
    source = source.strip()
    
    if source.startswith("data:image/"):
        _, encoded = source.split(",", 1)
        raw = base64.b64decode(encoded)
        return Image.open(io.BytesIO(raw))

    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(source, timeout=20) as response:
            raw = response.read()
        return Image.open(io.BytesIO(raw))

    path = Path(source)
    if path.exists():
        return Image.open(path)

    # Assume raw base64 string without data URL prefix
    raw = base64.b64decode(source)
    return Image.open(io.BytesIO(raw))


def _fit_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Fit image to specified size"""
    from PIL import ImageOps
    fitted = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    return fitted

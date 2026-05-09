from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    width: int = Field(1024, ge=256, le=4096)
    height: int = Field(1024, ge=256, le=4096)
    seed: Optional[int] = None
    input_type: str = Field("text-to-image", max_length=100)
    image: Optional[str] = None
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    upscale_factor: Optional[int] = Field(None, ge=2, le=4)
    metadata: Optional[Dict[str, Any]] = None


class EditRequest(BaseModel):
    method: str = Field(..., min_length=3, max_length=128)
    prompt: Optional[str] = Field(None, max_length=4000)
    style_prompt: Optional[str] = Field(None, max_length=4000)
    image: str = Field(..., min_length=1)
    mask: Optional[str] = None
    crop: Optional[str] = None
    direction: Optional[str] = Field(None, max_length=20)
    pixels: Optional[int] = Field(None, ge=1, le=4096)
    upscale_factor: Optional[int] = Field(None, ge=2, le=4)
    seed: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    image_url: str
    width: int
    height: int

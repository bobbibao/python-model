"""
Prompt Service: Optional prompt enhancement and processing.

Lightweight service for prompt engineering:
- Validation
- Enhancement (optional)
- Combination of positive/negative prompts
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PromptService:
    """
    Service for prompt processing and enhancement.
    
    Features:
    - Prompt validation
    - Optional enhancement (stubbed out for future extensions)
    - Negative prompt handling
    """
    
    _instance: "PromptService" = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize PromptService."""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        logger.info("🔄 PromptService initialized")
    
    def validate_prompt(self, prompt: str, max_length: int = 4000) -> Tuple[bool, str]:
        """
        Validate a prompt string.
        
        Args:
            prompt: Prompt text to validate
            max_length: Maximum allowed length
        
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        if not isinstance(prompt, str):
            return False, "Prompt must be a string"
        
        if len(prompt.strip()) == 0:
            return False, "Prompt cannot be empty"
        
        if len(prompt) > max_length:
            return False, f"Prompt exceeds max length {max_length}"
        
        return True, "Valid"
    
    def enhance_prompt(self, prompt: str) -> str:
        """
        Optionally enhance a prompt with additional context.
        
        Currently a pass-through; can be extended to use:
        - LLM-based prompt enhancement
        - Template expansion
        - Style injection
        
        Args:
            prompt: Original prompt
        
        Returns:
            Enhanced prompt (currently unchanged)
        """
        # TODO: Implement actual prompt enhancement
        # - Use library like promptenhancer
        # - Or call local LLM for improvement
        # For now, return as-is
        return prompt.strip()
    
    def combine_prompts(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
    ) -> Tuple[str, str]:
        """
        Combine and normalize positive and negative prompts.
        
        Args:
            positive_prompt: Main prompt
            negative_prompt: Negative prompt (what to avoid)
        
        Returns:
            Tuple of (positive, negative) prompts
        """
        pos = (positive_prompt or "").strip()
        neg = (negative_prompt or "").strip()
        
        if not pos:
            pos = "a photo"
        
        logger.debug(f"[prompt] Combined: pos='{pos[:50]}...', neg='{neg[:50]}...'")
        return pos, neg


def get_prompt_service() -> PromptService:
    """Get the global PromptService singleton."""
    return PromptService()

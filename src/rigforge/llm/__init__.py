"""LLM 模块：模型构建、调用与配置"""

from .providers import build_llm, invoke_with_rate_limit, invoke_with_turn_timeout
from .prompts import (
    REQUIREMENT_EXTRACTION_PROMPT,
    CONVERSATIONAL_EXTRACTION_PROMPT,
    RECOMMENDATION_PROMPT,
    FALLBACK_PROMPT,
)

__all__ = [
    "build_llm",
    "invoke_with_rate_limit",
    "invoke_with_turn_timeout",
    "REQUIREMENT_EXTRACTION_PROMPT",
    "CONVERSATIONAL_EXTRACTION_PROMPT",
    "RECOMMENDATION_PROMPT",
    "FALLBACK_PROMPT",
]

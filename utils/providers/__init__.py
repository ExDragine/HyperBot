"""Provider extensions for model vendors used by HyperBot."""
from .deepseek import (
    DEEPSEEK_VALID_KWARGS,
    DeepSeekProvider,
    create_chat_model,
    official_api_mode,
)

__all__ = [
    "DEEPSEEK_VALID_KWARGS",
    "DeepSeekProvider",
    "create_chat_model",
    "official_api_mode",
]

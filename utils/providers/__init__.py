"""Provider extensions for model vendors used by HyperBot."""

from .deepseek import (
    DEEPSEEK_VALID_KWARGS,
    DeepSeekProvider,
    create_chat_model,
    official_api_mode,
)
from .deepseek_media import DeepSeekMediaError, ocr_file, synthesize_speech, transcribe_audio

__all__ = [
    "DEEPSEEK_VALID_KWARGS",
    "DeepSeekProvider",
    "DeepSeekMediaError",
    "create_chat_model",
    "official_api_mode",
    "ocr_file",
    "synthesize_speech",
    "transcribe_audio",
]

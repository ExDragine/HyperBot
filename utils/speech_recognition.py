"""DeepSeek hosted speech recognition with the historical Frontier contract."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from utils.configs import EnvConfig, SpeechRecognitionConfig
from utils.providers.deepseek.media import DeepSeekMediaError
from utils.providers.deepseek.media import transcribe_audio as _transcribe_audio


class SpeechRecognitionError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    audio_seconds: float
    elapsed_seconds: float


async def transcribe_audio(data: bytes, *, settings: SpeechRecognitionConfig | None = None) -> Transcription:
    settings = settings or EnvConfig.SPEECH_RECOGNITION
    if not settings.enabled:
        raise SpeechRecognitionError("disabled")
    if not data or len(data) > settings.max_audio_bytes:
        raise SpeechRecognitionError("audio_size_limit")
    started = time.monotonic()
    try:
        async with asyncio.timeout(settings.timeout_seconds):
            text, duration = await _transcribe_audio(
                data, provider=settings.credentials_profile, max_seconds=settings.max_audio_seconds,
            )
        if len(text) > settings.max_transcript_chars:
            raise SpeechRecognitionError("transcript_limit")
        return Transcription(text, duration, time.monotonic() - started)
    except TimeoutError as exc:
        raise SpeechRecognitionError("timeout") from exc
    except DeepSeekMediaError as exc:
        raise SpeechRecognitionError(exc.code) from exc


async def transcribe_message_audio(text: str, audio: list[bytes]) -> tuple[str, bool]:
    """Append DeepSeek ASR transcripts while retaining the original attachment."""
    settings = EnvConfig.SPEECH_RECOGNITION
    if not settings.enabled or not audio:
        return text, False
    additions: list[str] = []
    for index, data in enumerate(audio[: settings.max_audio_per_message], start=1):
        try:
            result = await transcribe_audio(data, settings=settings)
            additions.append(f"[用户语音 {index} 自动转写，可能有识别错误]\n{result.text}\n[语音转写结束]")
            from nonebot import logger

            logger.info(
                "DeepSeek 语音识别完成 audio_seconds={:.2f} elapsed_seconds={:.2f} chars={}",
                result.audio_seconds, result.elapsed_seconds, len(result.text),
            )
        except SpeechRecognitionError as exc:
            from nonebot import logger

            logger.warning("DeepSeek 语音识别失败 code={}", exc.code)
            additions.append(f"[用户语音 {index} 识别失败，错误码：{exc.code}；可请用户改发文字]")
    if len(audio) > settings.max_audio_per_message:
        additions.append("[其余语音超出本条消息的识别数量限制，尚未识别]")
    return "\n\n".join([text, *additions]).strip(), True


__all__ = ["SpeechRecognitionError", "Transcription", "transcribe_audio", "transcribe_message_audio"]

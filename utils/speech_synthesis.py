"""Hosted speech synthesis for QQ voice artifacts."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from utils.configs import EnvConfig, SpeechSynthesisConfig
from utils.providers.deepseek.media import DeepSeekMediaError
from utils.providers.deepseek.media import synthesize_speech as _synthesize_speech
from utils.providers.openai import audio as openai_audio


class SpeechSynthesisError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SynthesizedSpeech:
    wav: bytes
    silk: bytes
    audio_seconds: float
    elapsed_seconds: float


async def synthesize_speech(text: str, *, settings: SpeechSynthesisConfig | None = None) -> SynthesizedSpeech:
    settings = settings or EnvConfig.SPEECH_SYNTHESIS
    if not settings.enabled:
        raise SpeechSynthesisError("disabled")
    started = time.monotonic()
    try:
        async with asyncio.timeout(settings.timeout_seconds):
            if settings.backend == "openai":
                wav, silk, duration = await openai_audio.synthesize_speech(
                    text, provider=settings.credentials_profile, model=settings.model.strip(),
                    voice=settings.voice_id.strip(), max_text_chars=settings.max_text_chars,
                    max_seconds=settings.max_audio_seconds,
                )
            else:
                wav, silk, duration = await _synthesize_speech(
                    text, provider=settings.credentials_profile, voice_id=settings.voice_id,
                    max_text_chars=settings.max_text_chars,
                )
        if duration > settings.max_audio_seconds:
            raise SpeechSynthesisError("audio_duration_limit")
        return SynthesizedSpeech(wav, silk, duration, time.monotonic() - started)
    except TimeoutError as exc:
        raise SpeechSynthesisError("timeout") from exc
    except (DeepSeekMediaError, openai_audio.OpenAIAudioError) as exc:
        raise SpeechSynthesisError(exc.code) from exc


__all__ = ["SpeechSynthesisError", "SynthesizedSpeech", "synthesize_speech"]

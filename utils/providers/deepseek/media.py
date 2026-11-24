"""DeepSeek hosted OCR, ASR and TTS clients.

The public API deliberately stays small so the message pipeline can replace
the former local speech runtimes without changing its callers.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from utils.configs import get_provider_profile
from utils.http_client import get_http_client
from utils.speech_codec import (
    SILK_HEADERS,
    AudioCodecError,
    decode_silk,
)
from utils.speech_codec import (
    wav_duration as _audio_seconds,
)
from utils.speech_codec import (
    wav_to_silk as _wav_to_silk,
)

from .constants import DEFAULT_BASE_URL

_client = get_http_client("deepseek-media", timeout=600.0)


class DeepSeekMediaError(Exception):
    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or code)
        self.code = code


def _credentials(provider: str) -> tuple[str, str]:
    profile = get_provider_profile(provider)
    key = str(profile.get("api_key", "")).strip()
    if not key:
        raise DeepSeekMediaError("missing_api_key")
    raw = str(profile.get("base_url") or "").strip()
    if not raw:
        # Only the DeepSeek adapter defaults to the official host; any other
        # profile's key must not be sent there by omission.
        if str(profile.get("type") or "").strip().lower() != "deepseek":
            raise DeepSeekMediaError("provider_not_deepseek")
        raw = DEFAULT_BASE_URL
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DeepSeekMediaError("invalid_base_url")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DeepSeekMediaError("invalid_base_url")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")), key


def _error_code(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    return str(error.get("code") or error.get("type") or f"http_{response.status_code}") if isinstance(error, dict) else f"http_{response.status_code}"


async def _raise_for_status(response: Any) -> None:
    if getattr(response, "is_success", False):
        return
    status = getattr(response, "status_code", 0)
    raise DeepSeekMediaError(_error_code(response), f"DeepSeek media request failed: {status}")


async def _normalize_audio(data: bytes, max_seconds: int) -> tuple[bytes, str]:
    if not data:
        raise DeepSeekMediaError("empty_audio")
    with tempfile.TemporaryDirectory(prefix="frontier-deepseek-asr-") as directory:
        root = Path(directory)
        if data.startswith(SILK_HEADERS):
            wav = root / "input.wav"
            try:
                await decode_silk(data, wav, max_seconds)
            except AudioCodecError as exc:
                raise DeepSeekMediaError(exc.code) from exc
            return wav.read_bytes(), "speech.wav"
        if data.startswith(b"RIFF") or data.startswith(b"OggS"):
            return data, "speech.wav" if data.startswith(b"RIFF") else "speech.ogg"
        raise DeepSeekMediaError("unsupported_audio_format")


async def transcribe_audio(data: bytes, *, provider: str = "deepseek", max_seconds: int = 120) -> tuple[str, float]:
    base_url, key = _credentials(provider)
    payload, name = await _normalize_audio(data, max_seconds)
    started = time.monotonic()
    response = await _client.post(
        f"{base_url}/v1/tools/asr",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (name, payload, "audio/wav" if name.endswith("wav") else "audio/ogg")},
        timeout=600.0,
    )
    await _raise_for_status(response)
    try:
        result = response.json()
        text = str(result["text"]).strip()
        duration = float(result.get("duration_ms", 0)) / 1000
    except (KeyError, TypeError, ValueError) as exc:
        raise DeepSeekMediaError("invalid_transcript") from exc
    if not text:
        raise DeepSeekMediaError("no_speech")
    return text, duration or time.monotonic() - started


async def synthesize_speech(
    text: str,
    *,
    provider: str = "deepseek",
    voice_id: str = "stella",
    max_text_chars: int = 10_000,
) -> tuple[bytes, bytes, float]:
    text = text.strip()
    if not text:
        raise DeepSeekMediaError("empty_text")
    if len(text) > max_text_chars:
        raise DeepSeekMediaError("text_limit")
    base_url, key = _credentials(provider)
    response = await _client.post(
        f"{base_url}/v1/tools/tts",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"text": text, "voice_id": voice_id, "format": "wav", "filter_markdown": True},
        timeout=600.0,
    )
    await _raise_for_status(response)
    wav = bytes(response.content)
    try:
        return wav, await asyncio.to_thread(_wav_to_silk, wav), _audio_seconds(wav)
    except AudioCodecError as exc:
        raise DeepSeekMediaError(exc.code) from exc


async def ocr_file(
    data: bytes,
    file_name: str,
    *,
    provider: str = "deepseek",
    purpose: str = "text_recognition",
) -> dict[str, Any]:
    if purpose not in {"text_recognition", "rich_layout_recognition"}:
        raise DeepSeekMediaError("invalid_ocr_purpose")
    base_url, key = _credentials(provider)
    response = await _client.post(
        f"{base_url}/v1/tools/ocr",
        headers={"Authorization": f"Bearer {key}"},
        data={"purpose": purpose},
        files={"file": (Path(file_name).name or "document", data, "application/octet-stream")},
        timeout=600.0,
    )
    await _raise_for_status(response)
    try:
        result = response.json()
    except Exception as exc:
        raise DeepSeekMediaError("invalid_ocr_result") from exc
    if not isinstance(result, dict) or "pages" not in result:
        raise DeepSeekMediaError("invalid_ocr_result")
    return result


__all__ = ["DeepSeekMediaError", "ocr_file", "synthesize_speech", "transcribe_audio"]

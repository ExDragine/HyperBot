# ruff: noqa: S101

import io
import wave

import pytest

from utils import deepseek_media


class _Response:
    is_success = True
    status_code = 200
    content = b"wav"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def _wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\0\0" * 2400)
    return output.getvalue()


@pytest.mark.asyncio
async def test_asr_posts_multipart_to_deepseek(monkeypatch):
    client = _Client(_Response({"text": "你好", "duration_ms": 1200}))
    monkeypatch.setattr(deepseek_media, "_client", client)
    monkeypatch.setattr(deepseek_media, "_credentials", lambda provider: ("https://api.deepseek.com", "key"))

    text, duration = await deepseek_media.transcribe_audio(b"OggS audio", provider="deepseek")

    assert (text, duration) == ("你好", 1.2)
    url, kwargs = client.calls[0]
    assert url == ("https://api.deepseek.com/v1/tools/asr",)
    assert kwargs["headers"]["Authorization"] == "Bearer key"
    assert kwargs["files"]["file"][0] == "speech.ogg"


@pytest.mark.asyncio
async def test_tts_posts_json_and_returns_wav_and_silk(monkeypatch):
    wav = _wav()
    client = _Client(_Response({}))
    client.response.content = wav
    monkeypatch.setattr(deepseek_media, "_client", client)
    monkeypatch.setattr(deepseek_media, "_credentials", lambda provider: ("https://api.deepseek.com", "key"))
    monkeypatch.setattr(deepseek_media, "_wav_to_silk", lambda payload: b"\x02#!SILK_V3voice")

    result = await deepseek_media.synthesize_speech("你好", provider="deepseek", voice_id="tide")

    assert result[0] == wav
    assert result[1].startswith(b"\x02#!SILK_V3")
    assert result[2] == pytest.approx(0.1, abs=0.01)
    _, kwargs = client.calls[0]
    assert kwargs["json"]["voice_id"] == "tide"
    assert kwargs["json"]["format"] == "wav"


@pytest.mark.asyncio
async def test_ocr_posts_file_and_purpose(monkeypatch):
    client = _Client(_Response({"id": "ocr-1", "pages": [{"index": 0, "text": "正文"}]}))
    monkeypatch.setattr(deepseek_media, "_client", client)
    monkeypatch.setattr(deepseek_media, "_credentials", lambda provider: ("https://api.deepseek.com", "key"))

    result = await deepseek_media.ocr_file(b"pdf", "report.pdf", purpose="rich_layout_recognition")

    assert result["pages"][0]["text"] == "正文"
    _, kwargs = client.calls[0]
    assert kwargs["data"] == {"purpose": "rich_layout_recognition"}
    assert kwargs["files"]["file"][0] == "report.pdf"

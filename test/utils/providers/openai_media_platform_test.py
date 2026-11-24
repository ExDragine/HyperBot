# ruff: noqa: S101

import base64
import io
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from openai import AsyncOpenAI

from utils.configs import SpeechRecognitionConfig, SpeechSynthesisConfig
from utils.providers.openai import audio
from utils.providers.openai.platform import OpenAIPlatform
from utils.speech_codec import pcm_to_wav
from utils.speech_recognition import SpeechRecognitionError, transcribe_audio
from utils.speech_synthesis import SpeechSynthesisError, synthesize_speech


@pytest.mark.asyncio
async def test_openai_speech_recognition_checks_duration_before_upload(monkeypatch):
    wav = pcm_to_wav(b"\0\0" * 16_000, 16_000)
    calls = []

    async def handle(request):
        calls.append(request)
        body = await request.aread()
        assert request.url.path == "/v1/audio/transcriptions"
        assert b'gpt-transcribe' in body and b'filename="speech.wav"' in body
        assert wav in body
        return httpx2.Response(200, json={"text": " 语音内容 "})

    @asynccontextmanager
    async def client(profile):
        assert profile == "openai"
        async with AsyncOpenAI(
            api_key="fixture-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        ) as sdk:
            yield sdk

    monkeypatch.setattr(audio, "openai_client", client)
    settings = SpeechRecognitionConfig(enabled=True, backend="openai", max_audio_seconds=1)
    transcript = await transcribe_audio(wav, settings=settings)
    assert transcript.text == "语音内容" and transcript.audio_seconds == 1
    too_long = pcm_to_wav(b"\0\0" * 16_001, 16_000)
    with pytest.raises(SpeechRecognitionError, match="audio_duration_limit"):
        await transcribe_audio(too_long, settings=settings)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "incomplete"])
async def test_realtime_speech_requires_completed_response(monkeypatch, status):
    pcm = b"\0\0" * 2_400

    class Realtime:
        session = SimpleNamespace(update=AsyncMock())
        conversation = SimpleNamespace(item=SimpleNamespace(create=AsyncMock()))
        response = SimpleNamespace(create=AsyncMock())

        async def __aiter__(self):
            yield SimpleNamespace(type="response.output_audio.delta", delta=base64.b64encode(pcm).decode())
            yield SimpleNamespace(type="response.done", response=SimpleNamespace(status=status))

    @asynccontextmanager
    async def session(profile, *, model):
        assert (profile, model) == ("openai", "gpt-realtime-2.1-mini")
        yield Realtime()

    monkeypatch.setattr(audio, "realtime_session", session)
    monkeypatch.setattr(audio, "wav_to_silk", lambda wav: b"silk" if wav == pcm_to_wav(pcm, 24_000) else b"")
    settings = SpeechSynthesisConfig(enabled=True, backend="openai")
    if status == "completed":
        result = await synthesize_speech("你好", settings=settings)
        assert result.silk == b"silk" and result.audio_seconds == pytest.approx(0.1)
    else:
        with pytest.raises(SpeechSynthesisError, match="incomplete_audio"):
            await synthesize_speech("你好", settings=settings)
    assert Realtime.session.update.call_args.kwargs["session"]["audio"]["output"]["voice"] == "marin"
    assert Realtime.conversation.item.create.call_args.kwargs["item"]["content"][0]["text"] == "你好"
    Realtime.response.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_platform_calls_openai_endpoints_and_downloads_batch_output(tmp_path):
    requests = []
    batch_input = tmp_path / "requests.jsonl"
    batch_input.write_text('{"custom_id":"one","method":"POST","url":"/v1/responses","body":{}}\n', encoding="utf-8")
    batch_output = b'{"custom_id":"one","response":{"status_code":200}}\n'

    async def handle(request):
        content = await request.aread()
        requests.append((request.method, request.url.path, content))
        match request.url.path:
            case "/v1/conversations":
                return httpx2.Response(200, json={"id": "conv_1", "object": "conversation", "created_at": 1, "metadata": {}})
            case "/v1/responses" | "/v1/responses/resp_1" | "/v1/responses/resp_1/cancel":
                return httpx2.Response(200, json={"id": "resp_1", "object": "response", "output": []})
            case "/v1/files":
                return httpx2.Response(200, json={"id": "file_1", "object": "file"})
            case "/v1/batches" | "/v1/batches/batch_1" | "/v1/batches/batch_1/cancel":
                return httpx2.Response(200, json={"id": "batch_1", "object": "batch", "output_file_id": "file_out"})
            case "/v1/files/file_out/content":
                return httpx2.Response(200, content=batch_output)
            case "/v1/embeddings":
                return httpx2.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})
            case "/v1/moderations":
                return httpx2.Response(200, json={"results": [{"flagged": False}]})
            case _:
                pytest.fail(f"Unexpected OpenAI request {request.url.path}")

    async with AsyncOpenAI(
        api_key="fixture-key", base_url="https://api.openai.com/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
    ) as sdk:
        platform = OpenAIPlatform(sdk)
        conversation = await platform.create_conversation()
        await platform.respond("gpt-6.2-sol", "你好", conversation_id=conversation.id)
        response = await platform.start_background_response("gpt-6.2-sol", "摘要", completion_window="2h")
        await platform.retrieve_response(response.id)
        await platform.cancel_response(response.id)
        uploaded = await platform.upload_batch_file(batch_input)
        batch = await platform.create_batch(uploaded.id, "/v1/responses")
        await platform.retrieve_batch(batch.id)
        await platform.cancel_batch(batch.id)
        download = io.BytesIO()
        async for chunk in platform.batch_file_content(batch.output_file_id):
            download.write(chunk)
        embedding = await platform.embed(["测试"], dimensions=1)
        moderation = await platform.moderate([{"type": "text", "text": "测试"}])

    conversation_body = json.loads(requests[1][2])
    assert conversation_body["conversation"] == "conv_1" and conversation_body["store"] is False
    background_body = json.loads(requests[2][2])
    assert background_body["background"] is True and background_body["service_tier"] == "default"
    assert background_body["background_options"] == {"completion_window": "2h"}
    assert b'filename="requests.jsonl"' in requests[5][2] and b"batch" in requests[5][2]
    assert json.loads(requests[6][2]) == {"input_file_id": "file_1", "endpoint": "/v1/responses", "completion_window": "24h"}
    assert download.getvalue() == batch_output
    assert embedding.data[0].embedding == [0.1] and not moderation.results[0].flagged

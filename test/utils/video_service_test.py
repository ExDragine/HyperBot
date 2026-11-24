# ruff: noqa: S101

from types import SimpleNamespace

import httpx2
import pytest

from utils import video_service
from utils.providers.openai import videos


def _set_video_config(monkeypatch) -> None:
    monkeypatch.setattr(videos, "DefaultAsyncHttpxClient", lambda **kwargs: kwargs)
    monkeypatch.setattr(videos.EnvConfig, "VIDEO_MODEL", "sora-test", raising=False)
    monkeypatch.setattr(videos.EnvConfig, "VIDEO_MODEL_PROVIDER", "media", raising=False)
    monkeypatch.setattr(videos.EnvConfig, "VIDEO_SIZE", "1280x720", raising=False)
    monkeypatch.setattr(videos.EnvConfig, "VIDEO_SECONDS", "8", raising=False)
    monkeypatch.setattr(videos.EnvConfig, "VIDEO_POLL_INTERVAL_SECONDS", 0, raising=False)
    monkeypatch.setattr(videos.EnvConfig, "VIDEO_POLL_TIMEOUT_SECONDS", 30, raising=False)
    monkeypatch.setattr(
        videos,
        "get_provider_profile",
        lambda name: {
            "type": "openai",
            "base_url": "https://media.example.com/v1",
            "api_key": "sk-video",
        }
        if name == "media"
        else None,
    )


class DummyBinaryResponse:
    def __init__(self, data: bytes):
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def iter_bytes(self, *, chunk_size):
        yield self.data


class StreamingVideos:
    @property
    def with_streaming_response(self):
        return self


@pytest.mark.asyncio
async def test_generate_video_creates_polls_and_downloads(monkeypatch):
    calls = {}

    class DummyVideos(StreamingVideos):
        async def create(self, **kwargs):
            calls["create"] = kwargs
            return SimpleNamespace(id="video_123", status="queued")

        async def retrieve(self, video_id):
            calls["retrieve"] = video_id
            return SimpleNamespace(id=video_id, status="completed")

        def download_content(self, video_id):
            calls["download"] = video_id
            return DummyBinaryResponse(b"generated-video")

    class DummyClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.videos = DummyVideos()

        async def close(self):
            calls["closed"] = True

    _set_video_config(monkeypatch)
    monkeypatch.setattr(videos, "AsyncOpenAI", DummyClient)

    result = await video_service.generate_video("a happy horse")

    assert result == video_service.VideoGenerationResult(raw=b"generated-video")
    assert calls["client"] == {
        "api_key": "sk-video", "base_url": "https://media.example.com/v1", "http_client": {"follow_redirects": False},
    }
    assert calls["create"] == {
        "prompt": "a happy horse",
        "model": "sora-test",
        "size": "1280x720",
        "seconds": "8",
    }
    assert calls["retrieve"] == "video_123"
    assert calls["download"] == "video_123"
    assert calls["closed"] is True


@pytest.mark.asyncio
async def test_generate_video_uses_image_reference(monkeypatch):
    calls = {}

    class DummyVideos(StreamingVideos):
        async def create(self, **kwargs):
            calls["create"] = kwargs
            return SimpleNamespace(id="video_image", status="completed")

        def download_content(self, _video_id):
            return DummyBinaryResponse(b"image-video")

    class DummyClient:
        def __init__(self, **_kwargs):
            self.videos = DummyVideos()

        async def close(self):
            return None

    _set_video_config(monkeypatch)
    monkeypatch.setattr(videos, "AsyncOpenAI", DummyClient)

    result = await video_service.generate_video(
        "animate this",
        image=video_service.MediaReference(data=b"image-bytes", mime_type="image/png"),
    )

    assert result == video_service.VideoGenerationResult(raw=b"image-video")
    reference = calls["create"]["input_reference"]
    assert reference == ("reference.png", b"image-bytes", "image/png")


@pytest.mark.asyncio
async def test_generate_video_uses_edit_endpoint_for_video_reference(monkeypatch):
    calls = {}

    class DummyVideos(StreamingVideos):
        async def edit(self, **kwargs):
            calls["edit"] = kwargs
            return SimpleNamespace(id="video_edit", status="completed")

        def download_content(self, _video_id):
            return DummyBinaryResponse(b"edited-video")

    class DummyClient:
        def __init__(self, **_kwargs):
            self.videos = DummyVideos()

        async def close(self):
            return None

    _set_video_config(monkeypatch)
    monkeypatch.setattr(videos, "AsyncOpenAI", DummyClient)

    result = await video_service.generate_video(
        "extend this",
        video=video_service.MediaReference(data=b"video-bytes", mime_type="video/mp4"),
    )

    assert result == video_service.VideoGenerationResult(raw=b"edited-video")
    assert calls["edit"] == {
        "prompt": "extend this",
        "video": ("reference.mp4", b"video-bytes", "video/mp4"),
    }


@pytest.mark.asyncio
async def test_generate_video_returns_none_for_failed_job(monkeypatch):
    class DummyVideos:
        async def create(self, **_kwargs):
            return SimpleNamespace(id="video_failed", status="failed", error="blocked")

    class DummyClient:
        def __init__(self, **_kwargs):
            self.videos = DummyVideos()

        async def close(self):
            return None

    _set_video_config(monkeypatch)
    monkeypatch.setattr(videos, "AsyncOpenAI", DummyClient)

    assert await video_service.generate_video("a failed horse") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("downloadable", [True, False])
async def test_generate_video_falls_back_to_safe_result_download(monkeypatch, downloadable):
    class DummyVideos(StreamingVideos):
        async def create(self, **_kwargs):
            return SimpleNamespace(id="video_url", status="completed", url="https://cdn.example.com/video.mp4")

        def download_content(self, _video_id):
            raise RuntimeError("content endpoint unavailable")

    class DummyClient:
        def __init__(self, **_kwargs):
            self.videos = DummyVideos()

        async def close(self):
            return None

    class DummyFetcher:
        def __init__(self, *_args, **_kwargs):
            pass

        async def get(self, url, *, limit):
            assert url == "https://cdn.example.com/video.mp4"
            assert limit == videos.MAX_BYTES
            if not downloadable:
                raise ValueError("download rejected")
            return b"video", "video/mp4"

    _set_video_config(monkeypatch)
    monkeypatch.setattr(videos, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(videos, "PublicFetcher", DummyFetcher)

    result = await video_service.generate_video("a video")
    assert result == (video_service.VideoGenerationResult(raw=b"video") if downloadable else None)


@pytest.mark.asyncio
async def test_official_video_endpoint_does_not_construct_a_client(monkeypatch):
    _set_video_config(monkeypatch)
    monkeypatch.setattr(videos, "get_provider_profile", lambda _name: {
        "type": "openai", "base_url": "https://api.openai.com/v1", "api_key": "fixture-key",
    })

    def unexpected(**_kwargs):
        pytest.fail("official video request attempted")

    monkeypatch.setattr(videos, "AsyncOpenAI", unexpected)
    assert await video_service.generate_video("a video") is None


@pytest.mark.asyncio
async def test_video_gateway_sdk_streams_content_without_following_redirects(monkeypatch):
    _set_video_config(monkeypatch)
    paths = []

    def handle(request):
        assert request.url.host == "media.example.com"
        paths.append(request.url.path)
        if request.method == "POST":
            return httpx2.Response(200, json={"id": "video_test", "status": "completed"})
        return httpx2.Response(302, headers={"Location": "https://downloads.example.com/result.mp4"})

    class VerifiedFetcher:
        def __init__(self, *_args, **_kwargs):
            pass

        async def get(self, url, *, limit):
            assert url == "https://downloads.example.com/result.mp4"
            assert limit == videos.MAX_BYTES
            return b"video", "video/mp4"

    monkeypatch.setattr(videos, "DefaultAsyncHttpxClient", lambda **kwargs: httpx2.AsyncClient(
        transport=httpx2.MockTransport(handle), **kwargs,
    ))
    monkeypatch.setattr(videos, "PublicFetcher", VerifiedFetcher)
    assert await video_service.generate_video("a video") == video_service.VideoGenerationResult(raw=b"video")
    assert paths == ["/v1/videos", "/v1/videos/video_test/content"]


@pytest.mark.asyncio
async def test_generate_video_returns_none_when_model_empty(monkeypatch):
    monkeypatch.setattr(videos.EnvConfig, "VIDEO_MODEL", "", raising=False)

    assert await video_service.generate_video("a video") is None

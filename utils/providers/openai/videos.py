"""Video generation through the OpenAI Videos API."""

import asyncio
import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urljoin, urlsplit

from nonebot import logger
from openai import APIStatusError, AsyncOpenAI, DefaultAsyncHttpxClient
from openai.types.video_create_params import InputReference, VideoCreateParams
from openai.types.video_seconds import VideoSeconds
from openai.types.video_size import VideoSize

from utils.configs import EnvConfig, get_provider_profile
from utils.http_client import get_http_client
from utils.markdown_media import MAX_BYTES, PublicFetcher, failure_reason
from utils.providers.openai import OPENAI_API_HOST, resolve_openai_base_url

httpx_client = get_http_client("video_service")


@dataclass(frozen=True)
class VideoGenerationResult:
    raw: bytes | None = None
    url: str | None = None


@dataclass(frozen=True)
class MediaReference:
    data: bytes
    mime_type: str


def _guess_image_mime_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _guess_video_mime_type(data: bytes) -> str:
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "video/x-msvideo"
    return "video/mp4"


def _coerce_media_reference(value: MediaReference | bytes | None, *, default_mime_type: str) -> MediaReference | None:
    if value is None:
        return None
    if isinstance(value, MediaReference):
        return value
    return MediaReference(data=value, mime_type=default_mime_type)


def _file_tuple(reference: MediaReference, *, stem: str) -> tuple[str, bytes, str]:
    extension = {
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "video/webm": "webm",
        "video/x-msvideo": "avi",
    }.get(reference.mime_type, "mp4" if reference.mime_type.startswith("video/") else "jpg")
    return f"{stem}.{extension}", reference.data, reference.mime_type


def _build_openai_client() -> AsyncOpenAI:
    profile = get_provider_profile(EnvConfig.VIDEO_MODEL_PROVIDER)
    if str(profile.get("type", "")).strip().lower() != "openai":
        raise ValueError("video_model_provider 必须引用 type = 'openai' 的 provider")
    base_url = resolve_openai_base_url(profile.get("base_url"))
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("视频 provider 必须配置兼容网关的 HTTP(S) base_url")
    if endpoint.hostname.rstrip(".").lower() == OPENAI_API_HOST:
        raise ValueError("OpenAI 官方视频接口已停用，视频生成需要兼容网关")
    return AsyncOpenAI(
        api_key=str(profile.get("api_key", "")), base_url=base_url,
        http_client=DefaultAsyncHttpxClient(follow_redirects=False),
    )


async def _create_video_job(
    client: AsyncOpenAI,
    prompt: str,
    *,
    image: MediaReference | bytes | None,
    video: MediaReference | bytes | None,
):
    video_ref = _coerce_media_reference(
        video,
        default_mime_type=_guess_video_mime_type(video) if isinstance(video, bytes) else "video/mp4",
    )
    if video_ref is not None:
        return await client.videos.edit(prompt=prompt, video=_file_tuple(video_ref, stem="reference"))

    image_ref = _coerce_media_reference(
        image,
        default_mime_type=_guess_image_mime_type(image) if isinstance(image, bytes) else "image/jpeg",
    )
    options = VideoCreateParams(prompt=prompt, model=EnvConfig.VIDEO_MODEL)
    if image_ref is not None:
        options["input_reference"] = cast(InputReference, _file_tuple(image_ref, stem="reference"))
    size = str(EnvConfig.VIDEO_SIZE).strip()
    if size:
        options["size"] = cast(VideoSize, size)
    seconds = str(EnvConfig.VIDEO_SECONDS).strip()
    if seconds:
        options["seconds"] = cast(VideoSeconds, seconds)
    return await client.videos.create(**options)


async def _poll_video_job(client: AsyncOpenAI, job):
    timeout_seconds = EnvConfig.VIDEO_POLL_TIMEOUT_SECONDS
    try:
        async with asyncio.timeout(timeout_seconds if timeout_seconds > 0 else None):
            while getattr(job, "status", None) in {"queued", "in_progress"}:
                await asyncio.sleep(max(0, EnvConfig.VIDEO_POLL_INTERVAL_SECONDS))
                job = await client.videos.retrieve(job.id)
    except TimeoutError:
        logger.warning("视频网关轮询超时: timeout={}s", timeout_seconds)
        return None

    if getattr(job, "status", None) != "completed":
        error = getattr(job, "error", None) or getattr(job, "last_error", None)
        logger.warning(f"OpenAI 视频生成失败: status={getattr(job, 'status', None)}, error={error}")
        return None
    return job


async def _download_video_content(client: AsyncOpenAI, job_id: str) -> bytes:
    data = bytearray()
    async with client.videos.with_streaming_response.download_content(job_id) as response:
        async for chunk in response.iter_bytes(chunk_size=65536):
            if len(data) + len(chunk) > MAX_BYTES:
                raise ValueError("视频超过大小限制")
            data.extend(chunk)
    return bytes(data)


def _inline_video(encoded: object) -> bytes | None:
    if not isinstance(encoded, str) or not 0 < len(encoded) <= ((MAX_BYTES + 2) // 3) * 4:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        logger.warning("视频网关返回了无效的 base64 内容")
        return None
    return raw if 0 < len(raw) <= MAX_BYTES else None


async def _video_result(client: AsyncOpenAI, job) -> VideoGenerationResult | None:
    redirect = None
    try:
        if raw := await _download_video_content(client, job.id):
            return VideoGenerationResult(raw=raw)
    except APIStatusError as exc:
        if exc.response.is_redirect and (location := exc.response.headers.get("location")):
            redirect = urljoin(str(exc.request.url), location)
        else:
            logger.warning("视频网关内容接口返回 HTTP {}", exc.status_code)
    except Exception as exc:
        logger.warning("读取视频网关内容失败: {}", type(exc).__name__)

    if raw := _inline_video(getattr(job, "b64_json", None)):
        return VideoGenerationResult(raw=raw)

    for url in (redirect, getattr(job, "url", None), getattr(job, "download_url", None)):
        if isinstance(url, str) and url:
            try:
                fetcher = PublicFetcher(
                    httpx_client, Path("cache/video"), fake_ip_fallback=EnvConfig.MESSAGE_MEDIA_FAKE_IP_FALLBACK,
                )
                raw, _mime = await fetcher.get(url, limit=MAX_BYTES)
                if raw:
                    return VideoGenerationResult(raw=raw)
            except Exception as exc:
                logger.warning("下载视频来源失败: {}", failure_reason(exc))

    logger.warning("OpenAI 视频 API 响应缺少可发送的视频内容")
    return None


async def generate_video(
    prompt: str,
    *,
    image: MediaReference | bytes | None = None,
    video: MediaReference | bytes | None = None,
) -> VideoGenerationResult | None:
    if not EnvConfig.VIDEO_MODEL:
        logger.warning("VIDEO_MODEL 未配置，视频请求失败")
        return None

    logger.info("调用兼容视频网关, model={}, prompt_length={}", EnvConfig.VIDEO_MODEL, len(prompt))
    client: AsyncOpenAI | None = None
    try:
        client = _build_openai_client()
        job = await _create_video_job(client, prompt, image=image, video=video)
        if completed := await _poll_video_job(client, job):
            return await _video_result(client, completed)
        return None
    except ValueError as exc:
        logger.warning("视频网关配置或结果无效: {}", exc)
        return None
    except Exception as exc:
        logger.warning("调用兼容视频网关失败: {}", type(exc).__name__)
        return None
    finally:
        if client is not None:
            await client.close()

"""Image generation through the OpenAI Images and Responses protocols."""

import base64
import binascii
import io
from pathlib import Path
from typing import Any, Literal, cast
from xml.etree import ElementTree

from nonebot import logger
from openai import APIError, AsyncOpenAI
from openai.types.image_edit_params import ImageEditParamsNonStreaming
from openai.types.image_generate_params import ImageGenerateParamsNonStreaming
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_message_content_list_param import ResponseInputContentParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.tool_choice_types_param import ToolChoiceTypesParam
from openai.types.responses.tool_param import ImageGeneration
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.configs import EnvConfig, get_provider_profile
from utils.http_client import get_http_client
from utils.markdown_media import PublicFetcher, failure_reason
from utils.providers.openai import is_official_openai_base, resolve_openai_base_url

_MAX_IMAGE_BYTES = 20_000_000
_IMAGE_26_MODELS = ("gpt-image-2.6-sunburst", "gpt-image-2.6-flare")
_MARKUP_MODELS = (*_IMAGE_26_MODELS, "gpt-image-2.5-sunburst", "gpt-image-2.5-flare")


class ImageOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output_format: Literal["png", "jpeg", "webp", "gif", "svg"] = "png"
    complexity: Literal["low", "medium", "high"] | None = None
    frame_count: int | None = Field(default=None, ge=2, le=16)
    frame_duration_ms: int | None = Field(default=None, ge=20, le=2000)
    use_markup: bool = False

    @model_validator(mode="after")
    def validate_animation_fields(self):
        if self.output_format != "gif" and (self.frame_count is not None or self.frame_duration_ms is not None):
            raise ValueError("帧数和单帧时长只用于 GIF")
        return self


class ImageOptionError(ValueError):
    pass


def _normalize_reference_image(image: bytes) -> bytes:
    with Image.open(io.BytesIO(image)) as raw_image:
        mode = "RGBA" if "A" in raw_image.getbands() or "transparency" in raw_image.info else "RGB"
        normalized = raw_image.convert(mode)
        with io.BytesIO() as png_bytes:
            normalized.save(png_bytes, format="PNG")
            return png_bytes.getvalue()


def _image_generation_tool(reference_images: list[bytes]) -> ImageGeneration:
    tool: ImageGeneration = {
        "type": "image_generation",
        "action": "edit" if reference_images else "generate",
        "model": EnvConfig.PAINT_MODEL,
        "output_format": "png",
        "size": "auto",
    }
    if quality := str(EnvConfig.PAINT_QUALITY).strip():
        tool["quality"] = cast(Literal["low", "medium", "high", "auto"], quality)
    return tool


def _responses_input(prompt: str, reference_images: list[bytes]) -> ResponseInputParam:
    content: list[ResponseInputContentParam] = [ResponseInputTextParam(type="input_text", text=prompt)]
    for image in reference_images:
        encoded = base64.b64encode(_normalize_reference_image(image)).decode("ascii")
        content.append(
            cast(
                ResponseInputImageParam,
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encoded}",
                },
            )
        )
    return [EasyInputMessageParam(type="message", role="user", content=content)]


def _response_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _decode_image_base64(encoded: Any) -> bytes | None:
    if not isinstance(encoded, str) or not encoded.strip():
        return None
    encoded = encoded.strip()
    if encoded.startswith("data:"):
        _, separator, encoded = encoded.partition(",")
        if not separator:
            return None
    if len(encoded) > 4 * ((_MAX_IMAGE_BYTES + 2) // 3):
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
        return data if 0 < len(data) <= _MAX_IMAGE_BYTES else None
    except (binascii.Error, ValueError):
        logger.warning("绘图 API 返回了无效的 base64 图片")
        return None


def _image_bytes_from_response(response: Any) -> bytes | None:
    output = _response_value(response, "output") or []
    for item in output:
        if _response_value(item, "type") != "image_generation_call":
            continue
        if image := _decode_image_base64(_response_value(item, "result")):
            return image

    logger.warning("Responses API 响应中没有 image_generation_call 结果")
    return None


async def _image_bytes_from_images_response(response: Any, *, output_format: str) -> bytes | None:
    for item in _response_value(response, "data") or []:
        if image := _decode_image_base64(_response_value(item, "b64_json")):
            try:
                if output_format == "png" or _validate_result_format(image, output_format):
                    return image
            except (OSError, ValueError):
                logger.warning("Images API 返回了格式无效的图片")
        url = _response_value(item, "url")
        if not isinstance(url, str) or not url:
            continue
        fetcher = PublicFetcher(
            get_http_client("paint_service"), Path("cache/paint"),
            fake_ip_fallback=EnvConfig.MESSAGE_MEDIA_FAKE_IP_FALLBACK,
        )
        try:
            image, _mime = await fetcher.get(url, limit=_MAX_IMAGE_BYTES)
            if output_format != "svg":
                with Image.open(io.BytesIO(image)) as decoded:
                    decoded.verify()
            if output_format == "png" or _validate_result_format(image, output_format):
                return image
        except Exception as exc:
            logger.warning("下载生成图片失败: {}", failure_reason(exc))
    logger.warning("Images API 响应中没有可用的图片")
    return None


def _validate_image_options(
    options: ImageOptions, *, model: str, official: bool, reference_count: int,
) -> None:
    if not official and options != ImageOptions():
        raise ImageOptionError("当前绘图网关使用默认 PNG 格式")
    if (options.complexity or options.output_format in {"svg", "gif"}) and not model.startswith(_IMAGE_26_MODELS):
        raise ImageOptionError("SVG、GIF 和复杂度设置需要 GPT-Image-2.6")
    if reference_count and options.output_format == "svg":
        raise ImageOptionError("SVG 只支持生成，请选择生成模式或改用 PNG 编辑")
    if options.use_markup and (reference_count < 2 or not model.startswith(_MARKUP_MODELS)):
        raise ImageOptionError("标注编辑需要 GPT-Image-2.5 或 2.6，并按顺序提供原图和标注图")


def _validate_result_format(image: bytes, output_format: str) -> bool:
    if output_format == "svg":
        if b"<!DOCTYPE" in image.upper():
            return False
        try:
            root = ElementTree.fromstring(image.decode("utf-8"))  # noqa: S314 - DTDs rejected above
            return root.tag == "{http://www.w3.org/2000/svg}svg"
        except (ElementTree.ParseError, UnicodeDecodeError):
            return False
    with Image.open(io.BytesIO(image)) as decoded:
        return (decoded.format or "").lower() == output_format


async def _paint_images(
    client: AsyncOpenAI, prompt: str, reference_images: list[bytes], options: ImageOptions, *, model: str,
) -> bytes | None:
    params = ImageGenerateParamsNonStreaming(
        model=model, prompt=prompt, n=1, output_format=options.output_format, stream=False, size="auto",
    )
    if quality := str(EnvConfig.PAINT_QUALITY).strip():
        params["quality"] = cast(Literal["low", "medium", "high", "auto", "xhigh", "max"], quality)
    if options.complexity is not None:
        params["complexity"] = options.complexity
    if options.frame_count is not None:
        params["frame_count"] = options.frame_count
    if options.frame_duration_ms is not None:
        params["frame_duration_ms"] = options.frame_duration_ms
    if reference_images:
        images = [
            (f"reference-{index}.png", _normalize_reference_image(image), "image/png")
            for index, image in enumerate(reference_images, start=1)
        ]
        markup = images.pop() if options.use_markup else None
        edit_params = cast(ImageEditParamsNonStreaming, {**params, "image": images})
        if markup is not None:
            edit_params["markup"] = markup
        response = await client.images.edit(**edit_params)
    else:
        response = await client.images.generate(**params)
    return await _image_bytes_from_images_response(response, output_format=options.output_format)


async def _paint_responses(client: AsyncOpenAI, prompt: str, reference_images: list[bytes]) -> bytes | None:
    params = ResponseCreateParamsNonStreaming(
        model=EnvConfig.PAINT_MODEL,
        input=_responses_input(prompt, reference_images),
        tools=[_image_generation_tool(reference_images)],
        tool_choice=ToolChoiceTypesParam(type="image_generation"),
        store=False,
        stream=False,
    )
    response = await client.responses.create(**params)
    return _image_bytes_from_response(response)


async def _paint_with_openai(prompt: str, reference_images: list[bytes], options: ImageOptions) -> bytes | None:
    profile = get_provider_profile(EnvConfig.PAINT_MODEL_PROVIDER)
    if str(profile.get("type", "")).strip().lower() != "openai":
        raise ValueError("paint_model_provider 必须引用 type = 'openai' 的 provider")
    base_url = resolve_openai_base_url(profile.get("base_url"))
    official = is_official_openai_base(base_url)
    model = EnvConfig.PAINT_MODEL
    _validate_image_options(options, model=model, official=official, reference_count=len(reference_images))
    use_responses = profile.get("api_mode") == "responses" and not official
    logger.info(
        "🎨 调用 {} 绘图 API, model={}, prompt_length={}, references={}",
        "Responses" if use_responses else "Images",
        EnvConfig.PAINT_MODEL, len(prompt), len(reference_images),
    )
    client = AsyncOpenAI(api_key=str(profile.get("api_key", "")), base_url=base_url)
    try:
        if use_responses:
            return await _paint_responses(client, prompt, reference_images)
        return await _paint_images(client, prompt, reference_images, options, model=model)
    except APIError as exc:
        logger.warning("绘图 API 调用失败: {}", type(exc).__name__)
        return None
    finally:
        await client.close()


async def paint(
    prompt: str, reference_images: list[bytes] | None = None, *, options: ImageOptions | None = None,
) -> bytes | None:
    reference_images = reference_images or []

    if not EnvConfig.PAINT_MODEL:
        logger.warning("PAINT_MODEL 未配置，绘画请求失败")
        return None

    try:
        return await _paint_with_openai(prompt, reference_images, options or ImageOptions())
    except ImageOptionError:
        raise
    except Exception as exc:
        logger.warning("调用绘图 API 失败: {}", type(exc).__name__)
        return None

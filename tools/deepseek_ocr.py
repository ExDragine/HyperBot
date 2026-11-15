"""DeepSeek hosted OCR for files already present in the current workspace."""

import json
from typing import Annotated, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field

from utils.milky_tools import configurable, resolve_virtual_path
from utils.providers.deepseek.media import DeepSeekMediaError, ocr_file

_DEFAULT_CONFIG = cast(RunnableConfig, None)


@tool(response_format="content")
async def deepseek_ocr(
    file_path: Annotated[str, Field(min_length=1, max_length=2048)],
    purpose: Annotated[str, Field(pattern="^(text_recognition|rich_layout_recognition)$")] = "text_recognition",
    config: RunnableConfig = _DEFAULT_CONFIG,
) -> str:
    """Use DeepSeek OCR on a workspace or /memory file and return page results.

    The file must already be available in the current agent virtual filesystem.
    Use rich_layout_recognition when charts or images inside a document matter.
    """
    settings = configurable(config)
    roots = settings.get("virtual_roots") or {}
    try:
        path = resolve_virtual_path(file_path, roots)
    except (OSError, ValueError) as exc:
        return json.dumps({"status": "error", "code": "invalid_file_path", "message": str(exc)}, ensure_ascii=False)
    if path is None or not path.is_file():
        return json.dumps({"status": "error", "code": "file_not_found"}, ensure_ascii=False)
    try:
        result = await ocr_file(path.read_bytes(), path.name, purpose=purpose)
    except (OSError, DeepSeekMediaError) as exc:
        return json.dumps({"status": "error", "code": getattr(exc, "code", "ocr_failed")}, ensure_ascii=False)
    return json.dumps({"status": "success", **result}, ensure_ascii=False, separators=(",", ":"))


__all__ = ["deepseek_ocr"]

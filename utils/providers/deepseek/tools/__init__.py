"""DeepSeek tools available directly to the main Agent."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def direct_tools() -> tuple[tuple[BaseTool, bool], ...]:
    """Return each tool with its read-only error policy."""
    from .balance import get_deepseek_api_balance
    from .ocr import deepseek_ocr

    return ((get_deepseek_api_balance, True), (deepseek_ocr, True))

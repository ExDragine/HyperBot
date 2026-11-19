"""Shared tool error policy for the main graph and read-only subagents."""

import logging

from langchain.agents.middleware import ToolErrorMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError

from utils.milky_errors import is_forward_build_failure, milky_error_details
from utils.milky_tools import configurable, resolve_peer

from .session_errors import CheckpointCapacityExceeded, SessionInterruptedError

logger = logging.getLogger(__name__)
BUDGET_ERRORS = (ModelCallLimitExceededError, ToolCallLimitExceededError)
CONTROL_ERRORS = (*BUDGET_ERRORS, CheckpointCapacityExceeded, SessionInterruptedError)
WEB_SEARCH_TOOL_NAMES = frozenset({
    "deepseek_web_search", "deepseek_web_fetch",
    "web_search_exa", "web_fetch_exa", "tavily_search", "tavily_extract",
    "tavily_map", "tavily_crawl", "tavily_research",
})


class ToolExecutionUncertainError(RuntimeError):
    """A potentially mutating tool failed; do not invite the model to repeat it."""


def read_only_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    if any(marker in text for marker in ("429", "rate limit", "too many requests", "usage limit", "quota")):
        return "查询服务已限流。停止继续搜索或切换后端，使用已有证据完成回复，并说明资料可能不完整。"
    if isinstance(exc, PermissionError):
        return "工具访问被拒绝。不得绕过权限或重复尝试该操作。"
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return "查询参数无效或目标不存在。请检查参数或使用其他已有证据，不要编造结果。"
    return "查询工具暂时不可用。请基于已有证据继续，并说明未能完成的查询。"


def _forward_build_error_message(request) -> str:
    message = "QQ 未能构建聊天记录卡片，本次转发未送达。不要重试转发。"
    config = getattr(request.runtime, "config", None)
    args = request.tool_call["args"]
    scene = "friend" if configurable(config).get("group_id") in (None, "", 0, "0") else "group"
    current, _ = resolve_peer(scene, None, config)
    target, _ = resolve_peer(args.get("message_scene"), args.get("peer_id"), config)
    if not current or args.get("message_scene") != scene or target != current:
        return message + "请说明转发失败，勿将正文改发到当前会话。"
    return message + (
        "如果当前用户要收取正文，可用 send_plain_text(text=正文, fold_long_text=false) 发送可复制的文字，"
        "保留顺序和必要的署名。若只接受聊天记录格式，请说明失败。只处理本条消息的请求，不补发历史任务。"
    )


def tool_error_middleware(*, read_only_tools=(), all_read_only: bool = False) -> ToolErrorMiddleware:
    names = {tool if isinstance(tool, str) else tool.name for tool in read_only_tools}

    def on_error(exc, request):
        if isinstance(exc, (*CONTROL_ERRORS, ToolExecutionUncertainError)):
            return None
        name = request.tool_call["name"]
        logger.warning(
            "Tool failed name=%s error_type=%s%s", name, type(exc).__name__,
            milky_error_details(exc, sensitive=request.tool_call.get("args")),
        )
        if name == "send_forwarded_message" and is_forward_build_failure(exc):
            return _forward_build_error_message(request)
        if all_read_only or name in names:
            return read_only_error_message(exc)
        raise ToolExecutionUncertainError(name) from None

    return ToolErrorMiddleware(on_error=on_error)

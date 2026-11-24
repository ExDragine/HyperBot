"""DeepSeek's hosted web tools, follow-up context, and per-turn fallback."""

import json
import logging

from anthropic import APIStatusError as AnthropicStatusError
from langchain_core.exceptions import ModelAPIError, ModelInvalidRequestError
from openai import UnprocessableEntityError

from utils.agents.message_envelope import AGENT_MESSAGE_SCHEMA, DIRECT_MENTION_CONTEXT_MARKER

from ..web_search import NativeWebSearchMiddleware
from .chat import DEEPSEEK_WEB_TOOLS
from .web import SEARCH_PROMPT_HINT, TOOL_NAMES

logger = logging.getLogger(__name__)


DEEPSEEK_WEB_SEARCH_PROMPT_HINT = (
    "\n\n本轮可用 DeepSeek 服务端联网，通过 `deepseek_web_search` 搜索，通过 `deepseek_read_url` 阅读链接。"
    "前文有待核实的公开事实时，用户说「找一下」「查一下」「搜一下」或「帮我确认」就是在请求搜索核实。"
    "短追问从最近对话确定完整主题，带上对象名称和待核实的细节主动搜索。"
)

_DEEPSEEK_WEB_FALLBACK_HINT = (
    "\n\n本轮 DeepSeek 服务端搜索和网页读取暂不可用。请基于现有上下文和仍可用的工具继续原任务。"
)

_LOOKUP_FOLLOWUPS = frozenset({"找一下", "查一下", "搜一下", "帮我找一下", "帮我查一下", "帮我确认", "再找一下", "再查一下"})
_WEB_REQUEST_ERRORS = (AnthropicStatusError, ModelAPIError, ModelInvalidRequestError, UnprocessableEntityError, ValueError)


def _lookup_text(text: str, bot_user_id: str | None) -> str:
    text = text.strip().removeprefix(DIRECT_MENTION_CONTEXT_MARKER).strip()
    if bot_user_id and text.startswith(mention := f"@{bot_user_id}"):
        remaining = text[len(mention):]
        if not remaining or remaining[0] in " \t\r\n：:，,":
            text = remaining
    return text.strip(" \t\r\n：:。！？?!，,")


def _lookup_topic(message_text: str, user_id: str, bot_user_id: str | None) -> str | None:
    try:
        envelope, end = json.JSONDecoder().raw_decode(message_text)
    except ValueError:
        return None
    if not isinstance(envelope, dict) or not isinstance(sender := envelope.get("sender"), dict):
        return None
    if str(sender.get("user_id")) != user_id:
        return None
    schema = envelope.get("schema")
    if schema == AGENT_MESSAGE_SCHEMA:
        text = envelope.get("content")
    elif schema == "frontier.message_context.v1":
        text = message_text[end:]
    else:
        return None
    if not isinstance(text, str):
        return None
    topic = _lookup_text(text, bot_user_id)
    if not topic or topic in _LOOKUP_FOLLOWUPS or topic.startswith("["):
        return None
    return topic[:240]


def _deepseek_web_request_failed(exc: Exception) -> bool:
    if isinstance(exc, (ModelInvalidRequestError, UnprocessableEntityError)):
        # DeepSeek also uses HTTP 400 for content review. Keep those refusals.
        return (
            getattr(exc, "code", None) not in {"content_filter", "content_policy_violation"}
            and getattr(exc, "error_code", None) not in {"content_filter", "content_policy_violation"}
            and "content exists risk" not in str(exc).lower()
        )
    if isinstance(exc, ModelAPIError):
        return True
    if isinstance(exc, AnthropicStatusError):
        return exc.status_code == 422 or (exc.status_code == 200 and exc.type in {"api_error", "overloaded_error"})
    # The SDK adapters raise ValueError for these failures inside HTTP 200.
    detail = exc.args[0] if exc.args else None
    if isinstance(detail, dict):
        return detail.get("type") == "api_error" or detail.get("code") == "server_error"
    return getattr(detail, "code", None) == "server_error"


def _messages_web_calls(blocks):
    calls = {
        block.get("id"): block for block in blocks
        if isinstance(block, dict) and block.get("type") == "server_tool_use" and block.get("name") in DEEPSEEK_WEB_TOOLS
    }
    for result in blocks:
        if not isinstance(result, dict) or result.get("type") not in {"web_search_tool_result", "web_fetch_tool_result"}:
            continue
        call = calls.get(result.get("tool_use_id"))
        if call is None or not isinstance(raw_action := call.get("input"), dict):
            continue
        action = {**raw_action, "type": raw_action.get("action")}
        record = {"kind": call.get("kind"), "action": action}
        content = result.get("content")
        entries = content if isinstance(content, list) else [content]
        entries = [item for item in entries if isinstance(item, dict)]
        if error := next((item["error_code"] for item in entries if item.get("error_code")), None):
            yield {**record, "status": "failed", "error": {"code": error}}
        elif result["type"] == "web_search_tool_result":
            action["sources"] = [
                {**item, "published_at": item.get("published_at") or item.get("page_age")}
                for item in entries if item.get("type") == "web_search_result"
            ]
            yield {**record, "status": "completed"}
        elif isinstance(content, dict) and content.get("type") == "web_fetch_result":
            page = {key: content[key] for key in ("url", "published_at") if key in content}
            if isinstance(document := content.get("content"), dict):
                page["title"] = document.get("title")
            yield {**record, "status": "completed", "page": page}


def deepseek_web_calls(message):
    """Read search and page records across the DeepSeek wire protocols."""
    content = getattr(message, "content", None)
    if isinstance(content, list):
        yield from (block for block in content if isinstance(block, dict) and block.get("type") == "web_search_call")
        yield from _messages_web_calls(content)

    fields = getattr(message, "additional_kwargs", {})
    for step in fields.get("server_tool_calls") or ():
        if not isinstance(step, dict):
            continue
        for call in step.get("calls") or ():
            if not isinstance(call, dict) or not isinstance(action := call.get("action"), dict):
                continue
            if "sources" in call:
                action = {**action, "sources": call["sources"]}
            yield {**call, "action": action}


def _deepseek_web_result_unusable(response) -> bool:
    message = response.result[-1]
    metadata = message.response_metadata
    return (
        (metadata.get("status") == "completed" or metadata.get("finish_reason") in {"stop", "end_turn"})
        and not (message.text.strip() or message.tool_calls)
        and any(
            block.get("status") == "failed"
            and isinstance(error := block.get("error"), dict) and error.get("code") == "unavailable"
            for block in deepseek_web_calls(message)
        )
    )


def _tool_name(tool):
    name = tool.get("name", "") if isinstance(tool, dict) else getattr(tool, "name", "")
    return str(name).casefold()


class DeepSeekWebSearchMiddleware(NativeWebSearchMiddleware):
    prompt_hint = DEEPSEEK_WEB_SEARCH_PROMPT_HINT

    def __init__(self, *, lookup_user_id: str | None = None, bot_user_id: str | None = None):
        super().__init__(DEEPSEEK_WEB_TOOLS)
        self.lookup_user_id = lookup_user_id
        self.bot_user_id = bot_user_id
        self._hosted_disabled = False

    @classmethod
    def for_turn(cls, *, user_text: str, user_id: str, bot_user_id: str | None):
        lookup = _lookup_text(user_text, bot_user_id) in _LOOKUP_FOLLOWUPS
        return cls(lookup_user_id=user_id if lookup else None, bot_user_id=bot_user_id)

    def _with_lookup_context(self, request):
        user_id = self.lookup_user_id
        if user_id is None:
            return request
        turns = [
            (index, message) for index, message in enumerate(request.messages)
            if message.type == "human" and message.additional_kwargs.get("lc_source") != "pinned_skill"
        ]
        if len(turns) < 2:
            return request
        topic = next((
            topic for _, message in reversed(turns[:-1])
            if (topic := _lookup_topic(message.text, user_id, self.bot_user_id))
        ), None)
        if topic is None:
            return request

        index, current = turns[-1]
        note = f"\n承接前文要核实的问题「{topic}」。本轮可联网时，请搜索这个完整主题。"
        content = current.content
        content = [*content, {"type": "text", "text": note}] if isinstance(content, list) else content + note
        current = current.model_copy(update={"content": content})
        end = index + 1
        while end < len(request.messages) and request.messages[end].additional_kwargs.get("lc_source") == "pinned_skill":
            end += 1
        return request.override(messages=[
            *request.messages[:index], *request.messages[index + 1:end], current, *request.messages[end:],
        ])

    def _local_tools(self, request):
        # The standalone tools stay registered in the graph, but the hosted search
        # and its standalone equivalent have the same name.
        return [tool for tool in request.tools if _tool_name(tool) not in TOOL_NAMES]

    def _request_without_deepseek_web(self, request):
        tools = [
            tool for tool in request.tools
            if not isinstance(tool, dict) or tool.get("type") not in DEEPSEEK_WEB_TOOLS
        ]
        hint = _DEEPSEEK_WEB_FALLBACK_HINT
        if TOOL_NAMES <= {_tool_name(tool) for tool in tools}:
            hint += SEARCH_PROMPT_HINT
        return self._with_tools_and_hint(request, tools, hint)

    def _fallback_request(self, request):
        logger.warning("DeepSeek web tools failed. Continuing the current Agent turn without them.")
        self._hosted_disabled = True
        return self._request_without_deepseek_web(request)

    def wrap_model_call(self, request, handler):
        request = self._with_lookup_context(request)
        if self._hosted_disabled:
            return handler(self._request_without_deepseek_web(request))
        primary = self._request_with_web_search(request)
        try:
            response = handler(primary)
        except _WEB_REQUEST_ERRORS as exc:
            if not _deepseek_web_request_failed(exc):
                raise
        else:
            if not _deepseek_web_result_unusable(response):
                return response
        return handler(self._fallback_request(request))

    async def awrap_model_call(self, request, handler):
        request = self._with_lookup_context(request)
        if self._hosted_disabled:
            return await handler(self._request_without_deepseek_web(request))
        primary = self._request_with_web_search(request)
        try:
            response = await handler(primary)
        except _WEB_REQUEST_ERRORS as exc:
            if not _deepseek_web_request_failed(exc):
                raise
        else:
            if not _deepseek_web_result_unusable(response):
                return response
        return await handler(self._fallback_request(request))

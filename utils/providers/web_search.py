"""Inject hosted search tool specs at the model call boundary."""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

WEB_SEARCH_PROMPT_HINT = (
    "\n\n本轮可用原生 `web_search` 查询公开网页和实时信息。"
)


class NativeWebSearchMiddleware(AgentMiddleware):
    """Inject tool dictionaries after middleware that expects local tool objects."""

    prompt_hint = WEB_SEARCH_PROMPT_HINT

    def __init__(self, tool_types: tuple[str, ...] = ("web_search",)):
        self.tool_types = tool_types

    @staticmethod
    def _with_tools_and_hint(request, tools, hint):
        system = request.system_message or SystemMessage(content="")
        content = [*system.content_blocks, {"type": "text", "text": hint}]
        return request.override(tools=tools, system_message=system.model_copy(update={"content": content}))

    def _local_tools(self, request):
        return request.tools

    def _request_with_web_search(self, request):
        tools = self._local_tools(request)
        existing = {tool.get("type") for tool in tools if isinstance(tool, dict)}
        missing = [{"type": tool_type} for tool_type in self.tool_types if tool_type not in existing]
        return self._with_tools_and_hint(request, [*tools, *missing], self.prompt_hint)

    def wrap_model_call(self, request, handler):
        return handler(self._request_with_web_search(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request_with_web_search(request))

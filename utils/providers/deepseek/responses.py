"""Bind DeepSeek hosted tools through the OpenAI Responses SDK adapter."""

from langchain_openai import ChatOpenAI

from .chat import DEEPSEEK_WEB_TOOLS


class DeepSeekWebChatOpenAI(ChatOpenAI):
    """Bind DeepSeek extensions after LangChain converts local functions."""

    def bind_tools(self, tools, **kwargs):
        hosted, local = [], []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") in DEEPSEEK_WEB_TOOLS:
                hosted.append(tool)
            else:
                local.append(tool)
        bound = super().bind_tools(local, **kwargs)
        if not hosted:
            return bound
        return bound.bind(tools=[*bound.kwargs.get("tools", []), *hosted])

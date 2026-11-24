"""Bind DeepSeek hosted tools through the OpenAI Responses SDK adapter."""

from langchain_openai import ChatOpenAI

from .chat import bind_deepseek_web_tools


class DeepSeekWebChatOpenAI(ChatOpenAI):
    """Bind DeepSeek extensions after LangChain converts local functions."""

    def bind_tools(self, tools, **kwargs):
        return bind_deepseek_web_tools(super().bind_tools, tools, **kwargs)

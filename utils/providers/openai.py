"""OpenAI adapter and first-party OpenAI request semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_openai import ChatOpenAI

from .base import RESPONSES, BaseChatAdapter, BaseVendor, Route, targets_https_api
from .deepseek.chat import DEEPSEEK_WEB_TOOLS, DeepSeekVendor


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


class OpenAIAdapter(BaseChatAdapter):
    type = "openai"
    default_profiles = {"openai": {"type": "openai", "api_mode": RESPONSES}}
    valid_kwargs = frozenset(
        {
            "streaming",
            "max_retries",
            "timeout",
            "reasoning_effort",
            "verbosity",
            "temperature",
            "model_kwargs",
            "extra_body",
            "profile",
        }
    )
    kwarg_names = {"timeout": "request_timeout"}
    api_key_field = "openai_api_key"
    base_url_field = "openai_api_base"

    def model_class(self) -> Any:
        return ChatOpenAI

    def model_class_for(self, route: Route) -> Any:
        if DeepSeekVendor().native_web_search(route)[0]:
            return DeepSeekWebChatOpenAI
        return self.model_class()

    def prepare(self, route: Route, params: dict[str, Any]) -> None:
        params["use_responses_api"] = route.api_mode == RESPONSES
        if DeepSeekVendor().native_web_search(route)[0]:
            params["output_version"] = "responses/v1"
            params["store"] = False


class OpenAIVendor(BaseVendor):
    """``api.openai.com`` only: compatible proxies share the adapter, not the extensions."""

    name = "openai"

    def serves(self, route: Route) -> bool:
        return route.adapter == "openai" and targets_https_api(route, "api.openai.com", allow_default=True)

    def request_overrides(self, route: Route, *, user_id: str, cache_key: str) -> dict[str, Any]:
        return {"model_kwargs": {"prompt_cache_key": cache_key}}

    def native_web_search(self, route: Route) -> tuple[bool, str] | None:
        if route.api_mode != RESPONSES:
            return None
        return True, "official OpenAI Responses web_search"

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str | None:
        return "json_schema" if capabilities.get("structured_output") is True else None

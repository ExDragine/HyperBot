"""DeepSeek adapter and first-party DeepSeek request semantics.

The adapter composes ``langchain_deepseek.ChatDeepSeek`` so upstream fixes
remain available. The vendor spans three protocols because DeepSeek serves
Chat Completions, OpenAI Responses and Anthropic Messages from one host.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_deepseek import ChatDeepSeek

from ..base import CHAT_COMPLETIONS, MESSAGES, RESPONSES, BaseChatAdapter, BaseVendor, Route, targets_https_api

_HOST = "api.deepseek.com"
_FEATURE_FIELDS = ("safety_check", "title")
_EFFORT_ALIASES = {"medium": "high", "xhigh": "max"}


class DeepSeekAdapter(BaseChatAdapter):
    type = "deepseek"
    model_prefixes = ("deepseek",)
    valid_kwargs = frozenset(
        {"streaming", "max_retries", "timeout", "temperature", "model_kwargs", "extra_body", "profile"}
    )
    api_key_field = "api_key"
    base_url_field = "api_base"

    def model_class(self) -> Any:
        return ChatDeepSeek

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str:
        """Thinking mode accepts only ``tool_choice: auto``, so keep the schema in the prompt."""
        return "text_json"


class DeepSeekVendor(BaseVendor):
    name = "deepseek"
    native_search_tools = ("deepseek_web_search", "deepseek_read_url")

    def official_api_mode(self, route: Route) -> str | None:
        """Return the protocol of an official DeepSeek route, if this is one.

        The OpenAI and Anthropic adapters also serve arbitrary proxies, so they
        need an explicit DeepSeek base URL. ``ChatDeepSeek`` alone defaults to
        the official endpoint when the base URL is empty.
        """
        if route.model_family != "deepseek":
            return None

        if route.adapter == "deepseek" and route.api_mode == CHAT_COMPLETIONS:
            official = targets_https_api(route, _HOST, allow_default=True)
        elif route.adapter == "openai" and route.api_mode == RESPONSES:
            official = targets_https_api(route, _HOST, allow_default=False)
        elif route.adapter == "anthropic" and route.api_mode == MESSAGES:
            official = targets_https_api(route, _HOST, paths=frozenset({"/anthropic"}), allow_default=False)
        else:
            official = False
        return route.api_mode if official else None

    def serves(self, route: Route) -> bool:
        return self.official_api_mode(route) is not None

    def request_overrides(self, route: Route, *, user_id: str, cache_key: str) -> dict[str, Any]:
        features = {
            key: value
            for key in _FEATURE_FIELDS
            if isinstance(value := route.profile.get(key), dict) and value
        }
        if route.api_mode == RESPONSES:
            overrides: dict[str, Any] = {"model_kwargs": {"user": user_id}}
            if features:
                # Not part of the OpenAI Responses SDK signature; the body
                # extension keeps the SDK from rejecting them locally.
                overrides["extra_body"] = features
            return overrides
        if route.api_mode == CHAT_COMPLETIONS:
            return {"extra_body": {"user_id": user_id, **features}}
        return {"model_kwargs": {"metadata": {"user_id": user_id}, **features}}

    def normalize_reasoning_effort(self, route: Route, effort: str) -> str:
        """DeepSeek Responses accepts ``low``, ``high`` and ``max``."""
        if route.api_mode != RESPONSES:
            return effort
        return _EFFORT_ALIASES.get(effort, effort)

    def native_web_search(self, route: Route) -> tuple[bool, str]:
        if route.profile.get("native_web_search") is True:
            return True, f"official DeepSeek {route.api_mode} chatbot tools explicitly opted in"
        return False, "DeepSeek hosted search requires explicit opt-in; prefer standalone deepseek_web_search"

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str:
        return "text_json"

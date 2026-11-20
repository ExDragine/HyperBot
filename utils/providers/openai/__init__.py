"""OpenAI adapter and first-party OpenAI request semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_openai import ChatOpenAI

from ..base import CHAT_COMPLETIONS, RESPONSES, BaseChatAdapter, BaseVendor, Route, targets_https_api


class OpenAIAdapter(BaseChatAdapter):
    type = "openai"
    api_modes = frozenset({CHAT_COMPLETIONS, RESPONSES})
    default_api_mode = CHAT_COMPLETIONS
    tool_search_model_markers = ("gpt",)
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

    def prepare(self, route: Route, params: dict[str, Any]) -> None:
        params["use_responses_api"] = route.api_mode == RESPONSES

    def structured_output_options(self, method: str) -> dict[str, Any]:
        options = super().structured_output_options(method)
        if method == "json_schema":
            options["strict"] = True
        return options


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

    def decision_backend(self, route: Route, *, max_retries: int, timeout: int):
        from .decisions import OpenAIDecisions

        return OpenAIDecisions(route, max_retries, timeout)

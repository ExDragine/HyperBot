"""OpenAI adapter and first-party OpenAI request semantics."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from langchain_openai import ChatOpenAI

from models import ApiMode, ModelFeature, get_model

from ..base import CHAT_COMPLETIONS, RESPONSES, BaseChatAdapter, BaseVendor, Route, clean, is_https_api_base

OPENAI_API_HOST = "api.openai.com"
_OPENAI_API_BASE = f"https://{OPENAI_API_HOST}/v1"
_TOOL_SEARCH_MODEL = re.compile(r"gpt-(\d+)(?:\.(\d+))?(?:-[a-z0-9.-]+)?")


def resolve_openai_base_url(configured: object) -> str:
    base_url = (
        clean(configured) or clean(os.getenv("OPENAI_API_BASE"))
        or clean(os.getenv("OPENAI_BASE_URL")) or _OPENAI_API_BASE
    )
    return _OPENAI_API_BASE if is_official_openai_base(base_url) else base_url


def is_official_openai_base(base_url: str) -> bool:
    return is_https_api_base(base_url, OPENAI_API_HOST)


def openai_tool_search_supported(route: Route) -> bool:
    if route.api_mode != RESPONSES or route.profile.get("native_tool_search") is False:
        return False
    if not is_official_openai_base(route.base_url):
        return route.profile.get("native_tool_search") is True
    model = route.model.lower()
    version = _TOOL_SEARCH_MODEL.fullmatch(model)
    card = get_model("openai", model)
    return (
        version is not None and (int(version[1]), int(version[2] or 0)) >= (5, 4)
        and card is not None and ApiMode.RESPONSES in card.capabilities.api_modes
        and ModelFeature.TOOL_CALLING in card.capabilities.features
    )


def validate_native_decisions(model: str, profile: Mapping[str, Any]) -> None:
    if profile.get("type") != "openai" or not is_official_openai_base(resolve_openai_base_url(profile.get("base_url"))):
        raise ValueError("native_decisions 需要支持该功能的官方 provider")
    if model != "gpt-6-luna":
        raise ValueError("OpenAI Decisions 的 decision_model 必须是 gpt-6-luna")
    if not clean(profile.get("api_key")):
        raise ValueError("OpenAI Decisions 需要在 provider profile 中配置官方 API key")


class OpenAIAdapter(BaseChatAdapter):
    type = "openai"
    api_modes = frozenset({CHAT_COMPLETIONS, RESPONSES})
    default_api_mode = CHAT_COMPLETIONS
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
        if route.api_mode == RESPONSES:
            params["store"] = route.profile.get("responses_store", False)
            if params["store"] is False and is_official_openai_base(route.base_url):
                params["include"] = ["reasoning.encrypted_content"]

    def structured_output_options(self, method: str) -> dict[str, Any]:
        options = super().structured_output_options(method)
        if method == "json_schema":
            options["strict"] = True
        return options


class OpenAIVendor(BaseVendor):
    """``api.openai.com`` only: compatible proxies share the adapter, not the extensions."""

    name = "openai"

    def serves(self, route: Route) -> bool:
        return route.adapter == "openai" and is_official_openai_base(resolve_openai_base_url(route.base_url))

    def request_overrides(self, route: Route, *, user_id: str, cache_key: str) -> dict[str, Any]:
        return {"model_kwargs": {"prompt_cache_key": cache_key}}

    def normalize_reasoning_effort(self, route: Route, effort: str) -> str:
        if route.model.lower().startswith("gpt-6.2-") and effort in {"none", "minimal"}:
            return "low"
        return effort

    def prepare_chat_model(self, route: Route, model_class: Any, params: dict[str, Any]) -> Any:
        if route.model.lower().startswith("gpt-6.2-"):
            params.pop("temperature", None)
            if effort := params.get("reasoning_effort"):
                params["reasoning_effort"] = self.normalize_reasoning_effort(route, effort)
        return model_class

    def native_web_search(self, route: Route) -> tuple[bool, str] | None:
        if route.api_mode != RESPONSES:
            return None
        card = get_model("openai", route.model)
        if (card is None or ApiMode.RESPONSES not in card.capabilities.api_modes
                or ModelFeature.WEB_SEARCH not in card.capabilities.features):
            return False, "model does not declare OpenAI Responses web_search support"
        return True, "official OpenAI Responses web_search"

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str | None:
        return "json_schema" if capabilities.get("structured_output") is True else None

    def decision_backend(self, route: Route, *, max_retries: int, timeout: int):
        from .decisions import OpenAIDecisions

        validate_native_decisions(route.model, {**route.profile, "type": route.adapter})
        return OpenAIDecisions(route, max_retries, timeout)

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
from .constants import API_HOST, DEFAULT_BASE_URL

_SAFETY = {"level": "high"}
_EFFORT_ALIASES = {"medium": "high", "xhigh": "max"}
_WEB_MODELS = frozenset({"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"})
DEEPSEEK_WEB_TOOLS = ("deepseek_web_search", "deepseek_read_url")


class DeepSeekAdapter(BaseChatAdapter):
    type = "deepseek"
    api_modes = frozenset({CHAT_COMPLETIONS})
    default_api_mode = CHAT_COMPLETIONS
    # One profile per protocol DeepSeek serves; the last two reuse other adapters.
    default_profiles = {
        "deepseek": {"type": "deepseek", "api_mode": CHAT_COMPLETIONS, "safety_check": _SAFETY},
        "deepseek_responses": {
            "type": "openai",
            "api_mode": RESPONSES,
            "base_url": DEFAULT_BASE_URL,
            "safety_check": _SAFETY,
        },
        "deepseek_anthropic": {
            "type": "anthropic",
            "api_mode": MESSAGES,
            "base_url": f"{DEFAULT_BASE_URL}/anthropic",
            "safety_check": _SAFETY,
        },
    }
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
    native_search_tools = DEEPSEEK_WEB_TOOLS
    profile_fields = ("safety_check", "title")

    def official_api_mode(self, route: Route) -> str | None:
        """Return the protocol of an official DeepSeek route, if this is one.

        The OpenAI and Anthropic adapters also serve arbitrary proxies, so they
        need an explicit DeepSeek base URL. ``ChatDeepSeek`` alone defaults to
        the official endpoint when the base URL is empty.
        """
        if route.model_family != "deepseek":
            return None

        if route.adapter == "deepseek" and route.api_mode == CHAT_COMPLETIONS:
            official = targets_https_api(route, API_HOST, allow_default=True)
        elif route.adapter == "openai" and route.api_mode == RESPONSES:
            official = targets_https_api(route, API_HOST, allow_default=False)
        elif route.adapter == "anthropic" and route.api_mode == MESSAGES:
            official = targets_https_api(route, API_HOST, paths=frozenset({"/anthropic"}), allow_default=False)
        else:
            official = False
        return route.api_mode if official else None

    def serves(self, route: Route) -> bool:
        return self.official_api_mode(route) is not None

    def request_overrides(self, route: Route, *, user_id: str, cache_key: str) -> dict[str, Any]:
        features = {
            key: value
            for key in self.profile_fields
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
        if self.official_api_mode(route) != RESPONSES or route.model.lower() not in _WEB_MODELS:
            return False, "DeepSeek search and URL reading require an official Flash Responses route"
        if route.profile.get("native_web_search") is False:
            return False, "DeepSeek web tools are disabled on this provider profile"
        return True, "official DeepSeek Flash Responses web tools"

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str:
        return "text_json"

    def prepare_chat_model(self, route: Route, model_class: Any, params: dict[str, Any]) -> Any:
        if not self.native_web_search(route)[0]:
            return model_class

        from .responses import DeepSeekWebChatOpenAI

        params["output_version"] = "responses/v1"
        params["store"] = False
        return DeepSeekWebChatOpenAI

    def web_search_middleware(self, *, user_text: str, user_id: str, bot_user_id: str | None):
        from .agent_web import DeepSeekWebSearchMiddleware

        return DeepSeekWebSearchMiddleware.for_turn(user_text=user_text, user_id=user_id, bot_user_id=bot_user_id)

    def standalone_web_search_hint(self, tool_names: set[str]) -> str:
        from .web import SEARCH_PROMPT_HINT, TOOL_NAMES

        return SEARCH_PROMPT_HINT if TOOL_NAMES <= tool_names else ""

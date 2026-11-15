"""Anthropic adapter and first-party Anthropic request semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from langchain_anthropic import ChatAnthropic

from .base import MESSAGES, BaseChatAdapter, BaseVendor, Route, targets_https_api

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class AnthropicAdapter(BaseChatAdapter):
    type = "anthropic"
    model_prefixes = ("claude-", "anthropic")
    valid_kwargs = frozenset({"streaming", "max_retries", "timeout", "temperature", "model_kwargs", "profile"})
    kwarg_names = {"timeout": "default_request_timeout"}
    api_key_field = "anthropic_api_key"
    base_url_field = "anthropic_api_url"

    def model_class(self) -> Any:
        return ChatAnthropic

    def prepare(self, route: Route, params: dict[str, Any]) -> dict[str, Any] | None:
        """Keep API metadata away from BaseChatModel's tracing metadata field."""
        model_kwargs = params.get("model_kwargs")
        if not isinstance(model_kwargs, dict) or "metadata" not in model_kwargs:
            return None

        remaining = {key: value for key, value in model_kwargs.items() if key != "metadata"}
        if remaining:
            params["model_kwargs"] = remaining
        else:
            params.pop("model_kwargs")
        return {"metadata": model_kwargs["metadata"]}

    def finalize(self, instance: BaseChatModel, state: dict[str, Any] | None) -> None:
        if not state:
            return
        existing = getattr(instance, "model_kwargs", None)
        if isinstance(existing, dict):
            existing.update(state)
        else:
            cast(Any, instance).model_kwargs = state

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str:
        return "function_calling"


class AnthropicVendor(BaseVendor):
    name = "anthropic"

    def serves(self, route: Route) -> bool:
        return (
            route.adapter == "anthropic"
            and route.api_mode == MESSAGES
            and targets_https_api(route, "api.anthropic.com", allow_default=True)
        )

    def request_overrides(self, route: Route, *, user_id: str, cache_key: str) -> dict[str, Any]:
        return {"model_kwargs": {"cache_control": {"type": "ephemeral"}}}

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str | None:
        return "json_schema" if capabilities.get("structured_output") is True else None

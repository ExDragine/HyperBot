"""DeepSeek extensions for LangChain's default adapter.

The package owns DeepSeek-specific transport details while returning a normal
LangChain chat model to the rest of HyperBot. It uses composition around
langchain_deepseek.ChatDeepSeek so upstream fixes remain available.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from langchain_core.language_models import ModelProfile
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

DEEPSEEK_VALID_KWARGS = frozenset(
    {
        "streaming",
        "max_retries",
        "timeout",
        "temperature",
        "model_kwargs",
        "extra_body",
        "profile",
    }
)


@dataclass(frozen=True, slots=True)
class DeepSeekProvider:
    """Build a DeepSeek LangChain model and classify its endpoint."""

    model: str
    profile: Mapping[str, Any]

    def create_chat_model(
        self,
        kwargs: Mapping[str, Any] | None = None,
        *,
        catalog_profile: ModelProfile | None = None,
    ) -> BaseChatModel:
        return create_chat_model(
            self.model,
            self.profile,
            kwargs or {},
            catalog_profile=catalog_profile,
        )

    def normalize_reasoning_effort(self, effort: str | None) -> str | None:
        """Map generic reasoning effort aliases to DeepSeek Responses values."""
        if effort is None:
            return None
        return {"medium": "high", "xhigh": "max"}.get(effort, effort)

    def needs_prompt_schema(self, provider_type: str) -> bool:
        """DeepSeek routes use prompt JSON when forced tool choice is unsafe."""
        return provider_type == "deepseek" or self.official_api_mode() is not None

    def official_api_mode(self) -> str | None:
        provider_type = str(self.profile.get("type") or "deepseek").lower()
        api_mode = str(self.profile.get("api_mode") or "chat_completions").lower()
        if provider_type == "deepseek_responses":
            provider_type, api_mode = "openai", "responses"
        return official_api_mode(
            model=self.model,
            provider_type=provider_type,
            api_mode=api_mode,
            profile=self.profile,
        )


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _targets_https_api(
    profile: Mapping[str, Any],
    hostname: str,
    *,
    paths: frozenset[str],
    allow_default: bool,
) -> bool:
    base_url = _clean(profile.get("base_url"))
    if not base_url:
        return allow_default

    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError:
        return False

    return (
        parsed.scheme.lower() == "https"
        and (parsed.hostname or "").lower() == hostname
        and parsed.path.rstrip("/") in paths
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
    )


def official_api_mode(
    *,
    model: str,
    provider_type: str,
    api_mode: str,
    profile: Mapping[str, Any],
) -> str | None:
    """Return the official DeepSeek protocol for this route, if applicable."""

    route_model = model.partition("/")[2] or model
    if not route_model.startswith("deepseek"):
        return None

    provider_type = provider_type.lower()
    api_mode = api_mode.lower()
    base_url = _clean(profile.get("base_url"))

    if provider_type == "deepseek" and api_mode == "chat_completions":
        official = _targets_https_api(
            profile,
            "api.deepseek.com",
            paths=frozenset({"", "/v1"}),
            allow_default=True,
        )
    elif provider_type == "openai" and api_mode == "responses":
        official = bool(base_url) and _targets_https_api(
            profile,
            "api.deepseek.com",
            paths=frozenset({"", "/v1"}),
            allow_default=False,
        )
    elif provider_type == "anthropic" and api_mode == "messages":
        official = bool(base_url) and _targets_https_api(
            profile,
            "api.deepseek.com",
            paths=frozenset({"/anthropic"}),
            allow_default=False,
        )
    else:
        official = False

    return api_mode if official else None


def create_chat_model(
    model: str,
    profile: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    catalog_profile: ModelProfile | None = None,
) -> BaseChatModel:
    """Create ChatDeepSeek with supported and normalized parameters."""

    filtered: dict[str, Any] = {
        key: value
        for key, value in kwargs.items()
        if key in DEEPSEEK_VALID_KWARGS or key == "tags"
    }
    if "profile" not in filtered and catalog_profile is not None:
        filtered["profile"] = catalog_profile

    base_url = _clean(profile.get("base_url"))
    if base_url:
        filtered["api_base"] = base_url

    api_key = SecretStr(_clean(profile.get("api_key")))
    return ChatDeepSeek(
        api_key=api_key,
        model=model,
        **filtered,
    )

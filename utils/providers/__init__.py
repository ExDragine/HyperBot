"""Provider registry.

Adapters and vendors load on first use, so importing this package (or its
``base`` tables) pulls in no vendor SDK. See ``docs/providers.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

from .base import ChatAdapter, Route, Vendor

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Order is the model-name inference order; ``openai`` is the fallback.
_ADAPTERS: dict[str, tuple[str, str]] = {
    "deepseek": ("utils.providers.deepseek", "DeepSeekAdapter"),
    "google": ("utils.providers.google", "GoogleAdapter"),
    "anthropic": ("utils.providers.anthropic", "AnthropicAdapter"),
    "openai": ("utils.providers.openai", "OpenAIAdapter"),
}
_VENDORS: dict[str, tuple[str, str]] = {
    "deepseek": ("utils.providers.deepseek", "DeepSeekVendor"),
    "openai": ("utils.providers.openai", "OpenAIVendor"),
    "anthropic": ("utils.providers.anthropic", "AnthropicVendor"),
}
_FALLBACK_ADAPTER = "openai"
_PROFILE_ORDER = ("openai", "google", "anthropic", "deepseek")
_TOOL_PACKAGES = ("utils.providers.deepseek.tools",)


def _load(entry: tuple[str, str]):
    module, attribute = entry
    return getattr(import_module(module), attribute)()


def adapter_types() -> tuple[str, ...]:
    return tuple(_ADAPTERS)


@cache
def get_adapter(adapter_type: str) -> ChatAdapter:
    return _load(_ADAPTERS[adapter_type])


@cache
def get_vendor(name: str) -> Vendor:
    return _load(_VENDORS[name])


def infer_adapter(model: str) -> str:
    """Pick the adapter a bare model id implies; ``vendor/model`` ids name it."""
    prefix, separator, route_key = model.partition("/")
    if separator and prefix.lower() in _ADAPTERS:
        return prefix.lower()
    route_key = route_key if separator else model
    for adapter_type in _ADAPTERS:
        prefixes = get_adapter(adapter_type).model_prefixes
        if prefixes and route_key.startswith(prefixes):
            return adapter_type
    return _FALLBACK_ADAPTER


def adapter_protocol(adapter_type: str) -> tuple[str, frozenset[str]]:
    if adapter_type not in _ADAPTERS:
        return "", frozenset()
    adapter = get_adapter(adapter_type)
    return adapter.default_api_mode, adapter.api_modes


def provider_tools() -> tuple[tuple[BaseTool, bool], ...]:
    """Collect direct tools and their read-only flags from provider packages."""
    return tuple(entry for package in _TOOL_PACKAGES for entry in import_module(package).direct_tools())


def standalone_web_search_hint(tool_names: set[str]) -> str:
    return "".join(get_vendor(name).standalone_web_search_hint(tool_names) for name in _VENDORS)


def default_profiles() -> dict[str, dict[str, Any]]:
    """Return fresh copies of every provider's built-in profiles."""
    profiles: dict[str, dict[str, Any]] = {}
    for adapter_type in _PROFILE_ORDER:
        for name, profile in get_adapter(adapter_type).default_profiles.items():
            profiles[name] = deepcopy(dict(profile))
    return profiles


def validate_profile_fields(name: str, profile: Mapping[str, Any]) -> None:
    """Reject vendor-specific profile keys that are not tables."""
    for vendor_name in _VENDORS:
        for field in get_vendor(vendor_name).profile_fields:
            if field in profile and not isinstance(profile[field], Mapping):
                raise ValueError(f"[providers.{name}].{field} 必须是 table")


def vendor_for(route: Route) -> Vendor | None:
    """Return the first-party vendor behind a route, or ``None`` for proxies."""
    for name in _VENDORS:
        vendor = get_vendor(name)
        if vendor.serves(route):
            return vendor
    return None


__all__ = [
    "Route",
    "adapter_protocol",
    "adapter_types",
    "default_profiles",
    "get_adapter",
    "get_vendor",
    "infer_adapter",
    "provider_tools",
    "standalone_web_search_hint",
    "validate_profile_fields",
    "vendor_for",
]

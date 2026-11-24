# ruff: noqa: S101
from __future__ import annotations

import pytest

from utils.providers import get_adapter, get_vendor, infer_adapter, vendor_for
from utils.providers.base import Route


def _route(model: str, adapter: str, api_mode: str, **profile) -> Route:
    return Route(model, profile, adapter, api_mode, infer_adapter(model))


@pytest.mark.parametrize(
    ("adapter", "api_mode", "base_url", "expected"),
    [
        ("deepseek", "chat_completions", "", "chat_completions"),
        ("deepseek", "chat_completions", "https://api.deepseek.com/v1", "chat_completions"),
        ("deepseek", "chat_completions", "https://proxy.example.com/v1", None),
        ("openai", "responses", "https://api.deepseek.com", "responses"),
        ("openai", "responses", "", None),
        ("openai", "responses", "https://proxy.example.com/v1", None),
        ("openai", "chat_completions", "https://api.deepseek.com", None),
        ("anthropic", "messages", "https://api.deepseek.com/anthropic", "messages"),
        ("anthropic", "messages", "https://api.deepseek.com", None),
        ("openai", "responses", "http://api.deepseek.com", None),
    ],
)
def test_official_routes_are_classified_by_protocol(adapter, api_mode, base_url, expected):
    route = _route("deepseek-v4-pro", adapter, api_mode, base_url=base_url)
    vendor = get_vendor("deepseek")

    assert vendor.official_api_mode(route) == expected
    assert (vendor_for(route) is vendor) is (expected is not None)


def test_other_model_families_never_reach_the_deepseek_vendor():
    route = _route("gpt-5.5", "openai", "responses", base_url="https://api.deepseek.com")

    assert not get_vendor("deepseek").serves(route)


def test_adapter_filters_unsupported_parameters(monkeypatch):
    captured = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("utils.providers.deepseek.chat.ChatDeepSeek", FakeChatDeepSeek)
    route = _route("deepseek-chat", "deepseek", "chat_completions", api_key="sk-test", base_url="https://gw.example/v1")
    model = get_adapter("deepseek").create_chat_model(
        route,
        {"temperature": 0.2, "reasoning_effort": "high", "unsupported": True, "tags": ["t"]},
        catalog_profile={"tool_calling": True},
    )

    assert isinstance(model, FakeChatDeepSeek)
    assert captured["model"] == "deepseek-chat"
    assert captured["api_key"].get_secret_value() == "sk-test"
    assert captured["api_base"] == "https://gw.example/v1"
    assert captured["temperature"] == 0.2
    assert captured["tags"] == ["t"]
    assert captured["profile"] == {"tool_calling": True}
    assert "unsupported" not in captured
    assert captured["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("api_mode", "effort", "expected"),
    [("responses", "medium", "high"), ("responses", "xhigh", "max"), ("responses", "low", "low"),
     ("chat_completions", "medium", "high"), ("chat_completions", "xhigh", "max"),
     ("chat_completions", "none", "none"), ("messages", "medium", "medium")],
)
def test_reasoning_effort_aliases_apply_to_openai_format_routes(api_mode, effort, expected):
    adapter = {"responses": "openai", "chat_completions": "deepseek", "messages": "anthropic"}[api_mode]
    route = _route("deepseek-v4-pro", adapter, api_mode, base_url="https://api.deepseek.com")

    assert get_vendor("deepseek").normalize_reasoning_effort(route, effort) == expected


def test_every_deepseek_route_keeps_the_schema_in_the_prompt():
    capabilities = {"tool_calling": True, "structured_output": True}
    official = _route("deepseek-v4-pro", "openai", "responses", base_url="https://api.deepseek.com")
    proxied = _route("deepseek-v4-pro", "deepseek", "chat_completions", base_url="https://proxy.example.com")

    assert get_vendor("deepseek").structured_output_method(official, capabilities) == "text_json"
    assert vendor_for(proxied) is None
    assert get_adapter("deepseek").structured_output_method(proxied, capabilities) == "text_json"


def test_hosted_flash_search_can_be_disabled():
    vendor = get_vendor("deepseek")
    base = {"base_url": "https://api.deepseek.com"}

    assert vendor.native_web_search(_route("deepseek-v4-pro", "openai", "responses", **base))[0] is False
    assert vendor.native_web_search(_route("deepseek-flash", "openai", "responses", **base))[0] is True
    assert vendor.native_web_search(
        _route("deepseek-flash", "openai", "responses", native_web_search=False, **base)
    )[0] is False
    assert vendor.native_search_tools == ("deepseek_web_search", "deepseek_read_url")

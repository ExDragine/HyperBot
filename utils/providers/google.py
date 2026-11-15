"""Google Gemini adapter. No vendor extensions are sent to Google routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base import BaseChatAdapter, Route


class GoogleAdapter(BaseChatAdapter):
    type = "google"
    model_prefixes = ("gemini-", "google")
    valid_kwargs = frozenset({"streaming", "max_retries", "timeout", "temperature", "profile"})
    api_key_field = "google_api_key"
    base_url_field = "base_url"

    def model_class(self) -> Any:
        return ChatGoogleGenerativeAI

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str:
        return "json_schema"

"""Preserve in-memory reasoning for DeepSeek Chat Completions tool calls."""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


class DeepSeekThinkingChat(ChatDeepSeek):
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        if payload.get("tools"):
            for message, serialized in zip(messages, payload["messages"], strict=True):
                if isinstance(message, AIMessage):
                    reasoning = message.additional_kwargs.get("reasoning_content")
                    serialized["reasoning_content"] = reasoning if isinstance(reasoning, str) else ""
        return payload

"""DeepSeek extensions to the Chat Completions wire format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_deepseek import ChatDeepSeek

from .chat import bind_deepseek_web_tools

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput


def _capture_web_fields(message, raw):
    for field in ("server_tool_calls", "annotations"):
        if isinstance(raw.get(field), list) and raw[field]:
            message.additional_kwargs[field] = raw[field]


def _restore_tool_images(serialized):
    # ChatDeepSeek JSON-encodes tool content lists after converting media blocks
    # to API format. Flash accepts image_url blocks directly in tool messages.
    content = serialized["content"]
    if not isinstance(content, str):
        return
    blocks = json.loads(content)
    if all(isinstance(block, dict) and block.get("type") in {"text", "image_url"} for block in blocks):
        if any(block["type"] == "image_url" for block in blocks):
            serialized["content"] = blocks


class DeepSeekChatCompletions(ChatDeepSeek):
    def bind_tools(self, tools, **kwargs):
        return bind_deepseek_web_tools(super().bind_tools, tools, **kwargs)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        thinking = (payload.get("extra_body") or {}).get("thinking", payload.get("thinking")) or {}
        mode = thinking.get("type")
        thinking_enabled = mode == "enabled" or (mode is None and payload.get("reasoning_effort") != "none")
        for message, serialized in zip(messages, payload["messages"], strict=True):
            if isinstance(message, AIMessage):
                fields = message.additional_kwargs
                steps = fields.get("server_tool_calls")
                if steps:
                    serialized["server_tool_calls"] = steps
                if payload.get("tools") and thinking_enabled:
                    reasoning = fields.get("reasoning_content")
                    reasoning = reasoning if isinstance(reasoning, str) else ""
                    if steps and fields.get("deepseek_streamed_reasoning"):
                        # Streaming concatenates all thinking; the wire message
                        # keeps earlier thinking inside its respective steps.
                        prefix = "".join(step.get("reasoning_content") or "" for step in steps)
                        reasoning = reasoning.removeprefix(prefix)
                    serialized["reasoning_content"] = reasoning
            elif isinstance(message, ToolMessage) and isinstance(message.content, list):
                _restore_tool_images(serialized)
        return payload

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        choices = response["choices"] if isinstance(response, dict) else response.choices
        for generation, choice in zip(result.generations, choices, strict=True):
            raw = choice["message"] if isinstance(choice, dict) else choice.message.model_dump(exclude={"parsed"})
            _capture_web_fields(generation.message, raw)
            if isinstance(reasoning := raw.get("reasoning_content"), str):
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        if error := chunk.get("error"):
            raise ValueError(error)
        result = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if result and isinstance(result.message, AIMessageChunk) and (choices := chunk.get("choices")):
            delta = choices[0].get("delta") or {}
            _capture_web_fields(result.message, delta)
            if delta.get("server_tool_calls"):
                result.message.additional_kwargs["deepseek_streamed_reasoning"] = True
        return result

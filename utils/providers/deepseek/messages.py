"""Preserve DeepSeek's hosted web extensions in the Anthropic Messages SDK."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from .chat import DEEPSEEK_CHATBOT_FIELDS, DEEPSEEK_WEB_TOOLS, bind_deepseek_web_tools

_WEB_RESULT_TYPES = {"web_search_tool_result", "web_fetch_tool_result"}


def _web_block_key(block):
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    if kind == "server_tool_use" and block.get("name") in DEEPSEEK_WEB_TOOLS:
        return kind, block.get("id")
    if kind in _WEB_RESULT_TYPES:
        return kind, block.get("tool_use_id")
    return None


class _DeepSeekMessageOutput:
    """Serialize DeepSeek blocks with their concrete SDK types."""

    def __init__(self, response):
        self._response = response
        self.usage = response.usage

    def model_dump(self):
        # The SDK's outer content union only recognizes Anthropic tool names.
        # Serializing each block directly retains DeepSeek's extension fields.
        data = self._response.model_dump(exclude={"content"})
        data["content"] = [block if isinstance(block, dict) else block.model_dump() for block in self._response.content]
        return data


class DeepSeekMessagesChat(ChatAnthropic):
    def bind_tools(self, tools, **kwargs):
        # Unlike the OpenAI formats, Messages requires a name on each web tool.
        tools = [
            {**tool, "name": tool["type"]}
            if isinstance(tool, dict) and tool.get("type") in DEEPSEEK_WEB_TOOLS else tool
            for tool in tools
        ]
        return bind_deepseek_web_tools(super().bind_tools, tools, **kwargs)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)

        fields = {key: payload.pop(key) for key in DEEPSEEK_CHATBOT_FIELDS if key in payload}
        if fields:
            payload["extra_body"] = {**fields, **(payload.get("extra_body") or {})}

        # Anthropic's formatter strips provider extensions. Restore DeepSeek's
        # fields on the SDK-formatted blocks, preserving order and tool IDs.
        original = {
            key: block
            for message in messages if isinstance(message, AIMessage) and isinstance(message.content, list)
            for block in message.content if (key := _web_block_key(block))
        }
        for message in payload["messages"]:
            for block in message["content"] if isinstance(message["content"], list) else ():
                if raw := original.get(_web_block_key(block)):
                    for field in ("caller", "kind", "signature"):
                        if raw.get(field) is not None:
                            block[field] = raw[field]
        return payload

    def _format_output(self, data, **kwargs):
        if error := getattr(data, "error", None):
            raise ValueError(error)
        if any(
            getattr(block, "type", None) == "server_tool_use" and getattr(block, "name", None) in DEEPSEEK_WEB_TOOLS
            for block in getattr(data, "content", ())
        ):
            data = _DeepSeekMessageOutput(data)
        return super()._format_output(data, **kwargs)

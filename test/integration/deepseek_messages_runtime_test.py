# ruff: noqa: S101
"""Wire contract for DeepSeek's Anthropic-compatible hosted web tools."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.usefixtures("offline_agent_eval")
def test_deepseek_messages_round_trips_web_tools_evidence_and_signatures(tmp_path):
    script = r"""
import asyncio
import json
import sys
sys.path.insert(0, sys.argv[1])
from anthropic import APIStatusError, AsyncAnthropic
from httpx2 import AsyncClient, MockTransport, Response
from langchain_core.messages import HumanMessage, ToolMessage, message_chunk_to_message
from langchain_core.tools import tool
from utils.agents.citations import cited_message
from utils.configs import EnvConfig
from utils.llm_factory import create_llm, native_web_search_tools, provider_request_overrides
from utils.providers.deepseek.agent_web import _deepseek_web_request_failed, deepseek_web_calls

NAME = "deepseek-flash"
PROVIDER = "deepseek_anthropic"
URL = "https://example.com/source"
DATE = "2026-11-18T10:00:00Z"
SEARCH_ID = "srvtoolu_ds_test"
READ_ID = "srvtoolu_rl_test"
EnvConfig.reload({"config_version": 2, "providers": {PROVIDER: {
    "api_key": "test-key", "title": {"enabled": True},
}}})

@tool
def read_image() -> str:
    'Read the requested image.'
    return "image"

WEB_BLOCKS = [
    {"type": "server_tool_use", "id": SEARCH_ID, "name": "deepseek_web_search", "kind": "deep_search",
     "input": {"action": "search", "queries": ["latest news"]}, "caller": {"type": "direct"}},
    {"type": "server_tool_use", "id": READ_ID, "name": "deepseek_read_url", "kind": "read_link",
     "input": {"action": "read_link", "url": URL}, "caller": {"type": "direct"}},
    # Results can follow a different ordering from their matching calls.
    {"type": "web_fetch_tool_result", "tool_use_id": READ_ID, "kind": "read_link",
     "content": {"type": "web_fetch_result", "url": URL, "published_at": DATE,
                 "content": {"type": "document", "title": "Source",
                             "source": {"type": "text", "media_type": "text/plain", "data": ""}}},
     "signature": "encrypted-page"},
    {"type": "web_search_tool_result", "tool_use_id": SEARCH_ID, "kind": "deep_search",
     "content": [{"url": URL, "title": "Source", "snippet": "Verbatim evidence", "page_age": DATE,
                  "type": "web_search_result", "encrypted_content": "encrypted-result"}]},
]
CITATION = {"type": "web_search_result_location", "cited_text": "已查", "encrypted_index": "",
            "url": URL, "title": "Source"}
TOOL_USE = {"type": "tool_use", "name": "read_image", "id": "toolu_test", "input": {}}
USAGE = {"input_tokens": 10, "output_tokens": 8}

def body(content, stop):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": NAME,
            "content": content, "stop_reason": stop, "stop_sequence": None, "usage": USAGE}

def stream():
    events = []

    def event(kind, **fields):
        events.append(f"event: {kind}\ndata: {json.dumps({'type': kind, **fields})}\n\n")

    event("message_start", message=body([], None))
    for index, block in enumerate(WEB_BLOCKS):
        event("content_block_start", index=index, content_block=block)
        event("content_block_stop", index=index)
    for index, text in enumerate(["已查", "证。"], start=len(WEB_BLOCKS)):
        event("content_block_start", index=index, content_block={"type": "text", "text": ""})
        event("content_block_delta", index=index, delta={"type": "text_delta", "text": text})
        if index == len(WEB_BLOCKS):
            event("content_block_delta", index=index, delta={"type": "citations_delta", "citation": CITATION})
        event("content_block_stop", index=index)
    event("content_block_start", index=6, content_block=TOOL_USE)
    event("content_block_stop", index=6)
    event("message_delta", delta={"stop_reason": "tool_use", "stop_sequence": None}, usage=USAGE)
    event("message_stop")
    return Response(200, headers={"content-type": "text/event-stream"}, text="".join(events))

async def run(streaming):
    seen = []

    def handle(request):
        assert request.url.path == "/anthropic/v1/messages"
        payload = json.loads(request.content)
        seen.append(payload)
        function, search, read = payload["tools"]
        assert function["name"] == "read_image" and "input_schema" in function
        assert search == {"type": "deepseek_web_search", "name": "deepseek_web_search"}
        assert read == {"type": "deepseek_read_url", "name": "deepseek_read_url"}
        assert payload["safety_check"] == {"level": "high"}
        assert payload["title"] == {"enabled": True}
        assert payload["metadata"] == {"user_id": "workspace-id"}
        if len(seen) == 1:
            if streaming:
                return stream()
            text = [{"type": "text", "text": "已查", "citations": [CITATION]}, {"type": "text", "text": "证。"}]
            return Response(200, json=body([*WEB_BLOCKS, *text, TOOL_USE], "tool_use"))

        assistant, tool_message = payload["messages"][-2:]
        assert assistant["role"] == "assistant"
        raw = [block for block in assistant["content"] if block["type"] not in {"text", "tool_use"}]
        assert [block["kind"] for block in raw] == ["deep_search", "read_link", "read_link", "deep_search"]
        assert raw[1] == WEB_BLOCKS[1]
        assert raw[2]["tool_use_id"] == READ_ID and raw[2]["signature"] == "encrypted-page"
        assert raw[2]["content"]["content"]["source"] == {"type": "text", "media_type": "text/plain", "data": ""}
        assert "index" not in raw[2]
        tool_result, = tool_message["content"]
        assert tool_result["tool_use_id"] == "toolu_test"
        assert tool_result["content"][0]["source"] == {
            "type": "base64", "media_type": "image/png", "data": "aW1hZ2U=",
        }
        return Response(200, json=body([{"type": "text", "text": "done"}], "end_turn"))

    model = create_llm(NAME, PROVIDER, max_retries=0, **provider_request_overrides(
        NAME, PROVIDER, user_id="workspace-id", cache_key="test",
    ))
    async with AsyncClient(transport=MockTransport(handle)) as http:
        model._async_client = AsyncAnthropic(api_key="test-key", base_url="https://api.deepseek.com/anthropic",
                                             http_client=http)
        bound = model.bind_tools([read_image, *({"type": kind} for kind in native_web_search_tools(NAME, PROVIDER))])
        history = [HumanMessage("Search and read the source")]
        if streaming:
            received = None
            async for part in bound.astream(history):
                received = part if received is None else received + part
            result = message_chunk_to_message(received)
        else:
            result = await bound.ainvoke(history)
        assert [(call["name"], call["id"]) for call in result.tool_calls] == [("read_image", "toolu_test")]
        text, sources = cited_message(result)
        assert text == "已查〔1〕证。" and [source.url for source in sources] == [URL], (text, sources)
        read, search = deepseek_web_calls(result)
        assert read["page"] == {"url": URL, "title": "Source", "published_at": DATE}
        assert search["action"]["sources"][0]["published_at"] == DATE
        assert search["action"]["sources"][0]["snippet"] == "Verbatim evidence"
        result = await bound.ainvoke([*history, result, ToolMessage(
            [{"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"}], tool_call_id="toolu_test",
        )])
        assert result.text == "done"

async def check_stream_error():
    def handle(request):
        error = {"type": "error", "error": {"type": "api_error", "message": "Generation failed"}}
        return Response(200, headers={"content-type": "text/event-stream"},
                        text=f"event: error\ndata: {json.dumps(error)}\n\n")

    model = create_llm(NAME, PROVIDER, max_retries=0)
    async with AsyncClient(transport=MockTransport(handle)) as http:
        model._async_client = AsyncAnthropic(api_key="test-key", base_url="https://api.deepseek.com/anthropic",
                                             http_client=http)
        try:
            async for _ in model.astream("Search the web"):
                pass
        except APIStatusError as exc:
            assert _deepseek_web_request_failed(exc)
        else:
            raise AssertionError("Expected an API error from the Anthropic SSE stream")

async def main():
    await run(False)
    await run(True)
    await check_stream_error()

asyncio.run(main())
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pydantic serializer warnings" not in result.stderr, result.stderr

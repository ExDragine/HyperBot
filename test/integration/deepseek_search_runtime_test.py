# ruff: noqa: S101
"""DeepSeek's native web tools and evidence round-trip through the real SDKs."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.agent_eval
@pytest.mark.usefixtures("offline_agent_eval")
def test_deepseek_native_web_tools_round_trip_through_the_real_sdk(tmp_path):
    script = r"""
import asyncio
import json
import sys
sys.path.insert(0, sys.argv[1])
from httpx import AsyncClient, MockTransport, Response
from langchain_core.tools import tool
from openai import AsyncOpenAI
from utils.configs import EnvConfig
from utils.llm_factory import create_llm, native_web_search_tools

EnvConfig.reload({"config_version": 2, "providers": {"deepseek_responses": {"api_key": "test-key"}}})

@tool
def local_lookup(query: str) -> str:
    'Look up a local record.'
    return query

def handle(request):
    assert request.url.path == "/responses"
    payload = json.loads(request.content)
    local, search, read = payload["tools"]
    assert local["type"] == "function" and local["name"] == "local_lookup"
    assert search == {"type": "deepseek_web_search"}
    assert read == {"type": "deepseek_read_url"}
    assert payload["store"] is False
    url = "https://example.com/article"
    return Response(200, json={
        "id": "resp_test", "object": "response", "status": "completed", "created_at": 0,
        "model": "deepseek-flash", "output": [
            {"id": "search_test", "type": "web_search_call", "status": "completed", "kind": "deep_search",
             "action": {"type": "search", "queries": ["latest news"], "sources": [{
                 "type": "url", "url": url, "title": "Article", "snippet": "Verbatim search evidence",
                 "published_at": "2026-11-18T10:00:00Z",
             }]}},
            {"id": "read_test", "type": "web_search_call", "status": "completed", "kind": "read_link",
             "action": {"type": "open_page", "url": url}, "page": {"url": url}},
            {"id": "msg_test", "type": "message", "status": "completed", "role": "assistant",
             "content": [{"type": "output_text", "text": "Read and verified", "annotations": []}]},
        ],
    })

async def main():
    provider = "deepseek_responses"
    name = "deepseek-flash"
    model = create_llm(name, provider=provider, max_retries=0)
    async with AsyncClient(transport=MockTransport(handle)) as client:
        model.root_async_client = AsyncOpenAI(api_key="test-key", base_url="https://api.deepseek.com", http_client=client)
        hosted = [{"type": kind} for kind in native_web_search_tools(name, provider)]
        result = await model.bind_tools([local_lookup, *hosted]).ainvoke("Find and read the original source")
    search, read, _text = result.content
    assert search["action"]["sources"][0]["snippet"] == "Verbatim search evidence"
    assert read["page"]["url"] == "https://example.com/article"
    assert result.text == "Read and verified"
    assert not result.tool_calls

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


@pytest.mark.usefixtures("offline_agent_eval")
def test_deepseek_chat_completions_preserves_hosted_tools_citations_and_media(tmp_path):
    script = r"""
import asyncio
import json
import sys
sys.path.insert(0, sys.argv[1])
from httpx import AsyncClient, MockTransport, Response
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_chunk_to_message
from langchain_core.tools import tool
from openai import AsyncOpenAI
from utils.agents.citations import cited_message
from utils.configs import EnvConfig
from utils.llm_factory import create_llm, native_web_search_tools
from utils.providers.deepseek.agent_web import deepseek_web_calls

EnvConfig.reload({"config_version": 2, "providers": {"deepseek": {"api_key": "test-key"}}})
NAME = "deepseek-flash"
URL = "https://example.com/source"
DATE = "2026-11-18T10:00:00Z"
ANNOTATIONS = [{"type": "url_citation", "url_citation": {
    "url": URL, "title": "Source", "start_index": 0, "end_index": 2,
}}]
STEPS = [{"index": 0, "reasoning_content": "earlier thinking", "calls": [
    {"id": "search_test", "index": 0, "type": "deepseek_web_search", "kind": "deep_search",
     "action": {"type": "search", "queries": ["latest news"]}, "status": "completed",
     "sources": [{"url": URL, "title": "Source", "snippet": "Verbatim evidence", "published_at": DATE}]},
    {"id": "read_test", "index": 1, "type": "deepseek_read_url", "kind": "read_link", "status": "completed",
     "action": {"type": "read_link", "url": URL}, "page": {"url": URL, "published_at": DATE},
     "signature": "encrypted-page"},
]}]
TOOL_CALL = {"id": "call_image", "type": "function", "function": {"name": "read_image", "arguments": "{}"}}
TOOL_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}

@tool
def read_image() -> str:
    'Read the requested image.'
    return "image"

def reply(body, finish="stop"):
    return Response(200, json={"id": "chatcmpl_test", "object": "chat.completion", "created": 1,
                               "model": NAME, "choices": [{"index": 0, "message": body, "finish_reason": finish}]})

def chunks():
    parts = [
        ({"role": "assistant", "reasoning_content": "earlier thinking"}, None),
        ({"reasoning_content": "final reasoning", "content": "已查"}, None),
        ({"content": "证。", "tool_calls": [{"index": 0, **TOOL_CALL}]}, None),
        ({"server_tool_calls": STEPS, "annotations": ANNOTATIONS}, "tool_calls"),
    ]
    events = [{"id": "chatcmpl_test", "object": "chat.completion.chunk", "created": 1, "model": NAME,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]} for delta, finish in parts]
    return Response(200, headers={"content-type": "text/event-stream"},
                    text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n")

async def run(streaming):
    seen = []

    def handle(request):
        assert request.url.path == "/chat/completions"
        payload = json.loads(request.content)
        seen.append(payload)
        function, search, read = payload["tools"]
        assert function["type"] == "function" and function["function"]["name"] == "read_image"
        assert search == {"type": "deepseek_web_search"} and read == {"type": "deepseek_read_url"}
        assert "reasoning_effort" not in payload
        assert payload["messages"][0]["reasoning_content"] == ""
        if len(seen) == 1:
            assert payload["stream"] == streaming
            if streaming:
                return chunks()
            return reply({"role": "assistant", "content": "已查证。", "reasoning_content": "final reasoning",
                          "server_tool_calls": STEPS, "annotations": ANNOTATIONS, "tool_calls": [TOOL_CALL]}, "tool_calls")
        assistant, tool_message = payload["messages"][-2:]
        assert assistant["server_tool_calls"] == STEPS
        assert assistant["reasoning_content"] == "final reasoning" and assistant["content"] == "已查证。"
        assert tool_message["content"] == [{"type": "text", "text": "screenshot"}, TOOL_IMAGE]
        return reply({"role": "assistant", "content": "done"})

    model = create_llm(NAME, "deepseek", max_retries=0)
    async with AsyncClient(transport=MockTransport(handle)) as http:
        client = AsyncOpenAI(api_key="test-key", base_url="https://api.deepseek.com", http_client=http)
        model.async_client = client.chat.completions
        bound = model.bind_tools([read_image, *({"type": kind} for kind in native_web_search_tools(NAME, "deepseek"))])
        history = [AIMessage("Previous reply"), HumanMessage("Search and inspect the screenshot")]
        if streaming:
            received = None
            async for part in bound.astream(history):
                received = part if received is None else received + part
            result = message_chunk_to_message(received)
        else:
            result = await bound.ainvoke(history)
        assert [(call["name"], call["id"]) for call in result.tool_calls] == [("read_image", "call_image")]
        text, sources = cited_message(result)
        assert text == "已查〔1〕证。" and [source.url for source in sources] == [URL]
        search, read = deepseek_web_calls(result)
        assert search["action"]["sources"][0]["published_at"] == DATE
        assert read["signature"] == "encrypted-page"
        result = await bound.ainvoke([*history, result, ToolMessage(
            [{"type": "text", "text": "screenshot"},
             {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"}], tool_call_id="call_image",
        )])
        assert result.text == "done"

async def main():
    await run(False)
    await run(True)

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


@pytest.mark.parametrize("provider", ["deepseek_responses", "deepseek", "deepseek_anthropic"])
@pytest.mark.usefixtures("offline_agent_eval")
def test_deepseek_native_web_failure_resumes_the_existing_agent_turn(tmp_path, provider):
    script = r"""
import asyncio
import json
import sys
from itertools import count
sys.path.insert(0, sys.argv[1])
from anthropic import AsyncAnthropic
from httpx2 import AsyncClient, MockTransport, Response
from langchain.agents import create_agent
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.tools import tool
from openai import AsyncOpenAI
from tools.deepseek_search import deepseek_web_fetch, deepseek_web_search
from utils.agent_context import FrontierRuntimeContext
from utils.configs import EnvConfig
from utils.llm_factory import create_llm, native_web_search_middleware, native_web_search_tools
from utils.providers.deepseek import web

MODEL = "deepseek-flash"
PROVIDER = sys.argv[2]
CHAT = PROVIDER == "deepseek"
MESSAGES = PROVIDER == "deepseek_anthropic"
EnvConfig.reload({"config_version": 2, "providers": {name: {"api_key": "test-key"} for name in {
    PROVIDER, "deepseek_responses",
}}})
HOSTED = native_web_search_tools(MODEL, PROVIDER)
FAILED_SEARCH = {"id": "search_test", "type": "web_search_call", "status": "failed", "kind": "search",
                 "action": {"type": "search", "queries": ["verify"]}, "error": {"code": "unavailable"}}
ANSWER = {"id": "msg_test", "type": "message", "status": "completed", "role": "assistant",
          "content": [{"type": "output_text", "text": "Task continued", "annotations": []}]}
response_ids = count()

def response(*items, status="completed", error=None):
    if MESSAGES:
        if error:
            return Response(200, json={"type": "error", "error": {"type": "api_error", "message": "Generation failed"}})
        content, stop = [], "refusal" if status == "incomplete" else "end_turn"
        for item in items:
            if item["type"] == "web_search_call":
                content.extend([
                    {"type": "server_tool_use", "id": item["id"], "name": "deepseek_web_search", "kind": item["kind"],
                     "input": {"action": "search", "queries": item["action"]["queries"]}},
                    {"type": "web_search_tool_result", "tool_use_id": item["id"], "kind": item["kind"],
                     "content": [{"type": "web_search_tool_result_error", "error_code": item["error"]["code"]}]},
                ])
            elif item["type"] == "function_call":
                content.append({"type": "tool_use", "id": item["call_id"], "name": item["name"],
                                "input": json.loads(item["arguments"])})
                stop = "tool_use"
            else:
                content.append({"type": "text", "text": item["content"][0]["text"]})
        return Response(200, json={
            "id": f"msg_{next(response_ids)}", "type": "message", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 8},
        })
    if CHAT:
        if error:
            return Response(200, json={"error": {"type": "api_error", "message": "Generation failed"}})
        message = {"role": "assistant", "content": "", "reasoning_content": "fixture"}
        finish = "stop"
        for item in items:
            if item["type"] == "web_search_call":
                message["server_tool_calls"] = [{"index": 0, "reasoning_content": "", "calls": [
                    {**item, "type": "deepseek_web_search", "index": 0},
                ]}]
            elif item["type"] == "function_call":
                message["tool_calls"] = [{"id": item["call_id"], "type": "function", "function": {
                    "name": item["name"], "arguments": item["arguments"],
                }}]
                finish = "tool_calls"
            else:
                message["content"] = item["content"][0]["text"]
        if status == "incomplete":
            finish = "content_filter"
        return Response(200, json={
            "id": f"chatcmpl_{next(response_ids)}", "object": "chat.completion", "created": 0,
            "model": MODEL, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        })
    return Response(200, json={"id": f"resp_{next(response_ids)}", "object": "response", "status": status, "created_at": 0,
                               "model": MODEL, "output": items, "error": error})

async def check(case):
    requests, executions, standalone_calls = [], [], []
    source_url = "https://example.com/article"

    @tool
    def local_action(value: str) -> str:
        'Perform the requested local action.'
        executions.append(value)
        return f"done: {value}"

    def local_call(name, arguments):
        return response({"id": f"fc_{name}", "type": "function_call", "name": name,
                         "call_id": f"call_{name}", "arguments": json.dumps(arguments)})

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        step = len(requests)
        assert "previous_response_id" not in payload
        if not (CHAT or MESSAGES):
            assert payload["store"] is False
        input_key, output_key = ("messages", "content") if CHAT or MESSAGES else ("input", "output")
        assert [item["type"] for item in payload["tools"] if item.get("type") in HOSTED] == (
            list(HOSTED) if step <= 2 else []
        )
        expected_functions = {"local_action"} if step <= 2 else {
            "local_action", "deepseek_web_search", "deepseek_web_fetch",
        }
        assert {item.get("function", item)["name"] for item in payload["tools"]
                if item.get("type", "function") not in HOSTED} == expected_functions
        prompt = json.dumps(payload["system"] if MESSAGES else payload[input_key][0], ensure_ascii=False)
        assert prompt.count("Original system prompt") == 1
        assert prompt.count("本轮可用 DeepSeek 服务端联网") == (1 if step <= 2 else 0)
        assert prompt.count("DeepSeek 服务端搜索和网页读取暂不可用") == (0 if step <= 2 else 1)
        assert ("deepseek_web_fetch" in prompt) == (step > 2)
        if step == 1:
            return local_call("local_action", {"value": "first"})
        if step == 2:
            if case == "invalid_parameters":
                return Response(400, json={"error": {"message": "tools: deepseek_web_search rejected"}})
            if case == "unprocessable_parameters":
                return Response(422, json={"error": {"message": "Invalid tool parameters"}})
            if case == "failed_response":
                return response(FAILED_SEARCH, status="failed", error={"code": "server_error", "message": "Retry"})
            if case == "failed_call_with_answer":
                return response(FAILED_SEARCH, ANSWER)
            if case == "content_review":
                return Response(400, json={"error": {"message": "Content Exists Risk"}})
            if case == "content_filter":
                return response(FAILED_SEARCH, status="incomplete")
            return response(FAILED_SEARCH)
        if step == 3:
            system_length = 0 if MESSAGES else 1
            assert payload[input_key][system_length:] == requests[1][input_key][system_length:]
            return local_call("deepseek_web_search", {"query": "verify"})
        if step == 4:
            assert any(source_url in str(item.get(output_key, "")) for item in payload[input_key])
            return local_call("deepseek_web_fetch", {"url": source_url})
        assert step == 5
        assert any("verified page body" in str(item.get(output_key, "")) for item in payload[input_key])
        return response(ANSWER)

    def handle_standalone(request):
        assert request.headers["authorization"] == "Bearer test-key"
        operation = request.url.path
        standalone_calls.append(operation)
        if operation == "/v1/tools/search":
            assert json.loads(request.content)["query"] == "verify"
            return Response(200, json={"results": [{"title": "Article", "url": source_url,
                                                   "rerank_score": 2.0, "authority": 3}]})
        assert operation == "/v1/tools/fetch"
        assert json.loads(request.content) == {"url": source_url}
        return Response(200, json={"title": "Article", "content": "verified page body"})

    model = create_llm(MODEL, provider=PROVIDER, max_retries=0, streaming=False)
    async with (
        AsyncClient(transport=MockTransport(handle)) as client,
        AsyncClient(transport=MockTransport(handle_standalone)) as standalone,
    ):
        web.get_http_client = lambda *args, **kwargs: standalone
        if MESSAGES:
            model._async_client = AsyncAnthropic(api_key="test-key", base_url="https://api.deepseek.com/anthropic",
                                                 http_client=client)
        else:
            model.root_async_client = AsyncOpenAI(api_key="test-key", base_url="https://api.deepseek.com", http_client=client)
            if CHAT:
                model.async_client = model.root_async_client.chat.completions
        agent = create_agent(model, tools=[local_action, deepseek_web_search, deepseek_web_fetch],
                             context_schema=FrontierRuntimeContext, system_prompt="Original system prompt",
                             middleware=[native_web_search_middleware(
                                 MODEL, PROVIDER, user_text="Complete the task", user_id="test-user", bot_user_id=None,
                             )])
        try:
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "Complete the task"}]},
                                         context=FrontierRuntimeContext("test-user", None, None, "."))
        except ModelInvalidRequestError:
            assert case == "content_review"
        else:
            assert case != "content_review"
            expected = "" if case == "content_filter" else "Task continued"
            assert result["messages"][-1].text == expected, (case, executions, result["messages"])
    if case in {"content_review", "content_filter", "failed_call_with_answer"}:
        assert len(requests) == 2 and not standalone_calls
    else:
        assert len(requests) == 5 and standalone_calls == ["/v1/tools/search", "/v1/tools/fetch"]
    assert executions == ["first"]

async def main():
    for case in ("invalid_parameters", "unprocessable_parameters", "failed_response", "failed_call",
                 "failed_call_with_answer", "content_review"):
        await check(case)
    if CHAT or MESSAGES:
        await check("content_filter")

asyncio.run(main())
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2]), provider],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr

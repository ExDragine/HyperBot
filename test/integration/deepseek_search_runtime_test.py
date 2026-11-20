# ruff: noqa: S101
"""DeepSeek Responses keeps its native tool types and provider evidence on the wire."""

import subprocess
import sys
from pathlib import Path


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


def test_deepseek_native_web_failure_resumes_the_existing_agent_turn(tmp_path):
    script = r"""
import asyncio
import json
import sys
from itertools import count
sys.path.insert(0, sys.argv[1])
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

EnvConfig.reload({"config_version": 2, "providers": {"deepseek_responses": {"api_key": "test-key"}}})
MODEL = "deepseek-flash"
PROVIDER = "deepseek_responses"
HOSTED = native_web_search_tools(MODEL, PROVIDER)
FAILED_SEARCH = {"id": "search_test", "type": "web_search_call", "status": "failed", "kind": "search",
                 "action": {"type": "search", "queries": ["verify"]}, "error": {"code": "unavailable"}}
ANSWER = {"id": "msg_test", "type": "message", "status": "completed", "role": "assistant",
          "content": [{"type": "output_text", "text": "Task continued", "annotations": []}]}
response_ids = count()

def response(*items, status="completed", error=None):
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
        assert payload["store"] is False
        assert [item["type"] for item in payload["tools"] if item["type"] != "function"] == (
            list(HOSTED) if step <= 2 else []
        )
        expected_functions = {"local_action"} if step <= 2 else {
            "local_action", "deepseek_web_search", "deepseek_web_fetch",
        }
        assert {item["name"] for item in payload["tools"] if item["type"] == "function"} == expected_functions
        prompt = json.dumps(payload["input"][0], ensure_ascii=False)
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
            return response(FAILED_SEARCH)
        if step == 3:
            assert payload["input"][1:] == requests[1]["input"][1:]
            return local_call("deepseek_web_search", {"query": "verify"})
        if step == 4:
            assert any(source_url in item.get("output", "") for item in payload["input"])
            return local_call("deepseek_web_fetch", {"url": source_url})
        assert step == 5
        assert any("verified page body" in item.get("output", "") for item in payload["input"])
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
        model.root_async_client = AsyncOpenAI(api_key="test-key", base_url="https://api.deepseek.com", http_client=client)
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
            assert result["messages"][-1].text == "Task continued", (case, executions, result["messages"])
    if case in {"content_review", "failed_call_with_answer"}:
        assert len(requests) == 2 and not standalone_calls
    else:
        assert len(requests) == 5 and standalone_calls == ["/v1/tools/search", "/v1/tools/fetch"]
    assert executions == ["first"]

async def main():
    for case in ("invalid_parameters", "unprocessable_parameters", "failed_response", "failed_call",
                 "failed_call_with_answer", "content_review"):
        await check(case)

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

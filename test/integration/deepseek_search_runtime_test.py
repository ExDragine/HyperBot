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

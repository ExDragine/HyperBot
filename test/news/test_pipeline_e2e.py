# ruff: noqa: S101

"""End-to-end wiring for the agent-led DeepSeek news pipeline."""

import datetime as dt
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.news.config import NewsConfig
from plugins.news.repository import NewsRepository
from plugins.news.schemas import NewsPayload

ARTICLE_TEXT = "央行今日宣布启用新型货币政策工具，受访市场人士普遍认为这释放了明确的流动性支持信号，后续观察落地节奏。"
ARTICLE_URL = "https://example.com/news/2026/10/24/central-bank-announces-new-policy-tool.html"


async def fake_main_agent(_runtime, request):
    assert request.pinned_skills == ("daily-news",)
    assert request.job_prompt == "daily-news"
    assert request.include_messages and request.allow_native_web_search
    assert not request.allow_local_tools
    data = json.loads(request.prompt)
    assert data["mode"] == "research" and "articles" not in data
    window = data["window"]
    published = dt.datetime.fromisoformat(window[1]) - dt.timedelta(hours=1)
    payload = NewsPayload(
        top_stories=[
            {
                "title": "央行启用新型货币政策工具",
                "category": "财经",
                "summary": ARTICLE_TEXT[:60],
                "evidence": [
                    {
                        "article_id": ARTICLE_URL,
                        "quote": ARTICLE_TEXT[5:35],
                    }
                ],
            }
        ],
        worth_reading=[],
    )
    web_calls = [
        {"type": "web_search_call", "status": "completed", "kind": "deep_search",
         "action": {"type": "search", "sources": [{
                "title": "央行启用新型货币政策工具",
                "url": ARTICLE_URL,
                "snippet": ARTICLE_TEXT,
                "site_name": "example.com",
         }]}},
        {"type": "web_search_call", "status": "completed", "kind": "read_link",
         "action": {"type": "open_page", "url": ARTICLE_URL},
         "page": {"url": ARTICLE_URL, "published_at": published.isoformat()}},
    ]
    return {"status": "success", "response": {"messages": [SimpleNamespace(
        type="ai", content=[*web_calls, {"type": "text", "text": payload.model_dump_json()}],
    )]}}


class _Sent:
    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload

    async def send(self, target=None, bot=None):
        FakeUniMessage.recorded.append((self.kind, target.id, self.payload))


class FakeUniMessage:
    recorded = []

    @classmethod
    def text(cls, text):
        return _Sent("text", text)

    def image(self, raw=None, **_kwargs):
        return _Sent("image", raw)


@pytest.fixture
def news_e2e(monkeypatch, tmp_path):
    from plugins.news import delivery, editor, scheduler

    monkeypatch.setenv("FRONTIER_NEWS_DB", str(tmp_path / "news.db"))
    monkeypatch.setattr(
        scheduler,
        "load_news_config",
        lambda: NewsConfig(
            min_stories=1,
            target_stories=1,
            queries=("news",),
        ),
    )
    monkeypatch.setattr(scheduler, "render_image", AsyncMock(return_value=b"png-bytes"))
    monkeypatch.setattr(editor, "model_supports_native_web_search", lambda *_: True)
    monkeypatch.setattr(editor, "native_web_search_tools", lambda *_: ("deepseek_web_search", "deepseek_read_url"))
    monkeypatch.setattr(editor.FrontierAgentRuntime, "run", fake_main_agent)
    FakeUniMessage.recorded = []
    monkeypatch.setattr(delivery, "UniMessage", FakeUniMessage)

    clockwork_pkg = sys.modules.get("plugins.clockwork")
    if clockwork_pkg is None:
        clockwork_pkg = types.ModuleType("plugins.clockwork")
        clockwork_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "plugins" / "clockwork")]
        monkeypatch.setitem(sys.modules, "plugins.clockwork", clockwork_pkg)
    manager = SimpleNamespace(get_task_groups=AsyncMock(return_value=[123]))
    monkeypatch.setattr(clockwork_pkg, "task_manager", manager, raising=False)
    return scheduler


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepseek_responses", "deepseek", "deepseek_anthropic"])
async def test_daily_news_full_pipeline(news_e2e, tmp_path, monkeypatch, provider):
    from plugins.news import editor
    from utils.llm_factory import model_supports_native_web_search, native_web_search_tools

    scheduler = news_e2e
    monkeypatch.setattr(editor.EnvConfig, "ADVAN_MODEL", "deepseek-flash")
    monkeypatch.setattr(editor.EnvConfig, "ADVAN_MODEL_PROVIDER", provider)
    monkeypatch.setattr(editor, "model_supports_native_web_search", model_supports_native_web_search)
    monkeypatch.setattr(editor, "native_web_search_tools", native_web_search_tools)

    if provider == "deepseek":
        async def run_chat_completions(runtime, request):
            result = await fake_main_agent(runtime, request)
            message = result["response"]["messages"][0]
            search, read, text = message.content
            message.content = text["text"]
            message.additional_kwargs = {"server_tool_calls": [{"calls": [
                {**search, "type": "deepseek_web_search", "sources": search["action"]["sources"],
                 "action": {"type": "search"}},
                {**read, "type": "deepseek_read_url", "action": {"type": "read_link", "url": ARTICLE_URL}},
            ]}]}
            return result

        monkeypatch.setattr(editor.FrontierAgentRuntime, "run", run_chat_completions)
    elif provider == "deepseek_anthropic":
        async def run_messages(runtime, request):
            result = await fake_main_agent(runtime, request)
            message = result["response"]["messages"][0]
            search, read, text = message.content
            message.content = [
                {"type": "server_tool_use", "id": "web_search", "kind": "deep_search",
                 "name": "deepseek_web_search", "input": {"action": "search"}},
                {"type": "server_tool_use", "id": "web_read", "kind": "read_link",
                 "name": "deepseek_read_url", "input": {"action": "read_link", "url": ARTICLE_URL}},
                {"type": "web_fetch_tool_result", "tool_use_id": "web_read", "kind": "read_link",
                 "content": {"type": "web_fetch_result", **read["page"]}},
                {"type": "web_search_tool_result", "tool_use_id": "web_search", "kind": "deep_search",
                 "content": [{"type": "web_search_result", **source} for source in search["action"]["sources"]]},
                text,
            ]
            return result

        monkeypatch.setattr(editor.FrontierAgentRuntime, "run", run_messages)

    result = await scheduler.daily_news(job_id="daily_news")

    assert result.status == "success"
    assert result.messages_sent == 1
    assert FakeUniMessage.recorded == [("image", "123", b"png-bytes")]

    report = await NewsRepository(tmp_path / "news.db").latest("general")
    assert report["status"] == "ready"
    assert report["image"] == b"png-bytes"
    story = report["payload"]["top_stories"][0]
    assert story["title"] == "央行启用新型货币政策工具"
    assert report["evidence"][0]["text"] == ARTICLE_TEXT
    assert story["evidence"][0]["article_id"] == report["evidence"][0]["article_id"]
    assert len(story["evidence"][0]["article_id"]) == 32

    result = await scheduler.daily_news(job_id="daily_news")
    assert result.status == "success"
    assert result.messages_sent == 0
    assert len(FakeUniMessage.recorded) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_evidence_first", [False, True])
async def test_daily_news_repairs_draft_before_archiving_and_delivering(
    news_e2e, tmp_path, monkeypatch, missing_evidence_first,
):
    from plugins.news import editor

    calls = []

    async def generate_draft(runtime, request):
        calls.append(request.allow_native_web_search)
        data = json.loads(request.prompt)
        if request.allow_native_web_search:
            if missing_evidence_first and "feedback" not in data:
                return {"status": "success", "response": {"messages": [SimpleNamespace(type="ai", content="{}")]}}
            result = await fake_main_agent(runtime, request)
            block = result["response"]["messages"][0].content[-1]
            if missing_evidence_first:
                block["text"] = "{}"
            else:
                draft = NewsPayload.model_validate_json(block["text"])
                draft.worth_reading = [draft.top_stories[0].model_copy(update={"title": "流动性支持信号"})]
                block["text"] = draft.model_dump_json()
            return result
        assert request.pinned_skills == ("daily-news",) and not request.allow_local_tools
        assert request.job_prompt == "daily-news"
        assert data["mode"] == "revision"
        assert data["schema"] == NewsPayload.model_json_schema()
        assert data["validation_error"] == (
            "the initial draft did not match the news schema" if missing_evidence_first
            else "the same article cannot fill multiple news slots"
        )
        assert len(data["articles"]) == 1
        assert data["min_stories"] == 1
        draft = NewsPayload(
            top_stories=[{
                "title": "央行启用新型货币政策工具",
                "category": "财经",
                "summary": ARTICLE_TEXT[:60],
                "evidence": [{
                    "article_id": data["articles"][0]["article_id"],
                    "quote": ARTICLE_TEXT[5:35],
                }],
            }],
            worth_reading=[],
        )

        return {"status": "success", "response": {"messages": [SimpleNamespace(
            type="ai", content=draft.model_dump_json(),
        )]}}

    monkeypatch.setattr(editor.FrontierAgentRuntime, "run", generate_draft)

    result = await news_e2e.daily_news(job_id="daily_news")

    assert result.status == "success"
    assert result.messages_sent == 1
    assert calls == ([True, True, False] if missing_evidence_first else [True, False])
    report = await NewsRepository(tmp_path / "news.db").latest("general")
    assert report["status"] == "ready"
    assert report["payload"]["worth_reading"] == []
    assert len(report["payload"]["top_stories"]) == 1
    assert FakeUniMessage.recorded == [("image", "123", b"png-bytes")]

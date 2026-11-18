# ruff: noqa: S101

import datetime as dt
from types import SimpleNamespace

import pytest

from plugins.news.config import NewsConfig
from plugins.news.editor import _article_records, _articles_from_messages
from plugins.news.schemas import Edition
from plugins.news.sources import (
    ExaMcpSource,
    _mcp_articles,
    publication_time,
)


@pytest.mark.parametrize("value", ["2026-09-19T00:00:00Z", "Sat, 19 Sep 2026 00:00:00 GMT", "Sat, 19 Sep 2026 08:00:00 +0800"])
def test_publication_timestamp_formats(value):
    assert publication_time(value) == dt.datetime(2026, 9, 19, tzinfo=dt.UTC)


@pytest.mark.parametrize("value", [None, "", "yesterday", "2026-09-19", "2026-09-19T00:00:00"])
def test_unknown_timezone_or_date_is_not_invented(value):
    assert publication_time(value) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("dictionary_schema", [False, True])
async def test_exa_mcp_is_used_when_rest_key_is_missing(monkeypatch, dictionary_schema):
    calls = []

    class FakeTool:
        name = "web_search_exa"
        args_schema = SimpleNamespace(
            model_fields={
                "query": object(),
                "numResults": object(),
                "startPublishedDate": object(),
                "endPublishedDate": object(),
            }
        )

        async def ainvoke(self, args):
            calls.append(args)
            return SimpleNamespace(
                artifact={
                    "structured_content": {
                        "results": [
                            {
                                "title": "实时新闻标题",
                                "url": "https://example.com/news",
                                "text": "这是一段足够长的新闻正文，用于验证 Exa MCP 返回的证据可以进入日报流程。",
                                "publishedDate": "2026-09-18T12:00:00Z",
                            }
                        ]
                    }
                }
            )

    if dictionary_schema:
        FakeTool.args_schema = {"type": "object", "properties": {
            name: {"type": "string"} for name in FakeTool.args_schema.model_fields
        }}

    import sys
    import types

    mcp_client = types.ModuleType("tools.mcp_client")

    async def fake_discovery():
        return [FakeTool()]

    mcp_client.mcp_get_tools_async = fake_discovery
    monkeypatch.setitem(sys.modules, "tools.mcp_client", mcp_client)
    edition = Edition("general", dt.datetime(2026, 9, 19, 1, tzinfo=dt.UTC))
    cfg = NewsConfig(min_stories=1, target_stories=1, queries=("news",))
    source = ExaMcpSource(cfg)

    articles = await source.search("news", edition)

    assert len(articles) == 1
    assert articles[0].provider == "exa"
    assert articles[0].published_at == dt.datetime(2026, 9, 18, 12, tzinfo=dt.UTC)
    assert calls[0]["query"] == "news"
    assert calls[0]["numResults"] == cfg.source_results
    assert calls[0]["startPublishedDate"] == "2026-09-18T01:00:00+00:00"


def test_search_snippet_is_evidence_and_read_link_can_fill_its_date():
    url = "https://example.com/news"
    snippet = "这是服务端搜索真实返回的一段足够长的新闻摘要，用于验证日报可以保存并检查候选稿件引自此处的原句。"
    search = SimpleNamespace(type="ai", content=[{
        "type": "web_search_call", "kind": "search", "status": "completed",
        "action": {"type": "search", "sources": [{"url": url, "title": "来源标题", "snippet": snippet}]},
    }])
    read = SimpleNamespace(type="ai", content=[{
        "type": "web_search_call", "kind": "read_link", "status": "completed",
        "action": {"type": "open_page", "url": url}, "page": {"url": url, "published_at": "2026-09-18T12:00:00Z"},
    }, {"type": "text", "text": "这段由模型生成的要点不能用于逐字引用或代替来源摘要。"}])
    records = _article_records([search, read])
    assert "content" not in records[url]
    articles = _articles_from_messages(
        [search, read], Edition("general", dt.datetime(2026, 9, 19, 1, tzinfo=dt.UTC))
    )
    assert articles[0].text == snippet
    assert articles[0].published_at == dt.datetime(2026, 9, 18, 12, tzinfo=dt.UTC)


def test_exa_mcp_markdown_keeps_publication_date():
    result = """
[实时新闻标题](https://example.com/news)
Published Date: 2026-09-18T12:00:00Z
这是一段足够长的新闻正文，用于验证纯文本 MCP 结果中的发布时间不会丢失。
"""
    articles = _mcp_articles(result)
    assert len(articles) == 1
    assert articles[0].published_at == dt.datetime(2026, 9, 18, 12, tzinfo=dt.UTC)

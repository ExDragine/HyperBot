"""News research and editing through HyperBot's Agent runtime."""

import hashlib
import json
from datetime import UTC, datetime
from urllib.parse import urlsplit

from utils.agents.assistant import parse_structured_response_from_messages
from utils.agents.runtime_gateway import AgentRuntimeRequest, FrontierAgentRuntime
from utils.configs import EnvConfig
from utils.llm_factory import model_supports_native_web_search, native_web_search_tools

from .schemas import Article, Edition, NewsPayload, canonical_url, compact, eligible, validate_evidence
from .sources import publication_time


def _web_call_records(messages):
    for message in messages:
        blocks = getattr(message, "content", None)
        if getattr(message, "type", None) != "ai" or not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "web_search_call":
                continue
            action = block.get("action")
            if block.get("status") != "completed" or not isinstance(action, dict):
                continue
            if block.get("kind") in {"search", "deep_search"} and action.get("type") == "search":
                sources = action.get("sources")
                if isinstance(sources, list):
                    yield from ((item, True) for item in sources if isinstance(item, dict))
            elif isinstance(page := block.get("page"), dict):
                yield page, False


def _article_records(messages) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for item, is_search in _web_call_records(messages):
        try:
            url = canonical_url(str(item.get("url") or ""))
        except ValueError:
            continue
        record = records.setdefault(url, {"url": url})
        if is_search:
            record.update({key: item[key] for key in ("title", "snippet", "published_at", "site_name") if item.get(key)})
        elif item.get("published_at"):
            record.setdefault("published_at", item["published_at"])
    return records


def _articles_from_messages(messages, edition: Edition) -> list[Article]:
    now = datetime.now(UTC)
    timezone = edition.scheduled_at.tzinfo
    articles = []
    for record in _article_records(messages).values():
        title = compact(str(record.get("title") or ""))[:300]
        text = compact(str(record.get("snippet") or ""))[:6000]
        if len(title) < 1 or len(text) < 40:
            continue
        url = record["url"]
        fingerprint = hashlib.sha256((title + "\n" + text).encode()).hexdigest()
        articles.append(
            Article(
                article_id=hashlib.sha256((url + fingerprint).encode()).hexdigest()[:32],
                url=url,
                title=title,
                text=text,
                source=str(record.get("site_name") or urlsplit(url).hostname or "deepseek"),
                provider="deepseek",
                published_at=publication_time(record.get("published_at"), default_timezone=timezone),
                fetched_at=now,
                fingerprint=fingerprint,
            )
        )
    return articles


def _bind_evidence(payload: NewsPayload, articles: list[Article]) -> NewsPayload:
    aliases = {}
    for article in articles:
        aliases[article.article_id] = article.article_id
        aliases[article.url] = article.article_id
        aliases[article.title.casefold()] = article.article_id

    def bind(story):
        evidence = []
        for support in story.evidence:
            raw = support.article_id.strip()
            try:
                raw = canonical_url(raw)
            except ValueError:
                raw = raw.casefold()
            evidence.append(support.model_copy(update={"article_id": aliases.get(raw, support.article_id)}))
        return story.model_copy(update={"evidence": evidence})

    return payload.model_copy(
        update={
            "top_stories": [bind(story) for story in payload.top_stories],
            "worth_reading": [bind(story) for story in payload.worth_reading],
        }
    )


class NewsEditor:
    def __init__(self, cfg, *, research_agent: bool = False):
        self.cfg = cfg
        self.research_agent = research_agent

    async def _run_skill(self, edition: Edition, data: dict, *, research: bool) -> list:
        prompt = json.dumps({
            **data,
            "mode": "research" if research else "revision",
            "window": [edition.start.isoformat(), edition.scheduled_at.isoformat()],
            "target_stories": self.cfg.target_stories,
            "min_stories": self.cfg.min_stories,
            "schema": NewsPayload.model_json_schema(),
        }, ensure_ascii=False)
        result = await FrontierAgentRuntime().run(AgentRuntimeRequest(
            session_id=f"daily-news:{edition.report_id}",
            workspace_key=f"daily-news-{edition.namespace}",
            user_name="Daily news",
            prompt=prompt,
            access_profile="frontier",
            job_prompt="daily-news",
            pinned_skills=("daily-news",),
            include_messages=research,
            allow_local_tools=False,
            allow_native_web_search=research,
        ))
        if result.get("error") or result.get("status") != "success":
            raise RuntimeError("daily news Skill execution failed")
        return result["response"]["messages"]

    async def research_and_edit(self, edition: Edition):
        if not self.research_agent:
            raise RuntimeError("agent research is disabled")
        route = EnvConfig.ADVAN_MODEL, EnvConfig.ADVAN_MODEL_PROVIDER
        if not model_supports_native_web_search(*route) or "deepseek_read_url" not in native_web_search_tools(*route):
            raise RuntimeError("daily news requires a main model with official DeepSeek search and URL reading")
        data = {"queries": list(self.cfg.queries)}
        messages = await self._run_skill(edition, data, research=True)
        articles = eligible(_articles_from_messages(messages, edition), edition)
        if not articles:
            messages = await self._run_skill(edition, {
                **data, "feedback": "上一轮没有取得符合日期要求的搜索摘要。请调整搜索词和时间范围提示。",
            }, research=True)
            articles = eligible(_articles_from_messages(messages, edition), edition)
        try:
            payload = parse_structured_response_from_messages(messages, NewsPayload)
        except (KeyError, ValueError):
            return articles, None
        return articles, _bind_evidence(payload, articles)

    async def edit(self, edition, articles, *, validation_error: str | None = None):
        messages = await self._run_skill(edition, {
            "validation_error": validation_error,
            "articles": [article.model_dump(mode="json") for article in articles],
        }, research=False)
        return validate_evidence(parse_structured_response_from_messages(messages, NewsPayload), articles)

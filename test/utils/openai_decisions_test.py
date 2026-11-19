# ruff: noqa: S101

import json

import httpx2
import pytest

from utils.agents.capture import detect_browser_capture_intent
from utils.configs import EnvConfig, parse_config
from utils.decision import LLMDecisionProvider
from utils.decision import openai as decisions


@pytest.fixture
def native_decisions(monkeypatch):
    profile = {
        "type": "openai",
        "api_mode": "responses",
        "api_key": "sk-test",
        "native_decisions": True,
        "decision_extra_body": {"reasoning": {"effort": "none"}},
    }
    monkeypatch.setattr(EnvConfig, "LLM_PROVIDERS", {"openai": profile})
    monkeypatch.setattr(EnvConfig, "DECISION_MODEL", "gpt-6-luna")
    monkeypatch.setattr(EnvConfig, "DECISION_MODEL_PROVIDER", "openai")
    return profile


def response(answers):
    return httpx2.Response(200, json={
        "model": "gpt-6-luna",
        "answers": answers,
        "usage": {
            "input_tokens": 25,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 25,
        },
    })


def test_native_decisions_is_explicit_and_keeps_responses_as_chat_protocol():
    config = parse_config({
        "config_version": 2,
        "providers": {"openai": {"native_decisions": True}},
    })

    assert config.providers["openai"].native_decisions is True
    assert config.providers["openai"].api_mode == "responses"
    assert config.providers["deepseek"].native_decisions is False


@pytest.mark.asyncio
async def test_native_decisions_sends_typed_questions_and_normalizes_answers(monkeypatch, native_decisions):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com")

    def handle(request):
        assert str(request.url) == "https://api.openai.com/v1/decisions"
        assert request.headers["Authorization"] == "Bearer sk-test"
        payload = json.loads(request.content)
        assert set(payload) == {"model", "input", "questions"}
        assert payload["model"] == "gpt-6-luna"
        assert payload["input"] == [{
            "role": "user", "content": [{"type": "input_json", "data": {"kind": "request"}}],
        }]
        predicate, choice, score, _ = payload["questions"]
        assert predicate["name"] == "should_reply"
        assert predicate["type"] == "predicate"
        assert "latest message" in predicate["instructions"]
        assert "判断标准" in predicate["instructions"]
        assert choice["choices"] == [{"value": "accept"}, {"value": "defer"}]
        assert score["levels"] == [{"label": "low"}, {"label": "high"}]
        return response([
            {"name": "action", "type": "choice", "choice": "defer", "confidence": 0.8, "probabilities": []},
            {"name": "urgency", "type": "score", "score": 0.75, "confidence": 0.9, "probabilities": []},
            {"name": "should_reply", "type": "predicate", "probability": 0.6},
            {"name": "refused", "type": "refusal"},
        ])

    questions = {
        "should_reply": {"type": "predicate", "instructions": "Assess the latest message", "criteria": {"true": "request"}},
        "action": {"type": "choice", "choices": ["accept", "defer"]},
        "urgency": {"type": "score", "levels": ["low", "high"]},
        "refused": {"type": "predicate"},
    }
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        monkeypatch.setattr(decisions, "get_http_client", lambda _name: client)
        provider = LLMDecisionProvider(max_retries=0)
        result = await provider.decide({"kind": "request"}, questions)
        assert not client.is_closed

    assert result.provider == "openai:decisions"
    assert result.usage["input_tokens"] == 25
    assert result.answers["should_reply"] == {"type": "predicate", "probability": 0.6}
    assert result.answers["action"] == {"type": "choice", "choice": "defer", "confidence": 0.8}
    assert result.answers["urgency"] == {"type": "score", "score": 0.75, "confidence": 0.9}
    assert result.answers["refused"]["type"] == "refusal"
    assert result.answers["refused"]["reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize("recording_name, tools", [("recording", {"webpage_screenshot"}), ("screenshot", set())])
async def test_browser_capture_uses_native_decisions_and_rejects_duplicate_answers(
    monkeypatch, native_decisions, recording_name, tools,
):
    def handle(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/decisions"
        assert payload["input"] == "给这个网页截图"
        assert [(item["name"], item["type"]) for item in payload["questions"]] == [
            ("screenshot", "predicate"), ("recording", "predicate"),
        ]
        return response([
            {"name": "screenshot", "type": "predicate", "probability": 0.9},
            {"name": recording_name, "type": "predicate", "probability": 0.5},
        ])

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        monkeypatch.setattr(decisions, "get_http_client", lambda _name: client)
        assert await detect_browser_capture_intent("给这个网页截图") == tools


def test_native_decisions_rejects_proxy_profile(native_decisions):
    native_decisions["base_url"] = "https://example.com/v1"

    with pytest.raises(ValueError, match="官方 OpenAI provider"):
        LLMDecisionProvider()


@pytest.mark.asyncio
async def test_native_decisions_requires_a_key_for_the_configured_official_profile(monkeypatch, native_decisions):
    native_decisions.pop("api_key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-custom-gateway")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com")

    with pytest.raises(ValueError, match="官方 API key"):
        await LLMDecisionProvider().decide("test", {"answer": {"type": "predicate"}})

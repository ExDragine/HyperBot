# ruff: noqa: S101
from types import SimpleNamespace

import pytest

from utils.decision import DecisionEnvelope, LLMDecisionProvider, build_reply_gate_state, score_reply_gate


def test_build_reply_gate_state_is_bounded_and_normalizes_content():
    state = build_reply_gate_state(
        "最新问题",
        [
            {"role": "user", "content": "旧问题"},
            SimpleNamespace(role="assistant", content=[{"type": "text", "text": "旧回答"}]),
        ],
    )

    assert state == "user: 旧问题\nassistant: 旧回答\nuser: 最新问题"


@pytest.mark.asyncio
async def test_llm_decision_provider_adapts_structured_answers(monkeypatch):
    from utils.decision import llm as decision_llm

    captured = {}

    class FakeDecisionLLM:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        async def structured(self, system_prompt, user_prompt, schema, **kwargs):
            captured.update({"system": system_prompt, "user": user_prompt, "schema": schema, "options": kwargs})
            return DecisionEnvelope(
                answers={"should_reply": {"type": "predicate", "probability": 0.8, "confidence": 0.9}}
            )

    monkeypatch.setattr(decision_llm, "DecisionLLM", FakeDecisionLLM)
    provider = LLMDecisionProvider(model="decision-model", provider="decision-provider")

    score = await score_reply_gate(provider, "这个报错怎么解决？")

    assert score.should_reply is True
    assert score.probability == 0.8
    assert score.decision.provider == "llm:decision-provider"
    assert captured["kwargs"] == {"model": "decision-model", "provider": "decision-provider", "max_retries": 2, "timeout": 30}
    assert captured["schema"] is DecisionEnvelope
    assert captured["options"]["method"] == "text_json"
    assert '"should_reply"' in captured["user"]


@pytest.mark.asyncio
async def test_llm_decision_provider_rejects_out_of_range_probability(monkeypatch):
    from utils.decision import llm as decision_llm

    class FakeDecisionLLM:
        def __init__(self, **_kwargs):
            pass

        async def structured(self, *_args, **_kwargs):
            return DecisionEnvelope(answers={"should_reply": {"type": "predicate", "probability": 2}})

    monkeypatch.setattr(decision_llm, "DecisionLLM", FakeDecisionLLM)
    with pytest.raises(ValueError, match="between 0 and 1"):
        await LLMDecisionProvider().decide("hello", {"should_reply": {"type": "predicate"}})

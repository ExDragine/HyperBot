from __future__ import annotations


from utils.providers.deepseek import DeepSeekProvider, official_api_mode


def test_official_routes_are_classified_by_protocol():
    assert official_api_mode(
        model="deepseek-chat",
        provider_type="deepseek",
        api_mode="chat_completions",
        profile={"base_url": "https://api.deepseek.com/v1"},
    ) == "chat_completions"
    assert official_api_mode(
        model="deepseek-v4",
        provider_type="openai",
        api_mode="responses",
        profile={"base_url": "https://api.deepseek.com"},
    ) == "responses"
    assert official_api_mode(
        model="deepseek-chat",
        provider_type="openai",
        api_mode="responses",
        profile={"base_url": "https://proxy.example.com/v1"},
    ) is None


def test_provider_exposes_a_langchain_model_factory(monkeypatch):
    captured = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("utils.providers.deepseek.ChatDeepSeek", FakeChatDeepSeek)
    model = DeepSeekProvider("deepseek-chat", {"api_key": "sk-test"}).create_chat_model(
        {"temperature": 0.2, "unsupported": True}
    )

    assert isinstance(model, FakeChatDeepSeek)
    assert captured["model"] == "deepseek-chat"
    assert captured["temperature"] == 0.2
    assert "unsupported" not in captured

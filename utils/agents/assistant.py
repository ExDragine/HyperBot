"""Lightweight Agent helper used by scheduled jobs and tools."""

import json
import re
from typing import Any, Literal

from langchain.agents import create_agent
from pydantic import ValidationError

from utils.configs import EnvConfig
from utils.llm_factory import chat_reasoning_kwargs, create_llm, model_supports
from utils.message import extract_message_text

from .inputs import build_user_content

type ModelRole = Literal["basic", "decision", "signal", "advanced", "daily_news"]


def _configured_model_route(model: str, role: ModelRole | None) -> tuple[ModelRole | None, str | None]:
    role_routes: dict[ModelRole, tuple[str, str]] = {
        "basic": (EnvConfig.BASIC_MODEL, EnvConfig.BASIC_MODEL_PROVIDER),
        "advanced": (EnvConfig.ADVAN_MODEL, EnvConfig.ADVAN_MODEL_PROVIDER),
        "decision": (EnvConfig.DECISION_MODEL, EnvConfig.DECISION_MODEL_PROVIDER),
        "signal": (EnvConfig.SIGNAL_MODEL, EnvConfig.SIGNAL_MODEL_PROVIDER),
        "daily_news": (EnvConfig.DAILY_NEWS_MODEL, EnvConfig.DAILY_NEWS_MODEL_PROVIDER),
    }
    if role is not None:
        return role, role_routes[role][1]
    for configured_role, (configured_model, provider) in role_routes.items():
        if model == configured_model:
            return configured_role, provider
    return None, None


def json_document_candidates(text: str, *, prefer_object: bool = False) -> list[str]:
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    decoder = json.JSONDecoder()
    candidates = []
    for index, char in enumerate(text):
        if char not in ("{" if prefer_object else "{["):
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index : index + end])
    return candidates or [text]


def parse_structured_response_from_messages(messages: list, response_format):
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        text = extract_message_text(message).strip()
        if not text:
            continue
        if hasattr(response_format, "model_validate_json"):
            last_error = None
            for candidate in json_document_candidates(text, prefer_object=True):
                try:
                    return response_format.model_validate_json(candidate)
                except ValidationError as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
        break
    raise KeyError("structured_response")


async def assistant_agent(  # noqa: C901
    system_prompt: str = "",
    user_prompt: str = "",
    use_model: str | None = None,
    model_role: ModelRole | None = None,
    tools=None,
    response_format=None,
    middleware=None,
    images: list[bytes] | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    model_kwargs: dict | None = None,
    invoke_config: dict | None = None,
    return_result: bool = False,
    model_provider: str | None = None,
    model_timeout: int = 300,
    model_extra_body: dict | None = None,
) -> Any:
    if use_model is None:
        use_model = EnvConfig.BASIC_MODEL
        model_role = model_role or "basic"
    configured_role, provider = _configured_model_route(use_model, model_role)
    if model_provider is not None:
        provider = model_provider
    responses_fallback = None
    if reasoning_effort is None:
        if configured_role == "basic":
            reasoning_effort = EnvConfig.BASIC_REASONING_EFFORT
        elif configured_role == "advanced":
            reasoning_effort = EnvConfig.ADVANCED_REASONING_EFFORT
            responses_fallback = EnvConfig.AGENT_CAPABILITY
    llm_kwargs: dict[str, Any] = {
        "model": use_model,
        "streaming": False,
        "max_retries": 2,
        "timeout": model_timeout,
        "tags": ["frontier:assistant"],
        **chat_reasoning_kwargs(use_model, provider, reasoning_effort, responses_fallback=responses_fallback),
    }
    if provider is not None:
        llm_kwargs["provider"] = provider
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if model_kwargs is not None:
        llm_kwargs["model_kwargs"] = model_kwargs
    if model_extra_body:
        llm_kwargs["extra_body"] = model_extra_body
    model = create_llm(**llm_kwargs)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        response_format=response_format,
        debug=EnvConfig.AGENT_DEBUG_MODE,
    )
    request = {
        "messages": [
            {
                "role": "user",
                "content": build_user_content(
                    user_prompt,
                    images,
                    supports_vision=model_supports(use_model, "vision", role=model_role),
                ),
            }
        ]
    }
    if invoke_config is None:
        result = await agent.ainvoke(request)
    else:
        result = await agent.ainvoke(request, config=invoke_config)
    if return_result:
        return result
    if response_format:
        if "structured_response" in result:
            return result["structured_response"]
        return parse_structured_response_from_messages(result.get("messages", []), response_format)
    content = ""
    for message in result["messages"]:
        if message.type == "ai" and message.text:
            content += str(message.text)
    return content

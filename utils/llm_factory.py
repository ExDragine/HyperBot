from typing import TYPE_CHECKING, Any

from langchain_core.language_models import ModelProfile

from models import ApiMode, ModelFeature, ModelInput, ModelOutput, get_model, load_catalog
from utils.configs import EnvConfig
from utils.providers import Route, adapter_protocol, get_adapter, infer_adapter, vendor_for
from utils.providers.openai import openai_tool_search_supported, resolve_openai_base_url

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models import BaseChatModel

    from utils.decision.models import DecisionProvider


def _clean_optional(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _infer_provider(model: str) -> str:
    return infer_adapter(model)


def _provider_profile(model: str, provider: str | None) -> tuple[str, dict]:
    profile_name = _clean_optional(provider) or _infer_provider(model)
    profile = EnvConfig.LLM_PROVIDERS.get(profile_name)
    if profile is None:
        raise ValueError(f"未知 LLM provider profile: {profile_name!r}")
    if not isinstance(profile, dict):
        raise ValueError(f"LLM provider profile 必须是表格: {profile_name!r}")
    return profile_name, profile


def _normalize_capabilities(capabilities: object) -> set[str]:
    if not isinstance(capabilities, list):
        return set()
    normalized = {
        capability.strip().lower() for capability in capabilities if isinstance(capability, str) and capability.strip()
    }
    aliases = {"image": "vision", "attachment": "file", "pdf": "file"}
    return {aliases.get(capability, capability) for capability in normalized}


def _model_specific_capabilities(model: str, role: str | None = None) -> set[str]:
    role_capabilities = {
        "basic": (EnvConfig.BASIC_MODEL, EnvConfig.BASIC_MODEL_CAPABILITIES),
        "decision": (EnvConfig.DECISION_MODEL, EnvConfig.DECISION_MODEL_CAPABILITIES),
        "advanced": (EnvConfig.ADVAN_MODEL, EnvConfig.ADVAN_MODEL_CAPABILITIES),
        "daily_news": (EnvConfig.DAILY_NEWS_MODEL, EnvConfig.DAILY_NEWS_MODEL_CAPABILITIES),
    }
    if role == "signal":
        configured_model, configured_capabilities = EnvConfig.SIGNAL_MODEL, EnvConfig.SIGNAL_MODEL_CAPABILITIES
        return _normalize_capabilities(configured_capabilities) if model == configured_model else set()
    if role is not None:
        configured = role_capabilities.get(role)
        if configured is None:
            raise ValueError(f"未知模型角色: {role!r}")
        configured_model, role_capabilities_list = configured
        return _normalize_capabilities(role_capabilities_list) if model == configured_model else set()

    capabilities: set[str] = set()
    configured_roles = list(role_capabilities.values())
    configured_roles.append((EnvConfig.SIGNAL_MODEL, EnvConfig.SIGNAL_MODEL_CAPABILITIES))
    for configured_model, configured_capabilities in configured_roles:
        if model == configured_model:
            capabilities.update(_normalize_capabilities(configured_capabilities))
    return capabilities


def get_model_capabilities(model: str, *, role: str | None = None) -> set[str]:
    capabilities = _model_specific_capabilities(model, role)
    if capabilities:
        return capabilities

    profile = get_langchain_model_profile(model, _infer_provider(model))
    if profile is None:
        return {"text"}
    resolved = {"text"} if profile.get("text_inputs") else set()
    if profile.get("image_inputs") or profile.get("image_url_inputs"):
        resolved.add("vision")
    if profile.get("audio_inputs"):
        resolved.add("audio")
    if profile.get("video_inputs"):
        resolved.add("video")
    if profile.get("pdf_inputs") or profile.get("attachment"):
        resolved.add("file")
    return resolved or {"text"}


def model_supports(model: str, capability: str, *, role: str | None = None) -> bool:
    requested = _normalize_capabilities([capability])
    return bool(requested & get_model_capabilities(model, role=role))


def _provider_protocol(model: str, profile: dict) -> tuple[str, str]:
    provider_type = (_clean_optional(profile.get("type")) or _infer_provider(model)).lower()
    default_mode, supported_modes = adapter_protocol(provider_type)
    api_mode = _clean_optional(profile.get("api_mode")).lower() or default_mode

    if not supported_modes:
        raise ValueError(f"未知 LLM provider type: {provider_type!r}")
    if api_mode not in supported_modes:
        raise ValueError(f"provider type {provider_type!r} 不支持 api_mode {api_mode!r}")
    return provider_type, api_mode


def _route(model: str, provider: str | None) -> Route:
    """Resolve the profile and protocol once; providers only ever see this."""
    _, profile = _provider_profile(model, provider)
    adapter, api_mode = _provider_protocol(model, profile)
    if adapter == "openai":
        profile = {**profile, "base_url": resolve_openai_base_url(profile.get("base_url"))}
    return Route(model, profile, adapter, api_mode, _infer_provider(model))


def provider_uses_responses_api(model: str, provider: str | None = None) -> bool:
    return _route(model, provider).api_mode == ApiMode.RESPONSES.value


def provider_supports_tool_search(model: str, provider: str | None = None) -> bool:
    route = _route(model, provider)
    if route.adapter == "openai":
        return openai_tool_search_supported(route)
    return any(marker in model.lower() for marker in get_adapter(route.adapter).tool_search_model_markers)


def normalize_reasoning_effort(
    model: str,
    provider: str | None,
    effort: str | None,
) -> str | None:
    """Map HyperBot's generic effort names to the route's vendor-native values."""
    if effort is None:
        return None
    return _normalize_reasoning_effort(_route(model, provider), effort)


def _normalize_reasoning_effort(route: Route, effort: str) -> str:
    vendor = vendor_for(route)
    return vendor.normalize_reasoning_effort(route, effort) if vendor else effort


def chat_reasoning_kwargs(
    model: str, provider: str | None, effort: str | None, *, responses_fallback: str | None = None,
) -> dict[str, str]:
    if not effort and not responses_fallback:
        return {}
    route = _route(model, provider)
    if route.api_mode == ApiMode.RESPONSES.value:
        effort = effort or responses_fallback
    elif route.api_mode != ApiMode.CHAT_COMPLETIONS.value:
        return {}
    return {"reasoning_effort": _normalize_reasoning_effort(route, effort)} if effort else {}


def provider_request_overrides(
    model: str,
    provider: str | None,
    *,
    user_id: str,
    cache_key: str,
) -> dict[str, Any]:
    """Return ``create_llm`` kwargs accepted only by the route's first-party API.

    Compatible proxies get nothing: they share an adapter with a vendor, not
    its request extensions.
    """
    route = _route(model, provider)
    vendor = vendor_for(route)
    return vendor.request_overrides(route, user_id=user_id, cache_key=cache_key) if vendor else {}


def native_web_search_support(
    model: str,
    provider: str | None = None,
) -> tuple[bool, str]:
    """Return whether provider-hosted web search is really available.

    Responses is only a wire protocol. It does not imply that an OpenAI-compatible
    endpoint implements OpenAI-hosted built-in tools. First-party vendors decide
    for their own routes; compatible Responses providers must opt in explicitly
    with ``native_web_search = true`` after their upstream capability is verified.
    """
    route = _route(model, provider)
    if route.profile.get("native_web_search") is False:
        return False, "native_web_search is disabled for this provider profile"
    vendor = vendor_for(route)
    verdict = vendor.native_web_search(route) if vendor else None
    if verdict is not None:
        return verdict

    if route.api_mode != ApiMode.RESPONSES.value:
        return False, f"api_mode={route.api_mode or 'unknown'} is not Responses"

    if route.profile.get("native_web_search") is True:
        return True, "Responses-compatible provider explicitly opted in"

    return (
        False,
        "Responses-compatible route has not opted in to provider-hosted web_search; "
        "set native_web_search=true only if the upstream endpoint implements it",
    )


def model_supports_native_web_search(model: str, provider: str | None = None) -> bool:
    """Backward-compatible boolean wrapper around :func:`native_web_search_support`."""
    supported, _reason = native_web_search_support(model, provider)
    return supported


def native_web_search_tools(model: str, provider: str | None = None) -> tuple[str, ...]:
    """Return the hosted tool types to inject when native search is enabled."""
    vendor = vendor_for(_route(model, provider))
    return vendor.native_search_tools if vendor else ("web_search",)


def native_web_search_middleware(
    model: str, provider: str | None, *, user_text: str, user_id: str, bot_user_id: str | None,
) -> AgentMiddleware:
    """Build the middleware for a route whose native search is enabled."""
    from utils.providers.web_search import NativeWebSearchMiddleware

    vendor = vendor_for(_route(model, provider))
    if vendor is not None:
        middleware = vendor.web_search_middleware(user_text=user_text, user_id=user_id, bot_user_id=bot_user_id)
        if middleware is not None:
            return middleware
    return NativeWebSearchMiddleware(vendor.native_search_tools if vendor else ("web_search",))


def provider_native_decisions(
    model: str, provider: str | None, *, max_retries: int, timeout: int,
) -> DecisionProvider | None:
    try:
        _, profile = _provider_profile(model, provider)
    except ValueError:
        return None
    if not profile.get("native_decisions"):
        return None
    route = _route(model, provider)
    vendor = vendor_for(route)
    backend = vendor.decision_backend(route, max_retries=max_retries, timeout=timeout) if vendor else None
    if backend is None:
        raise ValueError("native_decisions 需要支持该功能的官方 provider")
    return backend


def get_langchain_model_profile(model: str, provider_type: str) -> ModelProfile | None:
    """Translate a catalog model card into LangChain's runtime model profile."""
    parts = model.split("/")
    candidates = ["/".join(parts[index:]) for index in range(len(parts))]
    card = next((match for candidate in candidates if (match := get_model(provider_type, candidate))), None)
    if card is None:
        catalog = load_catalog().models
        for candidate in candidates:
            matches = [item for item in catalog if item.id.lower() == candidate.lower()]
            if len(matches) == 1:
                card = matches[0]
                break
    if card is None:
        return None

    inputs = set(card.capabilities.input)
    outputs = set(card.capabilities.output)
    features = set(card.capabilities.features)
    profile: ModelProfile = {
        "name": card.display_name,
        "status": card.status.value,
        "text_inputs": ModelInput.TEXT in inputs,
        "image_inputs": ModelInput.IMAGE in inputs,
        "image_url_inputs": ModelInput.IMAGE in inputs,
        "pdf_inputs": ModelInput.FILE in inputs,
        "audio_inputs": ModelInput.AUDIO in inputs,
        "video_inputs": ModelInput.VIDEO in inputs,
        "text_outputs": ModelOutput.TEXT in outputs,
        "image_outputs": ModelOutput.IMAGE in outputs,
        "audio_outputs": ModelOutput.AUDIO in outputs,
        "video_outputs": ModelOutput.VIDEO in outputs,
        "reasoning_output": ModelFeature.REASONING in features,
        "tool_calling": ModelFeature.TOOL_CALLING in features,
        "tool_choice": ModelFeature.TOOL_CALLING in features,
        "tool_call_streaming": ModelFeature.PARALLEL_TOOL_CALLING in features,
        "structured_output": ModelFeature.STRUCTURED_OUTPUT in features,
        "attachment": ModelInput.FILE in inputs,
    }
    max_input_tokens = card.max_input_tokens if card.max_input_tokens is not None else card.context_window
    if max_input_tokens is not None:
        profile["max_input_tokens"] = max_input_tokens
    if card.max_output_tokens is not None:
        profile["max_output_tokens"] = card.max_output_tokens
    if card.released_at is not None:
        profile["release_date"] = card.released_at
    return profile


def provider_decision_extra_body(model: str, provider: str | None = None) -> dict[str, Any]:
    """Return the provider's decision-only request payload.

    只作用于 decision 轻量调用，不进入主 Agent；用于传递供应商的推理开关等
    ``extra_body`` 参数，例如 DeepSeek 的 ``{"thinking": {"type": "disabled"}}``。
    """
    try:
        _, profile = _provider_profile(model, provider)
    except ValueError:
        return {}
    payload = profile.get("decision_extra_body")
    if not isinstance(payload, dict) or not payload:
        payload = profile.get("signal_extra_body")
    return dict(payload) if isinstance(payload, dict) else {}


def provider_signal_extra_body(model: str, provider: str | None = None) -> dict[str, Any]:
    """Deprecated alias for :func:`provider_decision_extra_body`."""

    return provider_decision_extra_body(model, provider)


def structured_output_options(
    model: str,
    provider: str | None,
    llm: BaseChatModel,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Select schema enforcement without silently changing API endpoints.

    The first-party vendor can override the strategy selected by the adapter.
    """
    route = _route(model, provider)
    selected = method or route.profile.get("structured_output_method", "auto")
    if selected not in {"auto", "json_schema", "function_calling", "json_mode", "text_json"}:
        raise ValueError("不支持的 structured_output_method")
    if selected == "auto":
        capabilities = getattr(llm, "profile", None) or {}
        vendor = vendor_for(route)
        selected = (vendor.structured_output_method(route, capabilities) if vendor else None) or get_adapter(
            route.adapter
        ).structured_output_method(route, capabilities)
    return get_adapter(route.adapter).structured_output_options(selected)


def create_llm(model: str, provider: str | None = None, **kwargs) -> BaseChatModel:
    """根据供应商 profile 路由模型，并过滤底层 SDK 不支持的参数。

    ``provider`` 指向 ``[providers.<name>]``；未指定时按模型名称推断官方供应商。
    profile 的 ``type`` 决定底层 LangChain 适配器，``api_mode`` 决定接口协议。
    """
    if "endpoint" in kwargs:
        raise TypeError("create_llm() 不再接受 endpoint，请通过 provider 选择供应商 profile")
    if "use_responses_api" in kwargs:
        raise TypeError("create_llm() 不再接受 use_responses_api，请在供应商 profile 中配置 api_mode")
    route = _route(model, provider)
    catalog_profile = None if "profile" in kwargs else get_langchain_model_profile(model, route.model_family)
    return get_adapter(route.adapter).create_chat_model(
        route, kwargs, catalog_profile=catalog_profile, vendor=vendor_for(route),
    )

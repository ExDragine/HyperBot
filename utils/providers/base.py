"""Provider protocol shared by every model vendor HyperBot talks to.

Two roles are kept apart because they vary independently:

* A :class:`ChatAdapter` owns one ``[providers.<name>].type``: the LangChain
  class, the constructor parameters it accepts and the API modes it speaks.
* A :class:`Vendor` owns one first-party service: how to recognise its
  official endpoint and which request extensions are safe to send there.

DeepSeek shows why: it is one vendor reachable through three adapters
(``deepseek``, ``openai`` Responses and ``anthropic`` Messages), while the
``openai`` adapter also serves arbitrary compatible proxies that must not
receive OpenAI-only fields.

This module imports no vendor SDK, so configuration code may use its tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol
from urllib.parse import urlsplit

from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models import BaseChatModel, ModelProfile

    from utils.decision.models import DecisionProvider

CHAT_COMPLETIONS = "chat_completions"
RESPONSES = "responses"
MESSAGES = "messages"
GENERATE_CONTENT = "generate_content"

API_MODES = frozenset({CHAT_COMPLETIONS, RESPONSES, MESSAGES, GENERATE_CONTENT})


def clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class Route:
    """One resolved model route; the only input providers receive.

    ``adapter`` and ``api_mode`` are resolved once by the factory, so providers
    never re-derive them from the raw profile. ``model_family`` is the adapter
    type the model id alone would select.
    """

    model: str
    profile: Mapping[str, Any]
    adapter: str
    api_mode: str
    model_family: str

    @property
    def base_url(self) -> str:
        return clean(self.profile.get("base_url"))


def targets_https_api(
    route: Route,
    hostname: str,
    *,
    paths: frozenset[str] = frozenset({"", "/v1"}),
    allow_default: bool,
) -> bool:
    """Return whether the route's base URL is exactly a first-party endpoint.

    ``allow_default`` states whether an empty base URL means that endpoint,
    which is true only when the SDK's own default points there.
    """
    base_url = route.base_url
    if not base_url:
        return allow_default

    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == "https"
        and (parsed.hostname or "").lower() == hostname
        and parsed.path.rstrip("/") in paths
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
    )


class ChatAdapter(Protocol):
    """Builds the LangChain chat model for one provider ``type``."""

    type: ClassVar[str]
    api_modes: ClassVar[frozenset[str]]
    default_api_mode: ClassVar[str]
    tool_search_model_markers: ClassVar[tuple[str, ...]]
    model_prefixes: ClassVar[tuple[str, ...]]
    default_profiles: ClassVar[Mapping[str, Mapping[str, Any]]]

    def create_chat_model(
        self,
        route: Route,
        kwargs: Mapping[str, Any],
        *,
        catalog_profile: ModelProfile | None = None,
        vendor: Vendor | None = None,
    ) -> BaseChatModel:
        """Construct the model, dropping parameters the SDK does not accept."""
        ...

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str:
        """Return the schema strategy when no vendor has a stronger opinion."""
        ...

    def structured_output_options(self, method: str) -> dict[str, Any]:
        """Return SDK-specific options for the selected schema strategy."""
        ...


class Vendor(Protocol):
    """Request semantics of one first-party service."""

    name: ClassVar[str]
    native_search_tools: ClassVar[tuple[str, ...]]
    profile_fields: ClassVar[tuple[str, ...]]

    def serves(self, route: Route) -> bool:
        """Return whether the route reaches this vendor's official endpoint."""
        ...

    def request_overrides(self, route: Route, *, user_id: str, cache_key: str) -> dict[str, Any]:
        """Return ``create_llm`` kwargs that only this vendor's API accepts."""
        ...

    def normalize_reasoning_effort(self, route: Route, effort: str) -> str:
        """Map HyperBot's generic effort names to the vendor's values."""
        ...

    def native_web_search(self, route: Route) -> tuple[bool, str] | None:
        """Return ``(supported, reason)``, or ``None`` to use the generic rule."""
        ...

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str | None:
        """Return a schema strategy, or ``None`` to defer to the adapter."""
        ...

    def prepare_chat_model(self, route: Route, model_class: Any, params: dict[str, Any]) -> Any:
        """Apply constructor extensions and return the SDK class to instantiate."""
        ...

    def web_search_middleware(
        self, *, user_text: str, user_id: str, bot_user_id: str | None,
    ) -> AgentMiddleware | None:
        """Supply a vendor-specific middleware when generic hosted search is insufficient."""
        ...

    def standalone_web_search_hint(self, tool_names: set[str]) -> str:
        ...

    def decision_backend(self, route: Route, *, max_retries: int, timeout: int) -> DecisionProvider | None:
        ...


class BaseChatAdapter:
    """Table-driven :class:`ChatAdapter`; subclasses state only what differs."""

    type: ClassVar[str]
    api_modes: ClassVar[frozenset[str]]
    default_api_mode: ClassVar[str]
    tool_search_model_markers: ClassVar[tuple[str, ...]] = ()
    model_prefixes: ClassVar[tuple[str, ...]] = ()
    #: Built-in ``[providers.<name>]`` tables this provider ships with.
    default_profiles: ClassVar[Mapping[str, Mapping[str, Any]]] = {}
    valid_kwargs: ClassVar[frozenset[str]]
    kwarg_names: ClassVar[Mapping[str, str]] = {}
    api_key_field: ClassVar[str]
    base_url_field: ClassVar[str]

    def model_class(self) -> Any:
        """Return the SDK class, read at call time so tests can replace it."""
        raise NotImplementedError

    def prepare(self, route: Route, params: dict[str, Any]) -> Any:
        """Adjust constructor parameters in place; the result goes to :meth:`finalize`."""
        return None

    def finalize(self, instance: BaseChatModel, state: Any) -> None:
        """Apply settings that cannot go through the constructor."""

    def create_chat_model(
        self,
        route: Route,
        kwargs: Mapping[str, Any],
        *,
        catalog_profile: ModelProfile | None = None,
        vendor: Vendor | None = None,
    ) -> BaseChatModel:
        params: dict[str, Any] = {
            self.kwarg_names.get(key, key): value
            for key, value in kwargs.items()
            if key in self.valid_kwargs or key == "tags"
        }
        state = self.prepare(route, params)
        if "profile" not in params and catalog_profile is not None:
            params["profile"] = catalog_profile
        if route.base_url:
            params[self.base_url_field] = route.base_url

        api_key = SecretStr(clean(route.profile.get("api_key")))
        model_class = self.model_class()
        if vendor is not None:
            model_class = vendor.prepare_chat_model(route, model_class, params)
        instance = model_class(**{self.api_key_field: api_key, "model": route.model, **params})
        self.finalize(instance, state)
        return instance

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str:
        return "function_calling" if capabilities.get("tool_calling") is True else "json_mode"

    def structured_output_options(self, method: str) -> dict[str, Any]:
        return {"method": method}


class BaseVendor:
    """Neutral :class:`Vendor`; subclasses override what their API extends."""

    name: ClassVar[str]
    native_search_tools: ClassVar[tuple[str, ...]] = ("web_search",)
    #: Extra table-valued profile keys only this vendor's API understands.
    profile_fields: ClassVar[tuple[str, ...]] = ()

    def serves(self, route: Route) -> bool:
        raise NotImplementedError

    def request_overrides(self, route: Route, *, user_id: str, cache_key: str) -> dict[str, Any]:
        return {}

    def normalize_reasoning_effort(self, route: Route, effort: str) -> str:
        return effort

    def native_web_search(self, route: Route) -> tuple[bool, str] | None:
        return None

    def structured_output_method(self, route: Route, capabilities: Mapping[str, Any]) -> str | None:
        return None

    def prepare_chat_model(self, route: Route, model_class: Any, params: dict[str, Any]) -> Any:
        return model_class

    def web_search_middleware(
        self, *, user_text: str, user_id: str, bot_user_id: str | None,
    ) -> AgentMiddleware | None:
        return None

    def standalone_web_search_hint(self, tool_names: set[str]) -> str:
        return ""

    def decision_backend(self, route: Route, *, max_retries: int, timeout: int) -> DecisionProvider | None:
        return None

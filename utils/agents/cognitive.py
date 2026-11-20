"""Main HyperBot Deep Agent composition and execution."""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Literal, NotRequired, cast

from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    MemoryMiddleware,
    create_deep_agent,
)
from deepagents.graph import DeepAgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    FilesystemFileSearchMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    ProviderToolSearchMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command
from wcmatch import glob

import tools as _tool_registry
from plugins.acp.subagent import build_acp_subagents
from utils.agent_context import FrontierRuntimeContext
from utils.agent_protocol import AgentArtifact, ConversationRef, Participant
from utils.configs import EnvConfig
from utils.ens_gate import _ens_prefix
from utils.harness_profiles import register_frontier_harness_profiles
from utils.llm_factory import (
    create_llm,
    native_web_search_middleware,
    native_web_search_support,
    normalize_reasoning_effort,
    provider_request_overrides,
    provider_uses_responses_api,
)
from utils.media import detect_mime_type, inline_media_bytes, media_block_kind
from utils.providers import model_supports_tool_search, standalone_web_search_hint

from .capture import detect_browser_capture_intent
from .execution import current_run_id, managed_agent_turn
from .inputs import filter_messages_for_model_capabilities
from .media_inputs import MediaInputMiddleware
from .progress import (
    ProgressEvent,
    ProgressReporter,
    collect_progress,
    emit_progress,
    finish_progress_collection,
)
from .prompts import JobPrompt, PromptProfile, build_workspace_soul_prompt
from .prompts import load_system_prompt as compose_system_prompt
from .runtime import agent_thread_id, conversation_workspace_key, normalize_workspace_key
from .session_context import QQ_TRIGGER_ID_PREFIX, SessionHistoryMiddleware, history_budget, recent_complete_turns
from .skill_tools import TurnSkillsMiddleware
from .subagents import build_document_subagent
from .tool_errors import WEB_SEARCH_TOOL_NAMES, tool_error_middleware
from .workspace import SKILLS_BACKEND_PATH, build_agent_backend

register_frontier_harness_profiles()

logger = logging.getLogger(__name__)


_ARTIFACT_KINDS = frozenset({"image", "audio", "video", "file"})

def _normalized_artifact_kind(value: object) -> str | None:
    kind = str(value or "").lower()
    if kind == "voice":
        kind = "audio"
    return kind if kind in _ARTIFACT_KINDS else None


def _mapping_source(block: Mapping[str, object]) -> tuple[str | None, str | None]:
    """Read a neutral URL/path source from standard or legacy content blocks."""

    block_type = str(block.get("type") or "")
    value: object = block.get("url")
    if value is None and block_type in {"image_url", "audio_url", "video_url", "input_image"}:
        value = block.get(block_type)
    if isinstance(value, Mapping):
        value = value.get("url")
    url = str(value) if isinstance(value, str) and value else None
    path_value = block.get("path")
    path = str(path_value) if isinstance(path_value, str) and path_value else None
    if url and url.startswith("data:"):
        url = None
    return url, path


def _artifact_from_mapping(block: Mapping[str, object]) -> AgentArtifact | None:
    kind = media_block_kind(block) or _normalized_artifact_kind(block.get("kind"))
    if kind is None:
        return None
    decoded = inline_media_bytes(block)
    raw = bytes(decoded[0]) if decoded and decoded[0] else None
    if raw is None:
        raw_value = block.get("raw", block.get("data"))
        if isinstance(raw_value, (bytes, bytearray)) and raw_value:
            raw = bytes(raw_value)
    decoded_mime = decoded[1] if decoded else None
    url, path = _mapping_source(block)
    name_value = block.get("name") or block.get("filename")
    name = str(name_value) if isinstance(name_value, str) and name_value else None
    declared_mime = block.get("mime_type") or block.get("mimetype") or decoded_mime
    if raw is None and not url and not path:
        return None
    mime_type = detect_mime_type(
        raw or b"",
        kind=kind,
        declared_mime=str(declared_mime) if declared_mime else None,
        file_name=name,
    )
    return AgentArtifact(
        kind=kind,
        data=raw or b"",
        mime_type=mime_type,
        name=name,
        url=url,
        path=path,
    )


def _native_media_artifact(block: object) -> AgentArtifact | None:
    """Convert a model-native media block into a platform-neutral artifact."""

    if not isinstance(block, Mapping):
        return None
    return _artifact_from_mapping(block)


def _segment_value(segment: object, *names: str) -> object:
    if isinstance(segment, Mapping):
        for name in names:
            if name in segment:
                return segment[name]
        return None
    for name in names:
        value = getattr(segment, name, None)
        if value is not None:
            return value
    return None


def _artifact_from_segment(segment: object) -> AgentArtifact | None:
    """Convert a tool artifact segment without importing a platform SDK."""

    if isinstance(segment, AgentArtifact):
        return segment
    if isinstance(segment, Mapping):
        return _artifact_from_mapping(segment)

    kind = _normalized_artifact_kind(_segment_value(segment, "kind", "type"))
    if kind is None:
        return None
    raw_value = _segment_value(segment, "raw", "data")
    raw = bytes(raw_value) if isinstance(raw_value, (bytes, bytearray)) and raw_value else None
    url_value = _segment_value(segment, "url")
    path_value = _segment_value(segment, "path")
    url = str(url_value) if isinstance(url_value, str) and url_value else None
    path = os.fsdecode(path_value) if isinstance(path_value, (str, os.PathLike)) and path_value else None
    if url and url.startswith("data:"):
        decoded = inline_media_bytes({"type": kind, "url": url})
        if decoded and decoded[0]:
            raw = bytes(decoded[0])
            url = None
            declared_mime = decoded[1]
        else:
            declared_mime = None
    else:
        declared_mime = _segment_value(segment, "mime_type", "mimetype")
    name_value = _segment_value(segment, "name", "filename")
    name = str(name_value) if isinstance(name_value, str) and name_value else None
    if raw is None and not url and not path:
        return None
    mime_type = detect_mime_type(
        raw or b"",
        kind=kind,
        declared_mime=str(declared_mime) if declared_mime else None,
        file_name=name,
    )
    return AgentArtifact(
        kind=kind,
        data=raw or b"",
        mime_type=mime_type,
        name=name,
        url=url,
        path=path,
    )


def _iter_tool_artifact_segments(value: object):
    """Yield media-like values from a legacy segment-shaped artifact.

    This intentionally relies only on the small segment protocol (``type``,
    ``raw``, ``url`` and friends), so a tool's historical segment container
    can cross the boundary without making Agent Core import or return a
    platform type.
    """

    if isinstance(value, AgentArtifact) or isinstance(value, Mapping):
        yield value
        return
    if isinstance(value, (str, bytes, bytearray)):
        return
    if isinstance(value, Iterable):
        for item in value:
            yield from _iter_tool_artifact_segments(item)
        return
    yield value


class FrontierAgentState(DeepAgentState):
    """Mutable graph state; identity fields remain as a compatibility bridge for tools."""

    user_id: str
    group_id: int | None
    image_inputs: list[bytes]
    audio_inputs: list[bytes]
    video_inputs: list[bytes]
    suppress_reply: NotRequired[bool]


_DEFAULT_TOOL_RUNTIME = cast(ToolRuntime[FrontierRuntimeContext, FrontierAgentState], None)


@tool
def skip_reply(
    reason: Literal[
        "not_addressed",
        "conversation_complete",
        "duplicate_or_stale",
        "would_interrupt",
    ],
    runtime: ToolRuntime[FrontierRuntimeContext, FrontierAgentState] = _DEFAULT_TOOL_RUNTIME,
) -> Command:
    """在无需打扰群聊时结束本轮且不发送任何回复。

    仅当消息没有明确询问或点名你、话题已经自然结束、内容重复/过时，或回复会打断
    他人对话时使用。应在调用任何发送或写入工具之前选择；不要用它拒绝正常请求。
    私聊和明确点名场景不会提供此工具。
    """
    return Command(
        update={
            "suppress_reply": True,
            "messages": [
                ToolMessage(
                    content=f"本轮不回复：{reason}",
                    tool_call_id=getattr(runtime, "tool_call_id", None) or "skip_reply",
                )
            ],
        }
    )


_PROMPT_CACHE_NAMESPACE = "frontier-agent-v1"


def _named_item_key(item: Any) -> tuple[str, str]:
    """Return a deterministic sort key for tools and subagent specs."""
    name = item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
    normalized = str(name)
    return normalized.casefold(), normalized


def _stable_named_items(items) -> list:
    """Keep provider-visible tool/subagent arrays stable across requests."""
    return sorted(items, key=_named_item_key)


def _merge_named_items(*groups) -> list:
    """Merge tool-like values by public name while preserving stable order."""

    merged = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for item in group:
            key = _named_item_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return _stable_named_items(merged)


def _prompt_cache_key(*, model: str, workspace_key: str, access_profile: str) -> str:
    """Build a short pseudonymous OpenAI cache-routing key for one workspace."""
    material = "\0".join(
        (_PROMPT_CACHE_NAMESPACE, "openai-prompt-cache", model, access_profile, workspace_key)
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:40]
    return f"{_PROMPT_CACHE_NAMESPACE}:{digest}"


def _provider_user_id(*, workspace_key: str, access_profile: str) -> str:
    """Build a provider-safe pseudonym without exposing QQ or ACP identity."""
    material = "\0".join(
        (_PROMPT_CACHE_NAMESPACE, "provider-user", access_profile, workspace_key)
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:40]
    return f"{_PROMPT_CACHE_NAMESPACE}_user_{digest}"


def _provider_request_overrides(
    *,
    model: str,
    provider: str | None,
    workspace_key: str,
    access_profile: str,
) -> dict[str, Any]:
    """Return request fields supported by the route's first-party API only."""
    return provider_request_overrides(
        model,
        provider,
        user_id=_provider_user_id(workspace_key=workspace_key, access_profile=access_profile),
        cache_key=_prompt_cache_key(model=model, workspace_key=workspace_key, access_profile=access_profile),
    )


class SilentReplyMiddleware(AgentMiddleware):
    """End the graph immediately after ``skip_reply`` updates agent state."""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: FrontierAgentState, runtime) -> dict[str, Any] | None:
        del runtime
        if state.get("suppress_reply"):
            return {"jump_to": "end"}
        return None


class _LazyToolRegistry:
    def __getattr__(self, name):
        return getattr(_tool_registry.agent_tools, name)


agent_tools = _LazyToolRegistry()


class FrontierCognitive:
    def __init__(self):
        self._component_revision = None

    def __getattr__(self, name):
        if name in {"tools", "document_subagent"} and "_component_revision" in self.__dict__:
            self._build_components()
            return self.__dict__[name]
        raise AttributeError(name)

    def _build_components(self):
        self.tools = _stable_named_items(agent_tools.direct_tools)
        self.document_subagent = build_document_subagent()
        self._component_revision = EnvConfig.REVISION

    def _tools_for_capabilities(self, capabilities: frozenset[str]) -> list:
        """Select a tool snapshot for an explicit platform capability set.

        Empty capabilities preserve the pre-separation behavior used by ACP,
        scheduled tasks, and the existing QQ event path.  The registry method is
        optional so injected test registries and staged callers that only expose
        ``direct_tools`` continue to work.
        """

        legacy_tools = getattr(self, "tools", [])
        if not capabilities:
            return legacy_tools
        direct_for = getattr(agent_tools, "direct_tools_for", None)
        if callable(direct_for):
            return _stable_named_items(direct_for(capabilities))
        return legacy_tools

    async def _prepare_components(self):
        if "_component_revision" not in self.__dict__:
            return
        initialize = getattr(agent_tools, "initialize", None)
        if initialize is not None:
            await initialize()
        if self._component_revision != EnvConfig.REVISION:
            self._build_components()
        else:
            self.tools = _stable_named_items(agent_tools.direct_tools)

    @staticmethod
    def load_system_prompt(
        group_id: int | None = None,
        *,
        profile: PromptProfile = "qq",
    ) -> str:
        return compose_system_prompt(group_id, profile=profile)

    @staticmethod
    async def extract_artifacts(response) -> list[AgentArtifact]:
        """Extract neutral tool artifacts and native media blocks.

        Tool implementations still return their historical segment containers
        during the migration.  They are inspected structurally here and
        converted into :class:`AgentArtifact`; no platform message object is
        created or returned by Agent Core.
        """
        if not response or not isinstance(response, dict):
            logger.warning("⚠️ extract_artifacts: response 为空或不是字典类型")
            return []

        artifacts: list[AgentArtifact] = []
        response_messages = response.get("messages", [])
        for message in response_messages:
            if getattr(message, "type", None) != "tool":
                continue
            raw_artifact = getattr(message, "artifact", None)
            if raw_artifact is None:
                continue
            tool_name = getattr(message, "name", "unknown")
            extracted = [
                artifact
                for segment in _iter_tool_artifact_segments(raw_artifact)
                if (artifact := _artifact_from_segment(segment)) is not None
            ]
            if extracted:
                artifacts.extend(extracted)
                logger.info("📤 提取中性媒体工件: %s - 数量: %s", tool_name, len(extracted))

        ai_messages = [message for message in response_messages if getattr(message, "type", None) == "ai"]
        if ai_messages:
            final_ai = ai_messages[-1]
            blocks = getattr(final_ai, "content_blocks", None)
            if blocks is None:
                blocks = getattr(final_ai, "content", None)
            if isinstance(blocks, (list, tuple)):
                artifacts.extend(
                    artifact
                    for block in blocks
                    if (artifact := _native_media_artifact(block)) is not None
                )

        logger.info("📨 总共提取到 %s 个中性媒体工件", len(artifacts))
        return artifacts

    @staticmethod
    async def extract_uni_messages(response) -> list[AgentArtifact]:
        """Compatibility spelling for the pre-separation extraction entry point."""

        return await FrontierCognitive.extract_artifacts(response)

    @managed_agent_turn
    async def chat_agent(  # noqa: C901
        self,
        messages,
        user_id,
        user_name,
        capability: str = "none",
        group_id: int | None = None,
        image_inputs: list[bytes] | None = None,
        audio_inputs: list[bytes] | None = None,
        video_inputs: list[bytes] | None = None,
        thread_id_override: uuid.UUID | str | None = None,
        group_member_role: str | None = None,
        progress_reporter: ProgressReporter | None = None,
        user_text: str | None = None,
        access_profile: Literal["frontier", "acp"] = "frontier",
        enable_acp_subagents: bool = True,
        allow_silent_reply: bool = False,
        conversation: ConversationRef | None = None,
        principal: Participant | None = None,
        capabilities: frozenset[str] = frozenset(),
        workspace_key_override: str | None = None,
        tool_overrides: tuple[object, ...] = (),
        pinned_skills: tuple[str, ...] = (),
        include_messages: bool = False,
        allow_local_tools: bool = True,
        allow_native_web_search: bool = True,
        job_prompt: JobPrompt | None = None,
    ):
        allowed_capture_tools = (
            await detect_browser_capture_intent(user_text)
            if access_profile == "frontier" and allow_local_tools
            else set()
        )
        if allow_local_tools:
            await self._prepare_components()
        selected_tools = self._tools_for_capabilities(capabilities) if allow_local_tools else []
        if allow_local_tools and tool_overrides:
            selected_tools = _merge_named_items(selected_tools, tool_overrides)
        workspace_key = normalize_workspace_key(
            workspace_key_override or conversation_workspace_key(user_id, group_id)
        )
        uses_responses_api = provider_uses_responses_api(
            EnvConfig.ADVAN_MODEL,
            EnvConfig.ADVAN_MODEL_PROVIDER,
        )
        model_kwargs: dict = {
            "model": EnvConfig.ADVAN_MODEL,
            "streaming": False,
            "max_retries": 0,
            "timeout": EnvConfig.AGENT_LLM_TIMEOUT_SECONDS,
            "provider": EnvConfig.ADVAN_MODEL_PROVIDER,
            "tags": ["frontier:main"],
        }
        if uses_responses_api:
            reasoning_effort = normalize_reasoning_effort(
                EnvConfig.ADVAN_MODEL,
                EnvConfig.ADVAN_MODEL_PROVIDER,
                capability,
            )
            if reasoning_effort is not None:
                model_kwargs["reasoning_effort"] = reasoning_effort
            model_kwargs["verbosity"] = "low"
        model_kwargs.update(
            _provider_request_overrides(
                model=EnvConfig.ADVAN_MODEL,
                provider=EnvConfig.ADVAN_MODEL_PROVIDER,
                workspace_key=workspace_key,
                access_profile=access_profile,
            )
        )
        model = create_llm(**model_kwargs)
        working_dir = getattr(self, "working_dir", os.path.join(os.getcwd(), "cache", "sandbox"))
        thread_id = thread_id_override or agent_thread_id(user_id, group_id)
        if not isinstance(thread_id, uuid.UUID):
            thread_id = uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=str(thread_id))
        qq_group = conversation is not None and conversation.platform == "qq" and group_id is not None
        backend = build_agent_backend(working_dir, workspace_key, **({"qq_group": True} if qq_group else {}))
        workspace_dir = os.path.join(working_dir, "workspaces", workspace_key)
        memory_dir = os.path.join(working_dir, "memory", workspace_key)
        soul_path = f"/memory/{workspace_key}/SOUL.md"
        profile: PromptProfile
        if access_profile == "acp":
            profile = "acp"
        elif job_prompt is not None:
            profile = job_prompt
        else:
            profile = "qq" if conversation is None or conversation.platform == "qq" else "chat"
        system_prompt = self.load_system_prompt(group_id) if profile == "qq" else self.load_system_prompt(profile=profile)

        effective_tools = []
        skill_tools = []
        if access_profile == "frontier" and allow_local_tools:
            effective_tools = _stable_named_items(selected_tools)
            if profile == "scheduled-task":
                effective_tools = [tool for tool in effective_tools if tool.name != "send_plain_text"]
            if allow_silent_reply:
                effective_tools = _stable_named_items([*effective_tools, skip_reply])
            direct_tool_names = {tool.name for tool in effective_tools}
            skill_tools = _merge_named_items(
                tool for tool in getattr(agent_tools, "skill_tools", ()) if tool.name not in direct_tool_names
            )
        if allowed_capture_tools:
            for restricted_tool in _stable_named_items(agent_tools.restricted_tools):
                if restricted_tool.name in allowed_capture_tools:
                    effective_tools.append(restricted_tool)
                    logger.info(f"用户明确请求浏览器捕获工具，已暴露: {restricted_tool.name}")
        else:
            logger.debug("用户未请求截图/录屏，restricted 工具未暴露")

        supported, search_reason = native_web_search_support(
            EnvConfig.ADVAN_MODEL, EnvConfig.ADVAN_MODEL_PROVIDER,
        )
        search_middleware = None
        if allow_native_web_search and supported:
            bot_user_id = conversation.account_id if conversation and conversation.platform == "qq" else None
            search_middleware = native_web_search_middleware(
                EnvConfig.ADVAN_MODEL, EnvConfig.ADVAN_MODEL_PROVIDER,
                user_text=user_text or "", user_id=str(user_id), bot_user_id=bot_user_id,
            )
            logger.info("主 Agent 已挂载服务端联网工具: %s", search_reason)
        else:
            system_prompt += standalone_web_search_hint({tool.name for tool in effective_tools})
            if uses_responses_api and allow_native_web_search:
                logger.warning("主 Agent 使用 Responses API，但未挂载服务端原生联网工具: %s", search_reason)
            else:
                logger.debug("当前请求未挂载服务端原生联网工具: %s", search_reason)
        subagents = []
        if allow_local_tools:
            if document_subagent := getattr(self, "document_subagent", None):
                subagents.append(document_subagent)
        if allow_local_tools and access_profile == "frontier" and enable_acp_subagents:
            subagents.extend(_stable_named_items(build_acp_subagents()))

        messages = filter_messages_for_model_capabilities(
            messages,
            EnvConfig.ADVAN_MODEL,
            role="advanced",
        )
        current_id = messages[-1].get("id", "") if messages and isinstance(messages[-1], dict) else ""
        has_qq_history = isinstance(current_id, str) and current_id.startswith(QQ_TRIGGER_ID_PREFIX)
        history_ids = set()
        if has_qq_history:
            messages = [
                *recent_complete_turns(messages[:-1], history_budget(EnvConfig.SESSIONS, model.profile)),
                messages[-1],
            ]
            history_ids = {message["id"] for message in messages[:-1] if isinstance(message, dict) and "id" in message}
        blocked_virtual_paths: list[str] = []
        filesystem_permissions = [
            FilesystemPermission(
                operations=["write"],
                paths=[SKILLS_BACKEND_PATH, f"{SKILLS_BACKEND_PATH}/**"],
                mode="deny",
            )
        ]
        if qq_group:
            from utils.database import get_engine
            from utils.database.qq_lifecycle import QqLifecycleStore

            recalled_paths = []
            if has_qq_history:
                recalled_paths = await QqLifecycleStore(get_engine()).recalled_attachment_paths(int(conversation.account_id), group_id)
            blocked_virtual_paths = ["/conversation_history/", "/.qq-turn/", *recalled_paths]
            filesystem_permissions.append(FilesystemPermission(
                operations=["read", "write"], mode="deny",
                paths=["/conversation_history", "/conversation_history/**", *(glob.escape(path) for path in recalled_paths)],
            ))
        available_tools = [*effective_tools, *skill_tools]
        is_read_only_tool = getattr(agent_tools, "is_read_only_tool", None) if available_tools else None
        read_only_tools = [
            tool for tool in available_tools
            if callable(is_read_only_tool) and is_read_only_tool(tool)
        ]
        middleware: list[Any] = [
            PIIMiddleware(
                "api_key",
                detector=r"sk-[a-zA-Z0-9]{32}",
                strategy="mask",
            ),
            tool_error_middleware(read_only_tools=[
                *read_only_tools,
                *WEB_SEARCH_TOOL_NAMES, "ls", "glob", "grep", "read_file",
                "get_recent_conversation", "search_messages", "get_history_messages",
            ]),
            ModelCallLimitMiddleware(run_limit=EnvConfig.AGENT_MODEL_CALL_LIMIT, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=EnvConfig.AGENT_TOOL_CALL_LIMIT, exit_behavior="error"),
            ModelRetryMiddleware(on_failure="error"),
            FilesystemMiddleware(
                backend=backend,
                offload_binary_content=True,
                _permissions=filesystem_permissions,
            ),
            FilesystemFileSearchMiddleware(root_path=workspace_dir),
        ]
        if profile != "daily-news":
            middleware.append(MemoryMiddleware(
                backend=backend,
                sources=[soul_path],
                add_cache_control=True,
                system_prompt=build_workspace_soul_prompt(soul_path),
            ))
        if allow_silent_reply and access_profile == "frontier":
            middleware.append(SilentReplyMiddleware())
        if allow_local_tools and model_supports_tool_search(EnvConfig.ADVAN_MODEL):
            middleware.append(
                ProviderToolSearchMiddleware(
                    searchable_tools=[tool for tool in effective_tools if tool is not skip_reply]
                )
            )
        if has_qq_history:
            middleware.append(SessionHistoryMiddleware(current_id, EnvConfig.SESSIONS, model.profile))
        if search_middleware is not None:
            middleware.append(search_middleware)
        middleware.append(MediaInputMiddleware(
            EnvConfig.ADVAN_MODEL, role="advanced",
            max_images=EnvConfig.MAX_INLINE_IMAGES, max_bytes=EnvConfig.MAX_INLINE_MEDIA_BYTES,
        ))
        if skill_tools:
            middleware.append(TurnSkillsMiddleware(backend=backend, sources=[SKILLS_BACKEND_PATH], tools=skill_tools))
        agent = create_deep_agent(
            name=EnvConfig.BOT_NAME,
            model=model,
            system_prompt=system_prompt,
            tools=effective_tools,
            subagents=subagents,
            middleware=middleware,
            skills=[SKILLS_BACKEND_PATH],
            memory=None if profile == "daily-news" else [soul_path],
            permissions=filesystem_permissions,
            backend=backend,
            state_schema=FrontierAgentState,
            context_schema=FrontierRuntimeContext,
            debug=EnvConfig.AGENT_DEBUG_MODE,
        )
        start_time = time.time()
        logger.info(f"Agent烧烤中~🍖 思考等级: {capability} 用户: {user_name} (ID: {user_id})")
        config: RunnableConfig = {
            "metadata": {"frontier_run_id": current_run_id.get()},
            "configurable": {
                "thread_id": thread_id,
                **({"blocked_virtual_paths": blocked_virtual_paths} if blocked_virtual_paths else {}),
                "user_id": user_id,
                "group_id": group_id,
                "group_member_role": group_member_role,
                **({"bot_user_id": conversation.account_id} if conversation is not None and conversation.platform == "qq" else {}),
                "workspace_dir": workspace_dir,
                "virtual_roots": {
                    f"/memory/{workspace_key}/": memory_dir,
                    "/": workspace_dir,
                },
            }
        }
        runtime_context = FrontierRuntimeContext(
            user_id=str(user_id),
            group_id=group_id,
            group_member_role=group_member_role,
            workspace_dir=workspace_dir,
            conversation=conversation,
            principal=principal,
            capabilities=capabilities,
        )
        input_data: Any = {
            "messages": messages,
            "user_id": user_id,
            "group_id": group_id,
            "image_inputs": image_inputs or [],
            "audio_inputs": audio_inputs or [],
            "video_inputs": video_inputs or [],
            "suppress_reply": False,
        }
        skill_names = list(pinned_skills)
        if skill_tools and _ens_prefix.get() == "vep":
            skill_names.append("ens-weather")
        if allow_local_tools and profile == "qq":
            skill_names.append("rich-markdown")
        if skill_names:
            input_data["pinned_skills"] = list(dict.fromkeys(skill_names))
        stream = await agent.astream_events(
            input_data,
            config=config,
            context=runtime_context,
            version="v3",
        )
        progress_task = asyncio.create_task(collect_progress(stream, progress_reporter))
        try:
            response = await stream.output()
        finally:
            try:
                if callable(abort := getattr(stream, "abort", None)):
                    await abort()
            finally:
                await finish_progress_collection(progress_task)

        if response is None:
            response = {}
        if history_ids:
            response = {**response, "messages": [
                message for message in response.get("messages", []) if getattr(message, "id", None) not in history_ids
            ]}
        should_reply = not bool(response.get("suppress_reply", False))
        artifacts = (
            await FrontierCognitive.extract_uni_messages(response) if should_reply else []
        )
        ai_messages = [message for message in response.get("messages", []) if getattr(message, "type", None) == "ai"]
        if should_reply:
            final_response = (
                ai_messages[-1]
                if ai_messages
                else AIMessage("智能代理处理完成，但没有生成响应。")
            )
        else:
            final_response = AIMessage("")

        processing_time = time.time() - start_time
        logger.info(f"Agent烤熟了~🥓 (耗时: {processing_time:.2f}s)")
        await emit_progress(
            progress_reporter,
            ProgressEvent(type="done", message="Agent 已完成", detail={"success": True}),
        )

        return {
            "response": {"messages": response.get("messages", []) if include_messages else [final_response]},
            "total_time": processing_time,
            "artifacts": artifacts,
            "uni_messages": artifacts,
            "should_reply": should_reply,
        }

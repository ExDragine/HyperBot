# ruff: noqa: E402

import asyncio
import os
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from langchain.messages import AIMessage
from nonebot import get_bot, get_driver, logger, on_message, on_notice
from nonebot.adapters.milky.event import (
    GroupDisbandEvent,
    GroupMemberDecreaseEvent,
    GroupMemberIncreaseEvent,
    MessageEvent,
    MessageRecallEvent,
)
from nonebot_plugin_apscheduler import scheduler

from utils.agent_orchestration import ConversationOrchestrator, TurnOutcome, TurnStatus
from utils.agent_protocol import (
    AgentRequest,
    AgentResponse,
    ConversationRef,
    DeliveryReceipt,
    DeliveryStatus,
    GateDecision,
    MessageRef,
)
from utils.agents import (
    FrontierCognitive,
    conversation_workspace_key,
    run_serialized,
)
from utils.agents.message_envelope import (
    build_agent_attachment_payload,
    serialize_agent_payload,
)
from utils.agents.message_envelope import content_for_persisted_images as _remove_attached_image_placeholders
from utils.agents.neutral_core import FrontierAgentCore
from utils.agents.runtime_gateway import FrontierAgentRuntime
from utils.configs import EnvConfig
from utils.database import MessageDatabase
from utils.database.qq_sessions import QqSessionKey, QqTurnContext
from utils.database.schema import platform_message_key
from utils.media import resolve_media
from utils.message import (
    download_media,
    message_check,  # noqa: F401  # deprecated compatibility export; provider-side safety is authoritative
    message_extract,
    outgoing_message_content,  # noqa: F401  # re-exported: tests and callers read it from this module
    sanitize_outgoing_text,
    send_messages,
)
from utils.speech_recognition import transcribe_message_audio
from utils.timeutil import add_daily_job

from .adapters import (
    QqDelivery,
    QqHistoryStore,
    QqMessageAdapter,
    QqReplyPolicy,
    QqToolProvider,
)
from .attachments import cleanup_staged_message_files, extract_message_files, stage_message_files
from .document_parsing import parse_staged_files
from .gateway import message_gateway
from .message_normalizer import NORMALIZED_VERSION, normalize_segments, segments_to_raw_json
from .reply_context import (
    build_reply_context,
    hydrate_recent_media_context,
    reply_seq_from_segments,
    requests_recent_media,
    segments_directly_mention_user,
    sender_names_from_milky_message,
)
from .session_admission import QQ_REQUEST_CANCELLED, QqAdmission, cancel_group_work, find_pending, reserve

messages_db = MessageDatabase()


async def _is_session_notice(event) -> bool:
    return isinstance(event, (GroupDisbandEvent, GroupMemberDecreaseEvent, GroupMemberIncreaseEvent, MessageRecallEvent))


session_notice = on_notice(rule=_is_session_notice, priority=10, block=False)


@session_notice.handle()
async def handle_session_notice(
    event: GroupDisbandEvent | GroupMemberDecreaseEvent | GroupMemberIncreaseEvent | MessageRecallEvent,
):
    bot_id = int(event.self_id)
    if isinstance(event, MessageRecallEvent):
        if event.data.message_scene == "group":
            await messages_db.qq_lifecycle.recall(bot_id, event.data.peer_id, event.data.message_seq)
            cancel_group_work(bot_id, event.data.peer_id, sequence=event.data.message_seq)
        return
    group_id = event.data.group_id
    user_id = bot_id if isinstance(event, GroupDisbandEvent) else event.data.user_id
    active = isinstance(event, GroupMemberIncreaseEvent)

    async def apply_notice():
        whole_group = user_id == bot_id
        changed = await messages_db.qq_lifecycle.membership(bot_id, group_id, 0 if whole_group else user_id, event.time, active=active)
        if changed and not active:
            cancel_group_work(bot_id, group_id, user_id=None if whole_group else user_id)

    await run_serialized(f"qq-membership:{bot_id}:{group_id}", apply_notice)
f_cognitive = FrontierCognitive()
driver = get_driver()

common = on_message(priority=10)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_CLEANUP_JOB_ID = "frontier_daily_cache_cleanup"
_CANCEL_COMMAND = re.compile(r"^/cancel(?:\s+(\d{1,19}))?$", re.IGNORECASE)


@dataclass(slots=True)
class AgentRequestContext:
    event: MessageEvent
    user_id: str
    user_name: str
    event_id: int
    group_id: int | None
    msg_time: int
    text: str
    quoted_images: list[bytes]
    images: list[bytes]
    videos: list[bytes]
    audio: list[bytes] = field(default_factory=list)
    recent_images: list[bytes] = field(default_factory=list)
    attachments: list[dict[str, object]] = field(default_factory=list)
    current_attachments: list[dict[str, object]] = field(default_factory=list)
    recent_attachments: list[dict[str, object]] = field(default_factory=list)
    user_nickname: str | None = None
    user_card: str | None = None
    reply_to: dict[str, object] | None = None
    reply_seq: int | None = None
    direct_mention: bool = False
    message_id: int | None = None
    admission: QqAdmission | None = None
    delivery_started: bool = False


def _agent_workspace_key(user_id: str, group_id: int | None) -> str:
    return conversation_workspace_key(user_id, group_id)


def _qq_reply_ref(context: AgentRequestContext, conversation: ConversationRef) -> MessageRef | None:
    """Build an opaque reply reference without exposing the legacy payload."""

    message_id = None
    if isinstance(context.reply_to, dict):
        message_id = context.reply_to.get("message_id")
    if message_id is None:
        message_id = context.reply_seq
    if message_id is None:
        return None
    return MessageRef(platform="qq", conversation=conversation, message_id=str(message_id))


def _agent_memory_dir(user_id: str, group_id: int | None) -> Path:
    working_dir = Path(getattr(f_cognitive, "working_dir", os.path.join(os.getcwd(), "cache", "sandbox")))
    return working_dir / "memory" / _agent_workspace_key(user_id, group_id)


async def _collect_incoming_assets(
    media_coro,
    files_coro,
    quote_coro=None,
):
    coroutines = [media_coro, files_coro]
    if quote_coro is not None:
        coroutines.append(quote_coro)
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        for task in tasks:
            task.cancel()
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        cleanup_staged_message_files(settled[1] if isinstance(settled[1], list) else [])
        raise
    staged_files = results[1] if isinstance(results[1], list) else []
    if phase_error := next((result for result in results if isinstance(result, BaseException)), None):
        cleanup_staged_message_files(staged_files)
        raise phase_error
    quote_result = results[2] if quote_coro is not None else (None, [])
    return results[0], staged_files, quote_result


def _group_member_role(event: MessageEvent) -> str | None:
    member = getattr(getattr(event, "data", None), "group_member", None)
    role = getattr(member, "role", None)
    if role in (None, ""):
        return None
    return str(role)


def _allows_silent_reply(context: AgentRequestContext) -> bool:
    """Only opportunistic, non-addressed group turns may end silently."""
    if context.group_id is None or context.direct_mention:
        return False
    is_tome = getattr(context.event, "is_tome", None)
    return not bool(is_tome()) if callable(is_tome) else True


def _remove_structured_reply_marker(text: str, reply_seq: int | None) -> str:
    """Drop the legacy text marker once reply identity is carried structurally."""
    if reply_seq is None or not text:
        return text
    marker = f"[回复消息:{reply_seq}]"
    return text.removeprefix(marker).strip()


def _with_media_placeholders(text, audio_downloaders, video_downloaders) -> str:
    if video_downloaders and "[视频" not in text:
        text += "\n" + " ".join("[视频]" for _ in video_downloaders)
    if audio_downloaders and "[语音" not in text:
        text += "\n" + " ".join("[语音]" for _ in audio_downloaders)
    return text.strip()


async def _cancel_qq_request(key: QqSessionKey, sequence: int | None) -> None:
    if sequence is None:
        await common.finish("请引用你的触发消息发送 /cancel，或使用 /cancel <消息序号>。")
    pending = find_pending(key, sequence)
    if pending is None:
        await common.finish("这条消息没有可取消的本人请求。")
    platform_key = platform_message_key(msg_id=sequence, user_id=key.user_id, group_id=key.group_id, bot_user_id=key.bot_id)
    await messages_db.qq_lifecycle.cancel_ingress(platform_key)
    pending.cancel()
    await common.finish("已请求停止处理这条消息。")


def _read_cancel_command(segments):
    if any(segment.get("type") not in {"reply", "mention", "text"} for segment in segments):
        return None
    text = "".join(str(segment.get("data", {}).get("text", "")) for segment in segments if segment.get("type") == "text")
    return _CANCEL_COMMAND.fullmatch(text.strip())


async def _close_qq_turn(trigger_id: int, enqueue: asyncio.Task[bool]) -> None:
    try:
        if await asyncio.shield(enqueue):
            await messages_db.qq_sessions.interrupt(trigger_id)
    except Exception as exc:
        logger.warning("会话状态结算失败: {}", type(exc).__name__)


async def _may_continue(context: AgentRequestContext) -> bool:
    if context.admission is None:
        return True
    if context.admission.cancelled:
        return False
    if context.group_id is None or context.message_id is None:
        return True
    key = context.admission.key
    store = getattr(messages_db, "qq_lifecycle", None)
    if store is None:
        return True
    if not await store.permits(key.bot_id, context.group_id, key.user_id, context.message_id, context.event_id):
        return False
    sources = context.admission.sources
    return not sources or not await store.recalled(key.bot_id, context.group_id, sources)


def _qq_neutral_eligible(context: AgentRequestContext) -> bool:
    """Return whether the normalized QQ event contains an Agent turn.

    The reply gateway has already authorized the turn. Persisting an image can
    remove its text placeholder, so media and attachment payloads must also
    count as input here.
    """

    return bool(
        context.text.strip()
        or context.reply_seq is not None
        or context.reply_to
        or context.direct_mention
        or context.images
        or context.quoted_images
        or context.recent_images
        or context.audio
        or context.videos
        or context.attachments
        or context.current_attachments
        or context.recent_attachments
    )


class _QqNeutralCore:
    """Sanitize the neutral response before the QQ delivery port sends it."""

    def __init__(self, core: FrontierAgentCore) -> None:
        self._core = core

    async def run(self, request: AgentRequest, *, tools=(), progress_reporter=None) -> AgentResponse:
        if progress_reporter is None:
            response = await self._core.run(request, tools=tools)
        else:
            response = await self._core.run(request, tools=tools, progress_reporter=progress_reporter)
        sanitized = await sanitize_outgoing_text(response.text)
        sanitized_text = sanitized or ""
        if sanitized_text == response.text:
            return response
        return replace(response, text=sanitized_text)


class _QqNeutralDelivery(QqDelivery):
    """Reject empty turns before they reach the QQ delivery port."""

    def __init__(self, *, context: AgentRequestContext, trigger_id: int | None, text_sender):
        super().__init__(text_sender=text_sender)
        self.context = context
        self.trigger_id = trigger_id

    async def send(self, target: ConversationRef, response: AgentResponse) -> DeliveryReceipt:
        if not response.text.strip() and not response.artifacts:
            return DeliveryReceipt(DeliveryStatus.FAILED, errors=("empty_response",))
        if not await _may_continue(self.context):
            return DeliveryReceipt(DeliveryStatus.FAILED, errors=("cancelled",))
        if self.trigger_id is not None:
            await messages_db.qq_sessions.begin_delivery(self.trigger_id)
        self.context.delivery_started = True
        return await super().send(target, response)

    async def settle(self, outcome: TurnOutcome) -> None:
        if self.trigger_id is None:
            return
        if outcome.status in {TurnStatus.SILENT, TurnStatus.GATED}:
            await messages_db.qq_sessions.silent(self.trigger_id)
        elif self.context.delivery_started and outcome.receipt is not None and not outcome.receipt.delivered_text:
            await messages_db.qq_sessions.finish_without_text(
                self.trigger_id, outcome.receipt.status.value, time.time_ns() // 1_000_000,
            )


async def _qq_reply_id_for_context(context: AgentRequestContext) -> int | None:
    """Reuse the delayed-group reply heuristic for neutral sends."""

    if context.group_id is None or time.time() * 1000 - context.msg_time < 10_000:
        return None
    count_intervening = getattr(messages_db, "count_intervening_group_messages", None)
    if not callable(count_intervening):
        return None
    intervening_messages = await count_intervening(
        group_id=context.group_id,
        bot_user_id=int(context.event.self_id),
        user_id=int(context.user_id),
        after_time=context.msg_time,
        after_message_id=context.message_id,
    )
    return context.event_id if intervening_messages >= 5 else None


async def _send_qq_neutral_notice(context: AgentRequestContext, text: str) -> bool:
    """Send one bounded neutral-path failure notice without rerunning the Agent."""

    try:
        if not await _may_continue(context):
            return False
        reply_id = await _qq_reply_id_for_context(context)
        delivery = await send_messages(
            context.group_id,
            reply_id,
            {"messages": [AIMessage(content=text)]},
        )
    except Exception as exc:
        logger.warning("QQ 中立路径错误提示发送失败: {}", type(exc).__name__)
        return False
    return delivery.sent > 0


async def _run_qq_neutral(
    context: AgentRequestContext,
    history_messages: list[dict[str, Any]],
    turn: QqTurnContext | None = None,
) -> bool:
    """Run one QQ turn through the platform-neutral application boundary.

    Returns whether the turn is considered handled.  Every path goes through
    the neutral boundary, so the legacy Agent is never retried afterwards.
    """

    adapter = QqMessageAdapter(getattr(context.event, "self_id", ""))
    conversation = adapter.conversation(group_id=context.group_id, user_id=context.user_id)
    trigger_id = context.message_id if turn is not None else None
    metadata = None if turn is None else {
        "qq_trigger_id": trigger_id,
        "qq_session_context": {
            "own_messages": turn.own_messages,
            "group_background": turn.group_background,
            "replies_delivered_after_trigger": turn.later_replies,
        },
    }
    message = adapter.text_message(
        message_id=context.event_id,
        group_id=context.group_id,
        user_id=context.user_id,
        user_name=context.user_name,
        text=context.text or ("[用户叫了你一声]" if context.direct_mention else ""),
        created_at=datetime.fromtimestamp(context.msg_time / 1000, tz=UTC),
        role=_group_member_role(context.event),
        mentions_agent=context.direct_mention,
        images=context.images,
        audio=context.audio,
        videos=context.videos,
        attachments=context.current_attachments or context.attachments,
        recent_images=context.recent_images,
        recent_attachments=context.recent_attachments,
        reply_to=_qq_reply_ref(context, conversation),
        quoted_payload=context.reply_to,
        quoted_images=context.quoted_images,
        metadata=metadata,
    )

    async def load_history(_query):
        return turn.history if turn is not None else history_messages

    async def allow_gateway_result(_message, _history):
        return GateDecision(True, reason="legacy_gateway_passed")

    async def send_text(_target: ConversationRef, text: str):
        reply_id = await _qq_reply_id_for_context(context)
        return await send_messages(
            context.group_id,
            reply_id,
            {"messages": [AIMessage(content=text)]},
        )

    delivery = _QqNeutralDelivery(context=context, trigger_id=trigger_id, text_sender=send_text)
    outcome = await ConversationOrchestrator(
        _QqNeutralCore(FrontierAgentCore(FrontierAgentRuntime(cognitive=f_cognitive))),
        history_limit=EnvConfig.QUERY_MESSAGE_NUMBERS,
    ).handle(
        message,
        policy=QqReplyPolicy(allow_gateway_result),
        history=QqHistoryStore(database=messages_db, loader=load_history, session_trigger_id=trigger_id),
        delivery=delivery,
        tools=QqToolProvider(),
        workspace_key=_agent_workspace_key(context.user_id, context.group_id),
        capabilities=frozenset({"platform:qq", "qq:tools"}),
        allow_silent_reply=_allows_silent_reply(context),
    )
    await delivery.settle(outcome)
    if outcome.status in {TurnStatus.DELIVERED, TurnStatus.HISTORY_APPEND_FAILED}:
        if outcome.status == TurnStatus.HISTORY_APPEND_FAILED:
            logger.warning("QQ 回复已送达但历史写入失败: trigger_id={}", trigger_id)
        return outcome.receipt is not None and outcome.receipt.status == DeliveryStatus.DELIVERED
    if outcome.status in {TurnStatus.SILENT, TurnStatus.GATED}:
        return False
    if outcome.status in {TurnStatus.HISTORY_FAILED, TurnStatus.GATE_FAILED}:
        logger.warning("QQ 中立路径前置阶段失败: {}", outcome.error or outcome.status)
        return await _send_qq_neutral_notice(context, "本轮上下文准备失败，请稍后重试。")
    if outcome.status == TurnStatus.AGENT_FAILED:
        notice = (
            outcome.response.text.strip()
            if outcome.response is not None and outcome.response.text.strip()
            else "本轮处理失败，请稍后重试。"
        )
        return await _send_qq_neutral_notice(context, notice)
    logger.warning(
        "QQ 中立路径投递失败: {} {}", outcome.error or outcome.status,
        outcome.receipt.errors if outcome.receipt is not None else (),
    )
    return False


async def _process_agent_request(
    context: AgentRequestContext,
    history_messages: list[dict[str, Any]] | None = None,
) -> bool:
    if not await _may_continue(context):
        return False
    if not _qq_neutral_eligible(context):
        logger.info("忽略没有文本、引用或直接提及的 QQ 消息: event_id={}", context.event_id)
        return False
    if context.admission is not None:
        context.admission.running = True
    turn = None
    if context.admission is not None and context.message_id is not None:
        try:
            turn = await messages_db.qq_sessions.start(
                context.admission.key, context.message_id, context.msg_time, _turn_trigger(context),
                limit=EnvConfig.QUERY_MESSAGE_NUMBERS,
            )
        except Exception as exc:
            logger.warning("QQ 会话历史准备失败: {}", type(exc).__name__)
            return await _send_qq_neutral_notice(context, "本轮上下文准备失败，请稍后重试。")
        sequences = turn.source_sequences
        if quote := (context.reply_to or {}).get("message_id"):
            sequences = sequences | {str(quote)}
        context.admission.sources = sequences
        key = context.admission.key
        if key.group_id is not None and sequences and await messages_db.qq_lifecycle.recalled(key.bot_id, key.group_id, sequences):
            return False
    return await _run_qq_neutral(context, list(history_messages or []), turn)


def _turn_trigger(context: AgentRequestContext) -> dict[str, object]:
    trigger = {
        "message_id": str(context.event_id),
        "time_ms": context.msg_time,
        "sender": {"user_id": context.user_id, "display_name": context.user_name},
        "content": (context.text or "[用户叫了你一声]")[:8_000],
        "attachments": context.current_attachments[:12],
    }
    if context.reply_to:
        trigger["quote"] = {
            "message_id": context.reply_to.get("message_id"),
            "content": str(context.reply_to.get("content") or "")[:2_000],
        }
    return trigger


async def _set_group_processing_reaction(bot: Any, group_id: int, message_seq: int, *, is_add: bool) -> None:
    try:
        async with asyncio.timeout(2):
            await bot.send_group_message_reaction(
                group_id=group_id,
                message_seq=message_seq,
                reaction="32",
                is_add=is_add,
            )
    except Exception as exc:
        action = "添加" if is_add else "移除"
        logger.warning(
            "{}群消息处理表情失败: group_id={} message_seq={} error={}",
            action,
            group_id,
            message_seq,
            type(exc).__name__,
        )


async def _process_queued_agent_request(
    context: AgentRequestContext, history_messages: list[dict[str, Any]], *, bot: Any
) -> None:
    started = False
    group_id = context.group_id
    reaction_started = False

    async def process() -> bool:
        nonlocal started, reaction_started
        started = True
        if group_id is not None and _qq_neutral_eligible(context):
            reaction_started = True
            await _set_group_processing_reaction(bot, group_id, context.event_id, is_add=True)
        return await _process_agent_request(context, history_messages)

    try:
        try:
            delivery_key = f"delivery:{_agent_workspace_key(context.user_id, group_id)}"
            async with asyncio.timeout(EnvConfig.AGENT_JOB_TIMEOUT_SECONDS):
                if context.admission is not None:
                    await context.admission.wait()
                await run_serialized(delivery_key, process)
        finally:
            if group_id is not None and reaction_started:
                await _set_group_processing_reaction(bot, group_id, context.event_id, is_add=False)
    except TimeoutError:
        logger.warning("会话回复超时: group_id={} user_id={} started={}", context.group_id, context.user_id, started)
        if context.delivery_started or not await _may_continue(context):
            return
        notice = "本轮处理超时，请稍后重试。" if started else "等待处理超时，本轮请求尚未开始，请稍后重试。"
        try:
            async with asyncio.timeout(15):
                delivery = await send_messages(
                    context.group_id, context.event_id, {"messages": [AIMessage(content=notice)]}
                )
                if delivery.errors:
                    logger.warning("会话超时提示未送达: {}", delivery.errors)
        except Exception as exc:
            logger.warning("会话超时提示发送失败: {}", type(exc).__name__)


async def _run_agent_turn(
    *,
    bot: Any,
    context: AgentRequestContext,
    history_messages: list[dict[str, Any]],
    previous_reply_payload: dict[str, object] | None,
    fetched_reply_payload: dict[str, object] | None,
    original_text: str,
) -> None:
    finalize_message_context = getattr(messages_db, "finalize_message_context", None)
    if callable(finalize_message_context) and context.reply_to != previous_reply_payload:
        try:
            await finalize_message_context(
                time=context.msg_time,
                **({"message_id": context.message_id} if context.message_id is not None else {}),
                reply_context_json=(serialize_agent_payload(context.reply_to) if context.reply_to else None),
            )
        except Exception as exc:
            logger.warning("消息上下文定稿失败（不影响回复）: {}: {}", type(exc).__name__, exc)

    from utils.ens_gate import _ens_prefix

    cleaned = original_text.strip().lstrip("/")
    if cleaned[:3].lower() == "vep":
        _ens_prefix.set("vep")
    elif cleaned[:2].lower() == "ve":
        _ens_prefix.set("ve")
    else:
        _ens_prefix.set("")

    await _process_queued_agent_request(context, history_messages, bot=bot)



@driver.on_shutdown
async def on_shutdown():
    import_module("skills.ens-weather.tools.ens_professional").clear_ens_cache()
    from utils.browser_capture import close_browser
    from utils.http_client import aclose_all

    await close_browser()
    await aclose_all()


@driver.on_startup
async def on_startup():
    await messages_db.qq_sessions.interrupt()
    if EnvConfig.IMAGE_AUTO_CLEANUP:
        try:
            cleaned_attachments = await messages_db.cleanup_expired_attachments()
            if cleaned_attachments:
                logger.info("已清理过期消息附件: {}", cleaned_attachments)
        except Exception as exc:
            logger.warning("消息附件维护失败: {}: {}", type(exc).__name__, exc)

    add_daily_job(scheduler, CACHE_CLEANUP_JOB_ID, run_daily_cache_cleanup)


async def run_daily_cache_cleanup() -> None:
    """Run bounded cache maintenance once per day."""
    if EnvConfig.IMAGE_AUTO_CLEANUP:
        try:
            cleaned_attachments = await messages_db.cleanup_expired_attachments()
            if cleaned_attachments:
                logger.info("每日清理过期消息附件: {}", cleaned_attachments)
        except Exception as exc:
            logger.warning("每日消息附件清理失败: {}: {}", type(exc).__name__, exc)


@common.handle()
async def handle_common(event: MessageEvent):
    try:
        async with AsyncExitStack() as scope:
            await _handle_common(event, scope)
    except asyncio.CancelledError as exc:
        if exc.args != (QQ_REQUEST_CANCELLED,):
            raise


async def _handle_common(event: MessageEvent, scope: AsyncExitStack):  # noqa: C901
    msg_time = time.time_ns() // 1_000_000
    if EnvConfig.AGENT_MODULE_ENABLED is False:
        await common.finish(f"{EnvConfig.BOT_NAME}飞升了,暂时不可用")

    try:
        bot = get_bot()
    except ValueError:
        bot = getattr(event, "bot", None)
        if bot is None:
            await common.finish()
    user_id = event.get_user_id()
    user_name, user_nickname, user_card = sender_names_from_milky_message(event.data)
    event_id = event.data.message_seq
    group_id = event.data.group.group_id if event.data.group else None
    direct_mention = segments_directly_mention_user(event.data.segments, event.self_id)

    text, image_downloaders, audio_downloaders, video_downloaders = await message_extract(event.data.segments)
    file_items = extract_message_files(event.data.segments)
    reply_seq = reply_seq_from_segments(event.data.segments)
    key = QqSessionKey(int(event.self_id), group_id, int(user_id))
    cancel = _read_cancel_command(event.data.segments)
    if find_pending(key, event_id) is not None:
        await common.finish()
    admission = reserve(key, event_id) if cancel is None and key.user_id != key.bot_id else None
    if admission is not None:
        scope.callback(admission.release)

    raw_text = _with_media_placeholders(_remove_structured_reply_marker(text, reply_seq), audio_downloaders, video_downloaders)
    raw_segments = segments_to_raw_json(event.data.segments)
    needs_forward = any(segment.get("type") == "forward" for segment in event.data.segments)
    lifecycle = getattr(messages_db, "qq_lifecycle", None)
    platform_key = platform_message_key(
        msg_id=event_id, user_id=key.user_id, group_id=key.group_id, bot_user_id=key.bot_id,
    )

    async def store_incoming():
        result = await messages_db.insert(
            time=msg_time,
            msg_id=event_id,
            user_id=key.user_id,
            sender_user_id=key.user_id,
            group_id=group_id,
            user_name=user_name,
            user_nickname=user_nickname,
            user_card=user_card,
            role="user" if key.user_id != key.bot_id else "assistant",
            content=raw_text,
            bot_user_id=key.bot_id,
            directly_mentions_bot=direct_mention,
            raw_segments_json=raw_segments,
            normalized_version=NORMALIZED_VERSION,
            normalized_status="partial" if needs_forward else "complete",
            track_qq_ingress=admission is not None,
        )
        if lifecycle is not None and platform_key is not None and result is not None and admission is not None:
            if result.inserted or result.resume_ingress:
                scope.push_async_callback(lifecycle.finish_ingress, platform_key)
        return result

    workspace = _agent_workspace_key(user_id, group_id)
    inserted = await run_serialized(f"qq-ingress:{key.bot_id}:{workspace}", store_incoming)
    if inserted is not None and not inserted.inserted and not getattr(inserted, "resume_ingress", False):
        logger.info("忽略已处理的平台消息: group_id={} message_seq={}", group_id, event_id)
        await common.finish()
    if key.user_id == key.bot_id:
        await common.finish()
    if cancel is not None:
        sequence = int(cancel[1]) if cancel[1] else reply_seq
        await _cancel_qq_request(key, sequence)
        return
    if admission is None:
        await common.finish()
    message_id = inserted.message_id if inserted is not None else None
    if group_id is not None and message_id is not None and lifecycle is not None:
        if not await lifecycle.permits(key.bot_id, group_id, key.user_id, message_id, event_id):
            await common.finish()
    preparation_deadline = await scope.enter_async_context(asyncio.timeout(EnvConfig.AGENT_JOB_TIMEOUT_SECONDS))

    normalized_message = await normalize_segments(bot, event.data.segments)
    current_text = normalized_message.content or text
    reply_payload = None
    if reply_seq:
        reply_payload, _ = await build_reply_context(
            bot,
            event,
            reply_seq,
            group_id,
            messages_db,
            load_images=False,
            workspace_key=workspace,
            memory_dir=_agent_memory_dir(user_id, group_id),
        )
        if reply_payload:
            current_text = _remove_structured_reply_marker(current_text, reply_seq)
    current_text = _with_media_placeholders(current_text, audio_downloaders, video_downloaders)
    text = current_text
    if not text and not reply_payload and not event.is_tome():
        await common.finish()
    message_identity = {"message_id": message_id} if message_id is not None else {}
    update_normalization = getattr(messages_db, "update_message_normalization", None)
    if callable(update_normalization) and (text != raw_text or needs_forward):
        await update_normalization(
            time=inserted.time if hasattr(inserted, "time") else msg_time,
            **message_identity,
            content=text,
            raw_segments_json=normalized_message.raw_segments_json,
            normalized_version=normalized_message.normalized_version,
            normalized_status=normalized_message.status,
        )
    finalize = getattr(messages_db, "finalize_message_context", None)
    if reply_payload and callable(finalize):
        await finalize(
            time=inserted.time if hasattr(inserted, "time") else msg_time, **message_identity,
            reply_context_json=serialize_agent_payload(reply_payload),
        )
    if normalized_message.derived_messages:
        await messages_db.replace_derived_messages(
            **({"parent_message_id": message_id} if message_id is not None else {}),
            parent_msg_time=msg_time,
            parent_msg_id=event_id,
            user_id=int(user_id),
            group_id=group_id,
            role="user" if user_id != str(event.self_id) else "assistant",
            derived_messages=normalized_message.derived_messages,
            normalized_version=NORMALIZED_VERSION,
        )

    gateway_messages = await messages_db.prepare_message(
        int(user_id),
        group_id,
        query_numbers=EnvConfig.QUERY_MESSAGE_NUMBERS,
        before_time=msg_time,
        bot_user_id=int(event.self_id),
    )

    if not await message_gateway(event, gateway_messages):
        await common.finish()

    if message_id is not None:
        trigger_id = message_id
        enqueue = asyncio.create_task(messages_db.qq_sessions.enqueue(key, trigger_id))
        scope.push_async_callback(_close_qq_turn, trigger_id, enqueue)
        if not await asyncio.shield(enqueue):
            await common.finish()

    media_task = download_media(image_downloaders, audio_downloaders, video_downloaders)
    files_task = stage_message_files(
        bot,
        file_items,
        memory_dir=_agent_memory_dir(user_id, group_id),
        workspace_key=_agent_workspace_key(user_id, group_id),
        message_time=msg_time,
        **message_identity,
        user_id=user_id,
        group_id=group_id,
    )
    if reply_seq:
        quote_task = build_reply_context(
            bot,
            event,
            reply_seq,
            group_id,
            messages_db,
            workspace_key=_agent_workspace_key(user_id, group_id),
            memory_dir=_agent_memory_dir(user_id, group_id),
        )
        (images, audio, videos), staged_files, (agent_reply_payload, quoted_images) = await _collect_incoming_assets(
            media_task,
            files_task,
            quote_task,
        )
    else:
        (images, audio, videos), staged_files, _quote_result = await _collect_incoming_assets(
            media_task,
            files_task,
        )
        agent_reply_payload, quoted_images = None, []
    resolved_reply_payload = agent_reply_payload or reply_payload

    recent_images: list[bytes] = []
    recent_attachments: list[dict[str, object]] = []
    should_hydrate_recent = (
        reply_seq is None
        and not images
        and not audio
        and not videos
        and not file_items
        and requests_recent_media(current_text)
    )
    if should_hydrate_recent:
        try:
            recent_images, recent_attachments, _recent_media_found = await hydrate_recent_media_context(
                bot,
                event,
                user_id=int(user_id),
                group_id=group_id,
                before_time=msg_time,
                messages_db=messages_db,
                workspace_key=_agent_workspace_key(user_id, group_id),
                memory_dir=_agent_memory_dir(user_id, group_id),
            )
        except Exception as exc:
            logger.warning("按需恢复近期媒体失败（不影响回复）: {}: {}", type(exc).__name__, exc)

    persisted_media = []
    if EnvConfig.IMAGE_ENABLED:
        persisted_media.extend(resolve_media(image, "image") for image in images)
    persisted_media.extend(resolve_media(item, "audio") for item in audio)
    persisted_media.extend(resolve_media(item, "video") for item in videos)
    persisted_attachments = []
    if persisted_media:
        try:
            persisted_attachments = await messages_db.insert_media(
                msg_time=msg_time,
                **message_identity,
                msg_id=event_id,
                user_id=int(user_id),
                group_id=group_id,
                media=persisted_media,
            )
        except Exception as e:
            logger.warning(f"⚠️ 媒体保存失败（不影响主流程）: {e}")

    persisted_image_count = sum(
        1 for attachment in persisted_attachments if getattr(attachment, "kind", None) == "image"
    )
    current_text, audio_transcribed = await transcribe_message_audio(current_text, audio)
    if audio_transcribed:
        audio = []
        update_normalization = getattr(messages_db, "update_message_normalization", None)
        if message_id is not None and callable(update_normalization):
            try:
                await update_normalization(
                    time=msg_time,
                    message_id=message_id,
                    content=current_text,
                    raw_segments_json=None,
                    normalized_version=normalized_message.normalized_version,
                    normalized_status=normalized_message.status,
                )
            except Exception as exc:
                logger.warning("语音转写历史保存失败（本轮仍可使用）: {}", type(exc).__name__)
    agent_text = _remove_attached_image_placeholders(current_text, persisted_image_count).strip()

    indexed_staged_paths: set[Path] = set()
    if staged_files and hasattr(messages_db, "insert_attachment"):
        expires_at = int(time.time() * 1000) + EnvConfig.MEDIA_TTL_DAYS * 86400 * 1000
        for staged_file in staged_files:
            try:
                await messages_db.insert_attachment(
                    msg_time=msg_time,
                    **message_identity,
                    msg_id=event_id,
                    user_id=int(user_id),
                    group_id=group_id,
                    kind="file",
                    physical_path=str(staged_file.local_path),
                    virtual_path=staged_file.virtual_path,
                    file_name=staged_file.file_name,
                    mime_type=staged_file.mime_type,
                    file_size=staged_file.file_size,
                    sha256=staged_file.sha256,
                    expires_at=expires_at,
                )
            except Exception as e:
                logger.warning(f"⚠️ 文件附件索引失败（不影响主流程）: {e}")
            else:
                indexed_staged_paths.add(Path(staged_file.local_path))
    unindexed_staged_files = [
        staged_file for staged_file in staged_files if Path(staged_file.local_path) not in indexed_staged_paths
    ]

    try:
        ocr_text, parsed_documents = await parse_staged_files(
            staged_files,
            EnvConfig.DOCUMENT_PARSING,
        )
        if ocr_text:
            agent_text = f"{agent_text}\n\n{ocr_text}".strip()
            update_normalization = getattr(messages_db, "update_message_normalization", None)
            if message_id is not None and callable(update_normalization):
                try:
                    await update_normalization(
                        time=msg_time,
                        message_id=message_id,
                        content=f"{current_text}\n\n{ocr_text}".strip(),
                        raw_segments_json=None,
                        normalized_version=normalized_message.normalized_version,
                        normalized_status=normalized_message.status,
                    )
                except Exception as exc:
                    logger.warning("OCR 结果历史保存失败（本轮仍可使用）: {}: {}", type(exc).__name__, exc)
            logger.info(
                "DeepSeek OCR 完成 files={} pages={} chars={}",
                len(parsed_documents),
                sum(document.page_count for document in parsed_documents),
                sum(len(document.text) for document in parsed_documents),
            )
        current_attachment_refs = [
            dict(
                build_agent_attachment_payload(
                    kind=attachment.kind,
                    mime_type=attachment.mime_type,
                    file_name=attachment.file_name,
                    path=attachment.virtual_path,
                )
            )
            for attachment in persisted_attachments
        ]
        current_attachment_refs.extend(
            dict(
                build_agent_attachment_payload(
                    kind="file",
                    mime_type=staged_file.mime_type,
                    file_name=staged_file.file_name,
                    path=staged_file.virtual_path,
                )
            )
            for staged_file in staged_files
        )
        known_attachment_paths = {str(attachment.get("path", "")) for attachment in current_attachment_refs}
        attachment_refs = [
            *current_attachment_refs,
            *(
                attachment
                for attachment in recent_attachments
                if str(attachment.get("path", "")) not in known_attachment_paths
            ),
        ]
        if recent_attachments:
            agent_text = f"{agent_text}\n[以上附件来自用户刚才发送的历史消息]".strip()
        context = AgentRequestContext(
            event=event,
            user_id=user_id,
            user_name=user_name,
            user_nickname=user_nickname,
            user_card=user_card,
            event_id=event_id,
            group_id=group_id,
            msg_time=msg_time,
            text=agent_text,
            quoted_images=quoted_images,
            recent_images=recent_images,
            images=images,
            audio=audio,
            videos=videos,
            attachments=attachment_refs,
            current_attachments=current_attachment_refs,
            recent_attachments=recent_attachments,
            reply_to=resolved_reply_payload,
            reply_seq=reply_seq,
            direct_mention=direct_mention,
            message_id=message_id,
            admission=admission if message_id is not None else None,
        )
        preparation_deadline.reschedule(None)
        await _run_agent_turn(
            bot=bot,
            context=context,
            history_messages=gateway_messages,
            previous_reply_payload=reply_payload,
            fetched_reply_payload=agent_reply_payload,
            original_text=text,
        )
    finally:
        cleanup_staged_message_files(unindexed_staged_files)

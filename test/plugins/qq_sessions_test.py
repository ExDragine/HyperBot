"""Verify QQ session routing, delivery and recovery with an actual SQLite database."""

# ruff: noqa: S101

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from sqlmodel import Session, select

from utils.agent_protocol import DeliveryReceipt, DeliveryStatus
from utils.database import Message, MessageDatabase, get_engine
from utils.database.qq_sessions import QqSessionKey, QqSessionTurn
from utils.delivery import DeliveryResult


@pytest.fixture
def database(tmp_path, monkeypatch):
    from utils.database import store

    url = f"sqlite:///{tmp_path / 'sessions.db'}"
    monkeypatch.setattr(store, "get_engine", lambda: get_engine(url))
    return MessageDatabase()


@pytest.fixture
def agent(database, monkeypatch):
    from plugins.agent import handlers

    monkeypatch.setattr(handlers, "messages_db", database)
    monkeypatch.setattr(handlers, "QqToolProvider", lambda: SimpleNamespace(tools=lambda _capabilities: ()))
    return handlers


async def _user(database, sequence, text, *, group=20, user=10, bot=99, when=1_000):
    return (await database.insert(when, sequence, user, group, str(user), "user", text, bot_user_id=bot)).message_id


def _turn_row(database, trigger_id):
    with Session(database.engine) as session:
        return session.get(QqSessionTurn, trigger_id)


async def _run(agent, key, trigger_id, sequence, text):
    from plugins.agent.session_admission import reserve

    admission = reserve(key)
    assert await agent.messages_db.qq_sessions.enqueue(key, trigger_id)
    context = agent.AgentRequestContext(
        event=SimpleNamespace(self_id=str(key.bot_id)),
        user_id=str(key.user_id), user_name=str(key.user_id), group_id=key.group_id,
        event_id=sequence, message_id=trigger_id, msg_time=1_000, text=text,
        quoted_images=[], images=[], videos=[], direct_mention=True, admission=admission,
    )
    try:
        return await agent._process_agent_request(context)
    finally:
        await agent.messages_db.qq_sessions.interrupt(trigger_id)
        admission.release()


@pytest.mark.asyncio
async def test_qq_turns_continue_only_for_the_same_person_and_bundle_the_interval(agent, database, monkeypatch):
    seen = {}

    class Cognitive:
        async def chat_agent(self, messages, **kwargs):
            seen[kwargs["user_text"]] = messages
            return {"response": {"messages": [AIMessage(content="答复 " + kwargs["user_text"])]}}

    async def send(*_args):
        return DeliveryResult(attempted=1, sent=1, message_ids=(5_000 + len(seen),))

    monkeypatch.setattr(agent, "f_cognitive", Cognitive())
    monkeypatch.setattr(agent, "send_messages", send)
    alice = QqSessionKey(99, 20, 10)
    bob = QqSessionKey(99, 20, 11)
    await _user(database, 900, "首次触发前")
    first = await _user(database, 901, "部署")
    await _user(database, 902, "先部署到测试环境")
    other = await _user(database, 903, "日志在哪里", user=11)
    await _user(database, 904, "生产库有写入", user=12)
    await _user(database, 905, "另一个机器人", bot=98)
    await _user(database, 906, "另一个群", group=21)
    await _user(database, 907, "私聊内容", group=None)
    follow_up = await _user(database, 908, "怎么回滚")
    await _user(database, 909, "触发后到达", when=900)

    assert await _run(agent, alice, first, 901, "部署")
    assert await _run(agent, bob, other, 903, "日志在哪里")
    assert await _run(agent, alice, follow_up, 908, "怎么回滚")
    assert len(seen["部署"]) == len(seen["日志在哪里"]) == 1
    assert json.loads(seen["部署"][-1]["content"][0]["text"])["session"]["own_messages"] == []
    messages = seen["怎么回滚"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "答复 部署"
    current = json.loads(messages[-1]["content"][0]["text"])
    assert current["sender"]["user_id"] == "10"
    assert [item["content"] for item in current["session"]["own_messages"]] == ["先部署到测试环境"]
    assert [item["content"] for item in current["session"]["group_background"]] == ["日志在哪里", "生产库有写入"]
    prior = _turn_row(database, first)
    assert prior.status == "settled"
    assert _turn_row(database, follow_up).previous_trigger_id == first
    assert current["session"]["replies_delivered_after_trigger"] == [
        {"reply_row_id": prior.reply_message_id, "delivered_at_ms": prior.delivered_at_ms}
    ]
    assert json.loads(messages[0]["content"])["reply_receipt"]["reply_row_id"] == prior.reply_message_id
    assert "群聊背景" not in _turn_row(database, follow_up).input_json
    assert "生产库有写入" not in _turn_row(database, follow_up).input_json


@pytest.mark.asyncio
async def test_reopen_preserves_personal_history_and_closes_unfinished_turns(database):
    store = database.qq_sessions
    key = QqSessionKey(99, None, 10)
    finished = await _user(database, 201, "first question", group=None)
    awaiting = await _user(database, 202, "waiting", group=None)
    sending = await _user(database, 203, "sending", group=None)
    next_id = await _user(database, 204, "continue", group=None)
    assert await store.enqueue(key, finished)
    assert not await store.enqueue(key, finished)
    await store.start(key, finished, 1_000, {"content": "first question"}, limit=10)
    await store.begin_delivery(finished)
    reply = await database.insert(
        1_100, 205, 10, None, "Assistant", "assistant", "confirmed", bot_user_id=99,
        session_trigger_id=finished,
    )
    await store.enqueue(key, awaiting)
    await store.enqueue(key, sending)
    await store.start(key, sending, 1_000, {"content": "sending"}, limit=10)
    await store.begin_delivery(sending)
    database.engine.dispose()

    reopened = MessageDatabase()
    await reopened.qq_sessions.interrupt()
    assert _turn_row(reopened, finished).reply_message_id == reply.message_id
    assert _turn_row(reopened, awaiting).status == "failed"
    assert _turn_row(reopened, sending).status == "unknown"
    await reopened.qq_sessions.enqueue(key, next_id)
    context = await reopened.qq_sessions.start(key, next_id, 1_000, {"content": "continue"}, limit=10)
    assert [message.role for message in context.history] == ["user", "assistant", "user"]
    assert context.history[1].content == "confirmed"
    assert json.loads(context.history[-1].content)["turn_status"] == "unknown"
    assert context.group_background == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [DeliveryStatus.PARTIAL, DeliveryStatus.FAILED, DeliveryStatus.UNKNOWN])
async def test_only_confirmed_delivery_is_linked_to_trigger(agent, database, monkeypatch, status):
    key = QqSessionKey(99, 20, 10)
    trigger_id = await _user(database, 600, "trigger")

    class Cognitive:
        async def chat_agent(self, *_args, **_kwargs):
            return {"response": {"messages": [AIMessage(content="reply")]}}

    monkeypatch.setattr(agent, "f_cognitive", Cognitive())

    async def send(_delivery, _target, _response):
        assert _turn_row(database, trigger_id).status == "sending"
        confirmed = "部分送达的文字" if status == DeliveryStatus.PARTIAL else None
        return DeliveryReceipt(status, delivered_text=confirmed, text_message_refs=())

    monkeypatch.setattr(agent.QqDelivery, "send", send)
    assert not await _run(agent, key, trigger_id, 600, "trigger")
    turn = _turn_row(database, trigger_id)
    assert turn.delivery_status == status.value
    assert turn.status == ("settled" if status == DeliveryStatus.PARTIAL else status.value)
    with Session(database.engine) as session:
        row = session.exec(select(Message).where(Message.role == "assistant")).first()
        if status == DeliveryStatus.PARTIAL:
            assert row.id == turn.reply_message_id
            assert row.content == "部分送达的文字"
        else:
            assert row is None
            assert turn.reply_message_id is None


@pytest.mark.asyncio
async def test_timeout_during_delivery_stays_unknown_and_does_not_send_another_reply(agent, database, monkeypatch):
    from plugins.agent.session_admission import reserve

    key = QqSessionKey(99, None, 10)
    trigger_id = await _user(database, 700, "trigger", group=None)
    await database.qq_sessions.enqueue(key, trigger_id)
    admission = reserve(key)
    context = agent.AgentRequestContext(
        event=SimpleNamespace(self_id="99"), user_id="10", user_name="Alice", group_id=None,
        event_id=700, message_id=trigger_id, msg_time=1_000, text="trigger",
        quoted_images=[], images=[], videos=[], direct_mention=True, admission=admission,
    )

    async def send(*_args):
        await asyncio.Future()

    mock_send = AsyncMock(side_effect=send)
    response = {"response": {"messages": [AIMessage(content="reply")]}}
    monkeypatch.setattr(agent, "f_cognitive", SimpleNamespace(chat_agent=AsyncMock(return_value=response)))
    monkeypatch.setattr(agent, "send_messages", mock_send)
    monkeypatch.setattr(agent.EnvConfig, "AGENT_JOB_TIMEOUT_SECONDS", 1)
    try:
        await agent._process_queued_agent_request(context, [], bot=object())
    finally:
        await database.qq_sessions.interrupt(trigger_id)
        admission.release()

    mock_send.assert_awaited_once()
    assert _turn_row(database, trigger_id).status == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_stage", ["media", "normalization"])
async def test_ingress_orders_same_user_even_when_later_media_is_ready_first(agent, database, monkeypatch, slow_stage):
    from plugins.agent.message_normalizer import NormalizedMessage

    first_preparing = asyncio.Event()
    second_prepared = asyncio.Event()
    release_first = asyncio.Event()
    called = []

    async def extract(segments):
        text = segments[0]["data"]["text"]

        async def download():
            if text == "first" and slow_stage == "media":
                first_preparing.set()
                await release_first.wait()
            elif text == "second":
                second_prepared.set()

        return text, [download], [], []

    async def normalize(_bot, segments):
        if segments[0]["data"]["text"] == "first" and slow_stage == "normalization":
            first_preparing.set()
            await release_first.wait()
        return NormalizedMessage("", "[]", [])

    async def download(lazy, *_args):
        await lazy[0]()
        return [], [], []

    class Cognitive:
        async def chat_agent(self, _messages, **kwargs):
            called.append(kwargs["user_text"])
            return {"response": {"messages": [AIMessage(content="回复")]}}

    monkeypatch.setattr(agent, "f_cognitive", Cognitive())
    monkeypatch.setattr(agent, "get_bot", lambda: SimpleNamespace(send_group_message_reaction=AsyncMock()))
    monkeypatch.setattr(agent, "message_extract", extract)
    monkeypatch.setattr(agent, "message_gateway", AsyncMock(return_value=True))
    monkeypatch.setattr(agent, "normalize_segments", normalize)
    monkeypatch.setattr(agent, "download_media", download)
    monkeypatch.setattr(agent, "send_messages", AsyncMock(return_value=DeliveryResult(attempted=1, sent=1)))
    monkeypatch.setattr(agent, "sender_names_from_milky_message", lambda _message: ("user", None, None))

    def event(sequence, text, user=10):
        return SimpleNamespace(
            self_id="99", get_user_id=lambda: str(user), is_tome=lambda: True,
            data=SimpleNamespace(
                message_seq=sequence, group=SimpleNamespace(group_id=20),
                segments=[{"type": "text", "data": {"text": text}}],
            ),
        )

    async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
        tasks.create_task(agent.handle_common(event(1, "first")))
        await first_preparing.wait()
        tasks.create_task(agent.handle_common(event(2, "second")))
        await second_prepared.wait()
        await agent.handle_common(event(3, "another user", user=11))
        assert called == ["another user"]
        release_first.set()

    assert called == ["another user", "first", "second"]
    with Session(database.engine) as session:
        triggers = session.exec(select(Message.id).where(Message.role == "user").order_by(Message.id)).all()
    assert _turn_row(database, triggers[1]).previous_trigger_id == triggers[0]


@pytest.mark.asyncio
async def test_recall_hides_cached_quote_and_snapshot_even_when_notice_precedes_insert(agent, database):
    from nonebot.adapters.milky.event import MessageRecallEvent

    from plugins.agent.reply_context import MessageSource, load_message_context

    async def recall(sequence):
        await agent.handle_session_notice(MessageRecallEvent(time=1, self_id=99, data={
            "message_scene": "group", "peer_id": 20, "sender_id": 10, "message_seq": sequence,
            "operator_id": 10, "display_suffix": "",
        }))

    key = QqSessionKey(99, 20, 10)
    first = await _user(database, 800, "first")
    await database.qq_sessions.enqueue(key, first)
    await database.qq_sessions.start(key, first, 1_000, {"message_id": "800", "content": "first"}, limit=10)
    await database.qq_sessions.silent(first)
    await _user(database, 801, "撤回前补充")
    second = await _user(database, 802, "second")
    await database.qq_sessions.enqueue(key, second)
    await database.qq_sessions.start(key, second, 1_000, {"message_id": "802", "quote": {"message_id": "801"}}, limit=10)
    await database.qq_sessions.silent(second)
    await recall(801)
    await recall(803)
    late = await _user(database, 803, "撤回先到")
    assert not await database.qq_lifecycle.permits(99, 20, 10, late, 803)
    assert await database.select_by_msg_id(msg_id=801, group_id=20) is None
    assert await database.search_messages(group_id=20, user_id=None, content_query="撤回前补充") == []
    assert await load_message_context(object(), MessageSource("99", "group", 20), 801, database) == (None, [])
    third = await _user(database, 804, "third")
    await database.qq_sessions.enqueue(key, third)
    context = await database.qq_sessions.start(key, third, 1_000, {"message_id": "804"}, limit=10)
    assert context.own_messages == []
    previous = json.loads(context.history[-1].content)
    assert previous["own_messages"] == [] and "quote" not in previous["trigger"]


@pytest.mark.asyncio
async def test_rejoin_preserves_prior_turns_but_advances_group_background_boundary(agent, database):
    from nonebot.adapters.milky.event import GroupMemberDecreaseEvent, GroupMemberIncreaseEvent

    key = QqSessionKey(99, 20, 10)
    first = await _user(database, 900, "first")
    await database.qq_sessions.enqueue(key, first)
    await database.qq_sessions.start(key, first, 1_000, {"message_id": "900"}, limit=10)
    await database.qq_sessions.silent(first)
    await agent.handle_session_notice(GroupMemberDecreaseEvent(time=2, self_id=99, data={"group_id": 20, "user_id": 10}))
    await _user(database, 901, "during absence", user=20)
    pending = await _user(database, 902, "delayed trigger")
    assert not await database.qq_lifecycle.permits(99, 20, 10, pending, 902)
    await agent.handle_session_notice(GroupMemberIncreaseEvent(time=3, self_id=99, data={"group_id": 20, "user_id": 10}))
    await agent.handle_session_notice(GroupMemberDecreaseEvent(time=2, self_id=99, data={"group_id": 20, "user_id": 10}))
    assert not await database.qq_lifecycle.permits(99, 20, 10, pending, 902)
    await _user(database, 903, "after rejoin", user=20)
    second = await _user(database, 904, "second")
    assert await database.qq_lifecycle.permits(99, 20, 10, second, 904)
    await database.qq_sessions.enqueue(key, second)
    context = await database.qq_sessions.start(key, second, 1_000, {"message_id": "904"}, limit=10)
    assert context.history
    assert [item["content"] for item in context.group_background] == ["after rejoin"]


@pytest.mark.asyncio
async def test_raw_ingress_is_recoverable_only_until_turn_enqueued(database):
    key = QqSessionKey(99, 20, 10)

    async def observe():
        return await database.insert(1_000, 910, 10, 20, "Alice", "user", "trigger", bot_user_id=99, track_qq_ingress=True)

    initial = await observe()
    await MessageDatabase().qq_sessions.interrupt()
    retried = await observe()
    assert not retried.inserted and retried.resume_ingress and retried.message_id == initial.message_id
    assert await database.qq_sessions.enqueue(key, initial.message_id)
    assert not (await observe()).resume_ingress


@pytest.mark.asyncio
async def test_cancel_only_stops_request_owned_by_command_sender(agent, database, monkeypatch):
    from nonebot.exception import FinishedException

    from plugins.agent.session_admission import QQ_REQUEST_CANCELLED, reserve

    key = QqSessionKey(99, 20, 10)
    ready = asyncio.Event()
    replies = []

    async def finish(text):
        replies.append(text)
        raise FinishedException

    async def wait_for_work():
        entry = reserve(key, 150)
        ready.set()
        try:
            await asyncio.Future()
        finally:
            entry.release()

    monkeypatch.setattr(agent.common, "finish", finish)
    monkeypatch.setattr(agent, "get_bot", lambda: object())
    monkeypatch.setattr(agent, "sender_names_from_milky_message", lambda _message: ("user", None, None))

    async def send_cancel(user_id, command_sequence):
        event = SimpleNamespace(
            self_id="99", get_user_id=lambda: str(user_id),
            data=SimpleNamespace(message_seq=command_sequence, group=SimpleNamespace(group_id=20), segments=[
                {"type": "reply", "data": {"message_seq": 150}},
                {"type": "mention", "data": {"user_id": 10}},
                {"type": "text", "data": {"text": " /cancel"}},
            ]),
        )
        await agent.handle_common(event)

    task = asyncio.create_task(wait_for_work())
    await ready.wait()
    try:
        with pytest.raises(FinishedException):
            await send_cancel(11, 151)
        assert not task.done()
        with pytest.raises(FinishedException):
            await send_cancel(10, 152)
        with pytest.raises(asyncio.CancelledError) as failure:
            await task
        assert failure.value.args == (QQ_REQUEST_CANCELLED,)
        row = await _user(database, 150, "cancelled trigger")
        assert not await database.qq_lifecycle.permits(99, 20, 10, row, 150)
        assert replies == ["这条消息没有可取消的本人请求。", "已请求停止处理这条消息。"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_membership_notice_suppresses_running_group_reply(agent, database, monkeypatch):
    from nonebot.adapters.milky.event import GroupMemberDecreaseEvent

    key = QqSessionKey(99, 20, 10)
    trigger = await _user(database, 250, "trigger")
    started = asyncio.Event()
    release = asyncio.Event()

    async def chat(*_args, **_kwargs):
        started.set()
        await release.wait()
        return {"response": {"messages": [AIMessage(content="reply")]}}

    send = AsyncMock()
    monkeypatch.setattr(agent, "f_cognitive", SimpleNamespace(chat_agent=chat))
    monkeypatch.setattr(agent, "send_messages", send)
    async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
        turn = tasks.create_task(_run(agent, key, trigger, 250, "trigger"))
        await started.wait()
        await agent.handle_session_notice(GroupMemberDecreaseEvent(time=2, self_id=99, data={"group_id": 20, "user_id": 10}))
        release.set()

    assert not turn.result()
    assert _turn_row(database, trigger).status == "failed"
    send.assert_not_awaited()


def test_admission_caps_person_and_group():
    from plugins.agent.session_admission import MAX_PENDING_PER_GROUP, MAX_PENDING_PER_USER, reserve

    entries = []
    try:
        key = QqSessionKey(99, 20, 10)
        entries.extend(reserve(key) for _ in range(MAX_PENDING_PER_USER))
        assert all(entry is not None for entry in entries)
        assert reserve(key) is None
        entries.extend(reserve(QqSessionKey(99, 20, user)) for user in range(100, 100 + MAX_PENDING_PER_GROUP - len(entries)))
        assert all(entry is not None for entry in entries)
        assert reserve(QqSessionKey(99, 20, 50)) is None
        entries.append(reserve(QqSessionKey(99, 21, 10)))
        assert entries[-1] is not None
    finally:
        for entry in entries:
            if entry is not None:
                entry.release()

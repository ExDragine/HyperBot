"""Persist QQ dialogue turns independently of the agent graph."""

import json
import time
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import CheckConstraint, Engine, exists, update
from sqlalchemy.dialects.sqlite import insert
from sqlmodel import Field, Index, Session, SQLModel, col, select

from utils.agent_protocol import ChatMessage

from .engine import _run_database
from .models import MESSAGE_SOURCE_TYPE_NORMAL, Message, MessageInsertResult
from .qq_lifecycle import (
    QqIngress,
    cancelled_sequences,
    close_ingress,
    group_boundaries,
    recalled_sequences,
    visible_qq_message,
)
from .rows import _message_agent_payload

qq_session_metrics: Counter[str] = Counter()

_OWN_BATCH_CHARS = 8_000
_GROUP_BACKGROUND_CHARS = 4_000
_LINE_CHARS = 2_000


@dataclass(frozen=True, slots=True)
class QqSessionKey:
    bot_id: int
    group_id: int | None
    user_id: int


class QqSession(SQLModel, table=True):
    __tablename__ = "qq_sessions"
    __table_args__ = (
        CheckConstraint("group_scope_id >= 0"),
        Index("ux_qq_session_scope", "bot_id", "group_scope_id", "user_id", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    bot_id: int
    group_scope_id: int
    user_id: int


class QqSessionTurn(SQLModel, table=True):
    __tablename__ = "qq_session_turns"
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','sending','settled','silent','failed','unknown')"),
        Index("ix_qq_session_turns", "session_id", "trigger_message_id"),
    )

    trigger_message_id: int = Field(primary_key=True, foreign_key="message.id", ondelete="CASCADE")
    session_id: int = Field(foreign_key="qq_sessions.id", ondelete="CASCADE")
    previous_trigger_id: int | None = Field(default=None, foreign_key="qq_session_turns.trigger_message_id", ondelete="SET NULL")
    status: str = "queued"
    queued_at_ms: int
    input_json: str | None = None
    reply_message_id: int | None = Field(default=None, foreign_key="message.id", ondelete="SET NULL", unique=True)
    delivered_at_ms: int | None = None
    delivery_status: str | None = None


@dataclass(frozen=True, slots=True)
class QqTurnContext:
    history: list[ChatMessage]
    own_messages: list[dict[str, object]]
    group_background: list[dict[str, object]]
    later_replies: list[dict[str, int]]
    source_sequences: frozenset[str] = frozenset()


def _bounded_messages(rows, max_chars: int) -> list[dict[str, object]]:
    messages = []
    for row in rows:
        text = (row.model_content if row.model_content is not None else row.content)[:_LINE_CHARS]
        message = _message_agent_payload(row, content=text, include_reply_to=False)
        size = len(json.dumps(message, ensure_ascii=False))
        if size > max_chars:
            continue
        max_chars -= size
        messages.append(message)
    return list(reversed(messages))


def _session_history(session: Session, key: QqSessionKey, rows, trigger_time: int):
    history = []
    later_replies = []
    inputs = [(turn, reply, json.loads(turn.input_json)) for turn, reply in reversed(rows)]
    sequences = {
        str(item["message_id"])
        for _, _, content in inputs for item in _input_sources(content) if item.get("message_id") is not None
    }
    sequences.update(str(reply.msg_id) for _, reply, _ in inputs if reply is not None and reply.msg_id is not None)
    excluded = cancelled_sequences(session, key.bot_id, key.group_id, key.user_id, sequences)
    if key.group_id is not None:
        excluded.update(recalled_sequences(session, key.bot_id, key.group_id, sequences))
    for turn, reply, content in inputs:
        trigger = content["trigger"]
        if str(trigger.get("message_id")) in excluded:
            continue
        if str((trigger.get("quote") or {}).get("message_id")) in excluded:
            trigger.pop("quote", None)
        content["own_messages"] = [item for item in content["own_messages"] if str(item.get("message_id")) not in excluded]
        if reply is not None and str(reply.msg_id) in excluded:
            reply = None
        content["turn_status"] = turn.status
        if reply is not None:
            receipt = {"reply_row_id": reply.id, "delivered_at_ms": turn.delivered_at_ms}
            content["reply_receipt"] = receipt
            if turn.delivered_at_ms is not None and turn.delivered_at_ms >= trigger_time:
                later_replies.append(receipt)
        history.append(ChatMessage(
            role="user", content=json.dumps(content, ensure_ascii=False),
            state_id=f"qq:session:turn:{turn.trigger_message_id}:input",
        ))
        if reply is not None:
            history.append(ChatMessage(role="assistant", content=reply.content, state_id=f"qq:session:reply:{reply.id}"))
    return history, later_replies, sequences - excluded


def _input_sources(content):
    trigger = content["trigger"]
    return [trigger, trigger.get("quote") or {}, *content["own_messages"]]


def persist_session_reply(session: Session, trigger_id: int, message: Message, delivery_status: str) -> MessageInsertResult:
    """Link confirmed text and its settled turn in the caller's transaction."""
    if message.role != "assistant" or delivery_status not in {"delivered", "partial"}:
        raise ValueError("Only confirmed QQ assistant text can settle a turn")
    _transition(session, trigger_id, "sending", "settled", delivery_status=delivery_status, delivered_at_ms=message.time)
    existing = None
    if message.platform_key is not None:
        existing = session.exec(select(Message).where(Message.platform_key == message.platform_key)).first()
    reply = existing or message
    if existing is None:
        session.add(reply)
        session.flush()
    else:
        # A self echo may arrive before the transport receipt.
        if reply.role != "assistant":
            raise ValueError("Confirmed QQ reply conflicts with an inbound message")
        reply.content = message.content
        reply.model_content = None
    session.get(QqSessionTurn, trigger_id).reply_message_id = reply.id
    return MessageInsertResult(message_id=reply.id, time=reply.time, inserted=existing is None)


def _transition(session: Session, trigger_id: int, expected: str, status: str, **values) -> None:
    result = session.execute(
        update(QqSessionTurn)
        .where(QqSessionTurn.trigger_message_id == trigger_id, QqSessionTurn.status == expected)
        .values(status=status, **values)
    )
    if result.rowcount != 1:
        raise ValueError(f"QQ turn must be {expected} to transition to {status}")


class QqSessionStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def enqueue(self, key: QqSessionKey, trigger_id: int) -> bool:
        def write():
            with Session(self.engine) as session:
                scope = {"bot_id": key.bot_id, "group_scope_id": key.group_id or 0, "user_id": key.user_id}
                session.execute(insert(QqSession).values(**scope).on_conflict_do_nothing())
                session_id = session.exec(select(QqSession.id).filter_by(**scope)).one()
                result = session.execute(insert(QqSessionTurn).values(
                    session_id=session_id, trigger_message_id=trigger_id, queued_at_ms=_now_ms(), status="queued",
                ).on_conflict_do_nothing(index_elements=["trigger_message_id"]))
                close_ingress(session, session.get(Message, trigger_id).platform_key)
                session.commit()
                return result.rowcount == 1

        return await _run_database(self.engine, write)

    async def start(
        self, key: QqSessionKey, trigger_id: int, trigger_time: int, current: dict[str, object], *, limit: int,
    ) -> QqTurnContext:
        def write():
            with Session(self.engine, expire_on_commit=False) as session:
                _transition(session, trigger_id, "queued", "running")
                turn = session.get(QqSessionTurn, trigger_id)
                prior = session.exec(
                    select(QqSessionTurn, Message)
                    .outerjoin(Message, QqSessionTurn.reply_message_id == Message.id)
                    .where(QqSessionTurn.session_id == turn.session_id, QqSessionTurn.trigger_message_id < trigger_id,
                           col(QqSessionTurn.input_json).is_not(None))
                    .order_by(col(QqSessionTurn.trigger_message_id).desc()).limit(limit)
                ).all()
                previous = prior[0][0].trigger_message_id if prior else None
                own, public = self._interval(session, key, previous, trigger_id, limit)
                turn.previous_trigger_id = previous
                turn.input_json = json.dumps({"own_messages": own, "trigger": current}, ensure_ascii=False)
                queued_at = turn.queued_at_ms
                history, later_replies, sources = _session_history(session, key, prior, trigger_time)
                session.commit()
                sources.update(str(item["message_id"]) for item in [*own, *public] if item.get("message_id") is not None)
                return QqTurnContext(history, own, public, later_replies, frozenset(sources)), queued_at

        context, queued_at = await _run_database(self.engine, write)
        qq_session_metrics["started"] += 1
        qq_session_metrics["continued"] += bool(context.history)
        qq_session_metrics["queue_ms"] += max(0, _now_ms() - queued_at)
        qq_session_metrics["context_chars"] += sum(len(message.content) for message in context.history)
        qq_session_metrics["context_chars"] += len(json.dumps([context.own_messages, context.group_background], ensure_ascii=False))
        return context

    @staticmethod
    def _interval(session: Session, key: QqSessionKey, previous: int | None, trigger_id: int, limit: int):
        if previous is None:
            return [], []
        rows = select(Message).where(
            Message.bot_user_id == key.bot_id, Message.group_id == key.group_id,
            Message.role == "user", Message.source_type == MESSAGE_SOURCE_TYPE_NORMAL,
            Message.id > previous, Message.id < trigger_id,
            visible_qq_message(),
            ~exists().where(QqIngress.platform_key == Message.platform_key, QqIngress.cancelled),
        ).order_by(col(Message.id).desc()).limit(limit)
        public = []
        if key.group_id is not None:
            floor = max((item.message_floor for item in group_boundaries(session, key.bot_id, key.group_id, key.user_id)), default=0)
            rows = rows.where(Message.id > floor)
            public = _bounded_messages(session.exec(rows.where(Message.sender_user_id != key.user_id)), _GROUP_BACKGROUND_CHARS)
        own = _bounded_messages(session.exec(rows.where(Message.sender_user_id == key.user_id)), _OWN_BATCH_CHARS)
        return own, public

    async def begin_delivery(self, trigger_id: int) -> None:
        await self._set_status(trigger_id, "running", "sending")

    async def silent(self, trigger_id: int) -> None:
        await self._set_status(trigger_id, "running", "silent")

    async def finish_without_text(self, trigger_id: int, delivery_status: str, delivered_at: int) -> None:
        status = {"delivered": "settled", "partial": "settled", "unknown": "unknown", "failed": "failed"}[delivery_status]
        await self._set_status(trigger_id, "sending", status, delivery_status=delivery_status,
                               delivered_at_ms=delivered_at if status == "settled" else None)

    async def _set_status(self, trigger_id: int, expected: str, status: str, **values) -> None:
        def write():
            with Session(self.engine) as session:
                _transition(session, trigger_id, expected, status, **values)
                session.commit()

        await _run_database(self.engine, write)
        qq_session_metrics[status] += 1

    async def interrupt(self, trigger_id: int | None = None) -> None:
        """Close unfinished work without replaying the graph or the transport."""
        def write():
            with Session(self.engine) as session:
                statement = update(QqSessionTurn)
                if trigger_id is not None:
                    statement = statement.where(QqSessionTurn.trigger_message_id == trigger_id)
                failed = session.execute(statement.where(col(QqSessionTurn.status).in_(("queued", "running")))
                                         .values(status="failed")).rowcount
                unknown = session.execute(statement.where(QqSessionTurn.status == "sending")
                                          .values(status="unknown", delivery_status="unknown")).rowcount
                session.commit()
                return failed, unknown

        failed, unknown = await _run_database(self.engine, write)
        qq_session_metrics["failed"] += failed
        qq_session_metrics["unknown"] += unknown


def _now_ms() -> int:
    return time.time_ns() // 1_000_000

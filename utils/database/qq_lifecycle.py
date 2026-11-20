"""Durable QQ group recall and membership boundaries."""

from sqlalchemy import Engine, delete, exists, func
from sqlalchemy.dialects.sqlite import insert
from sqlmodel import Field, Session, SQLModel, col, select

from .engine import _run_database
from .models import Message, MessageAttachment
from .schema import platform_message_key


class QqRecall(SQLModel, table=True):
    __tablename__ = "qq_message_recalls"

    platform_key: str = Field(primary_key=True)


class QqIngress(SQLModel, table=True):
    __tablename__ = "qq_session_ingress"

    platform_key: str = Field(primary_key=True)
    cancelled: bool = False


class QqMembership(SQLModel, table=True):
    __tablename__ = "qq_group_membership"

    bot_id: int = Field(primary_key=True)
    group_id: int = Field(primary_key=True)
    user_id: int = Field(primary_key=True)  # 0 represents the bot's access to the whole group.
    active: bool
    event_time: int
    message_floor: int


def visible_qq_message():
    return ~exists().where(QqRecall.platform_key == Message.platform_key)


def close_ingress(session: Session, platform_key: str | None) -> None:
    if platform_key is not None:
        session.execute(delete(QqIngress).where(QqIngress.platform_key == platform_key, ~col(QqIngress.cancelled)))


def recall_key(bot_id: int, group_id: int, sequence: int | str) -> str:
    return platform_message_key(msg_id=sequence, user_id=None, group_id=group_id, bot_user_id=bot_id)


def recalled_sequences(session: Session, bot_id: int, group_id: int, sequences) -> set[str]:
    keys = {recall_key(bot_id, group_id, sequence) for sequence in sequences if sequence is not None}
    if not keys:
        return set()
    found = session.exec(select(QqRecall.platform_key).where(col(QqRecall.platform_key).in_(keys))).all()
    return {key.rsplit(":", 1)[-1] for key in found}


def cancelled_sequences(session: Session, bot_id: int, group_id: int | None, user_id: int, sequences) -> set[str]:
    keys = {platform_message_key(msg_id=sequence, user_id=user_id, group_id=group_id, bot_user_id=bot_id) for sequence in sequences}
    if not keys:
        return set()
    found = session.exec(select(QqIngress.platform_key).where(col(QqIngress.platform_key).in_(keys), QqIngress.cancelled)).all()
    return {key.rsplit(":", 1)[-1] for key in found}


def group_boundaries(session: Session, bot_id: int, group_id: int, user_id: int):
    return session.exec(select(QqMembership).where(
        QqMembership.bot_id == bot_id, QqMembership.group_id == group_id,
        col(QqMembership.user_id).in_((0, user_id)),
    )).all()


class QqLifecycleStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def recall(self, bot_id: int, group_id: int, sequence: int) -> None:
        def write():
            with Session(self.engine) as session:
                session.execute(insert(QqRecall).values(
                    platform_key=recall_key(bot_id, group_id, sequence),
                ).on_conflict_do_nothing())
                session.commit()

        await _run_database(self.engine, write)

    async def recalled(self, bot_id: int, group_id: int, sequences) -> set[str]:
        def read():
            with Session(self.engine) as session:
                return recalled_sequences(session, bot_id, group_id, sequences)

        return await _run_database(self.engine, read)

    async def recalled_attachment_paths(self, bot_id: int, group_id: int) -> list[str]:
        def read():
            with Session(self.engine) as session:
                return list(session.exec(
                    select(MessageAttachment.virtual_path).join(Message, MessageAttachment.message_id == Message.id)
                    .where(Message.bot_user_id == bot_id, Message.group_id == group_id, ~visible_qq_message())
                    .distinct(),
                ).all())

        return await _run_database(self.engine, read)

    async def finish_ingress(self, platform_key: str) -> None:
        def write():
            with Session(self.engine) as session:
                close_ingress(session, platform_key)
                session.commit()

        await _run_database(self.engine, write)

    async def cancel_ingress(self, platform_key: str) -> None:
        def write():
            with Session(self.engine) as session:
                session.execute(insert(QqIngress).values(platform_key=platform_key, cancelled=True).on_conflict_do_update(
                    index_elements=["platform_key"], set_={"cancelled": True},
                ))
                session.commit()

        await _run_database(self.engine, write)

    async def membership(self, bot_id: int, group_id: int, user_id: int, event_time: int, *, active: bool) -> bool:
        def write():
            with Session(self.engine) as session:
                current = session.get(QqMembership, (bot_id, group_id, user_id))
                if current is not None and (current.event_time > event_time or (
                    current.event_time == event_time and current.active == active
                )):
                    return False
                floor = session.exec(select(func.max(Message.id)).where(
                    Message.bot_user_id == bot_id, Message.group_id == group_id,
                )).one() or 0
                statement = insert(QqMembership).values(
                    bot_id=bot_id, group_id=group_id, user_id=user_id,
                    event_time=event_time, active=active, message_floor=floor,
                )
                updated = session.execute(statement.on_conflict_do_update(
                    index_elements=["bot_id", "group_id", "user_id"],
                    set_={"event_time": event_time, "active": active, "message_floor": floor},
                    where=QqMembership.event_time <= event_time,
                ))
                session.commit()
                return updated.rowcount == 1

        return await _run_database(self.engine, write)

    async def permits(self, bot_id: int, group_id: int, user_id: int, message_id: int, sequence: int) -> bool:
        def read():
            with Session(self.engine) as session:
                boundaries = group_boundaries(session, bot_id, group_id, user_id)
                key = recall_key(bot_id, group_id, sequence)
                ingress = session.get(QqIngress, key)
                return all(item.active and message_id > item.message_floor for item in boundaries) and (
                    session.get(QqRecall, key) is None and not (ingress is not None and ingress.cancelled)
                )

        return await _run_database(self.engine, read)

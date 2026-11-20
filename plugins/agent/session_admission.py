"""Bound the QQ work awaiting preparation, execution and delivery."""

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field

from utils.database.qq_sessions import QqSessionKey

MAX_PENDING_PER_USER = 4
MAX_PENDING_PER_GROUP = 16
QQ_REQUEST_CANCELLED = object()

_pending: dict[QqSessionKey, deque[QqAdmission]] = {}
_group_counts: Counter[tuple[int, int]] = Counter()


@dataclass(slots=True, eq=False)
class QqAdmission:
    key: QqSessionKey
    sequence: int | None
    task: asyncio.Task | None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    sources: frozenset[str] = frozenset()
    running: bool = False
    cancelled: bool = False

    async def wait(self) -> None:
        await self.ready.wait()

    def cancel(self) -> None:
        if self.cancelled:
            return
        self.cancelled = True
        if self.task is not None:
            self.task.cancel(QQ_REQUEST_CANCELLED)

    def release(self) -> None:
        queue = _pending[self.key]
        queue.remove(self)
        if queue:
            queue[0].ready.set()
        else:
            del _pending[self.key]
        if self.key.group_id is not None:
            group = (self.key.bot_id, self.key.group_id)
            _group_counts[group] -= 1
            if not _group_counts[group]:
                del _group_counts[group]


def find_pending(key: QqSessionKey, sequence: int) -> QqAdmission | None:
    return next((entry for entry in _pending.get(key, ()) if entry.sequence == sequence), None)


def reserve(key: QqSessionKey, sequence: int | None = None) -> QqAdmission | None:
    queue = _pending.get(key)
    if queue is not None and len(queue) >= MAX_PENDING_PER_USER:
        return None
    group = (key.bot_id, key.group_id)
    if key.group_id is not None and _group_counts[group] >= MAX_PENDING_PER_GROUP:
        return None
    entry = QqAdmission(key, sequence, asyncio.current_task() if sequence is not None else None)
    if queue is None:
        queue = _pending[key] = deque()
        entry.ready.set()
    queue.append(entry)
    if key.group_id is not None:
        _group_counts[group] += 1
    return entry


def cancel_group_work(bot_id: int, group_id: int, *, user_id: int | None = None, sequence: int | None = None) -> None:
    for key, queue in _pending.items():
        if (key.bot_id, key.group_id) != (bot_id, group_id) or (user_id is not None and user_id != key.user_id):
            continue
        for entry in queue:
            if sequence is None or entry.sequence == sequence or entry.running:
                entry.cancel()

"""Single-process bounded SSE event buffer and background task registry."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import time
from typing import AsyncIterator, Awaitable, Callable

from app.schemas.events import RunEvent


TERMINAL_EVENTS = {"run.completed", "run.partial", "run.failed", "run.cancelled", "run.timed_out"}


@dataclass
class _RunChannel:
    events: deque[RunEvent]
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    next_id: int = 1
    terminal: bool = False
    touched_at: float = field(default_factory=time.monotonic)


class RunManager:
    def __init__(self, *, buffer_size: int = 200, retention_seconds: int = 600,
                 heartbeat_seconds: float = 15.0) -> None:
        self.buffer_size = buffer_size
        self.retention_seconds = retention_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._channels: dict[str, _RunChannel] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def register(self, run_id: str) -> None:
        self._cleanup()
        self._channels.setdefault(run_id, _RunChannel(deque(maxlen=self.buffer_size)))

    def start(self, run_id: str, worker: Callable[[], Awaitable[None]]) -> None:
        self.register(run_id)
        if run_id not in self._tasks or self._tasks[run_id].done():
            self._tasks[run_id] = asyncio.create_task(worker())

    async def publish(self, run_id: str, event: str, data: dict) -> RunEvent:
        self.register(run_id)
        channel = self._channels[run_id]
        async with channel.condition:
            item = RunEvent(channel.next_id, event, data)
            channel.next_id += 1
            channel.events.append(item)
            channel.touched_at = time.monotonic()
            if event in TERMINAL_EVENTS:
                channel.terminal = True
            channel.condition.notify_all()
            return item

    async def events(self, run_id: str, last_event_id: int = 0) -> AsyncIterator[str]:
        self.register(run_id)
        channel = self._channels[run_id]
        cursor = last_event_id
        while True:
            ready = [item for item in channel.events if item.id > cursor]
            for item in ready:
                cursor = item.id
                yield item.encode()
            if channel.terminal and not any(item.id > cursor for item in channel.events):
                return
            try:
                async with channel.condition:
                    await asyncio.wait_for(
                        channel.condition.wait(), timeout=self.heartbeat_seconds
                    )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"

    def _cleanup(self) -> None:
        threshold = time.monotonic() - self.retention_seconds
        for run_id in [
            key for key, value in self._channels.items()
            if value.terminal and value.touched_at < threshold
        ]:
            self._channels.pop(run_id, None)
            self._tasks.pop(run_id, None)

"""Server-Sent Events generator (plain sync generator so it is testable without HTTP)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from starlette.concurrency import run_in_threadpool

Refresh = Callable[[int], tuple[list[tuple[int, str, dict[str, Any]]], bool]]


def frame(event_id: int, event: str, data: dict[str, Any]) -> str:
    return f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def heartbeat_frame(now: float | None = None) -> str:
    """Id-less named event: EventSource never surfaces comments to JS, so liveness needs a real
    event. No ``id:`` line, so it never moves ``Last-Event-ID``."""
    ts = time.time() if now is None else now
    return f"event: heartbeat\ndata: {json.dumps({'ts': ts}, separators=(',', ':'))}\n\n"


def stream(
    refresh: Refresh,
    last_id: int,
    *,
    heartbeat_s: float,
    poll_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """``refresh(last_id)`` -> (events with id > last_id, finished); ends after ``done``."""
    last_sent = clock()
    while True:
        events, finished = refresh(last_id)
        for event_id, event, data in events:
            yield frame(event_id, event, data)
            last_id = event_id
            last_sent = clock()
        if finished:
            return
        if clock() - last_sent >= heartbeat_s:
            yield heartbeat_frame()
            last_sent = clock()
        sleep(poll_s)


class StreamLimiter:
    """Caps concurrent SSE streams globally and per client address."""

    def __init__(self, max_total: int, max_per_ip: int) -> None:
        self.max_total, self.max_per_ip = max_total, max_per_ip
        self._per_ip: Counter[str] = Counter()
        self._lock = threading.Lock()

    def acquire(self, ip: str) -> bool:
        with self._lock:
            if sum(self._per_ip.values()) >= self.max_total or self._per_ip[ip] >= self.max_per_ip:
                return False
            self._per_ip[ip] += 1
            return True

    def release(self, ip: str) -> None:
        with self._lock:
            if self._per_ip[ip] > 0:
                self._per_ip[ip] -= 1
            if self._per_ip[ip] == 0:
                del self._per_ip[ip]

    def active(self) -> int:
        with self._lock:
            return sum(self._per_ip.values())


async def astream(
    refresh: Refresh,
    last_id: int,
    *,
    heartbeat_s: float,
    poll_s: float,
    is_disconnected: Callable[[], Awaitable[bool]],
    on_close: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Async twin of :func:`stream`: no thread is held while idle (``refresh`` runs in the
    threadpool only for the instant of a poll) and it stops when the client goes away."""
    last_sent = clock()
    try:
        while True:
            events, finished = await run_in_threadpool(refresh, last_id)
            for event_id, event, data in events:
                yield frame(event_id, event, data)
                last_id = event_id
                last_sent = clock()
            if finished:
                return
            if await is_disconnected():
                return
            if clock() - last_sent >= heartbeat_s:
                yield heartbeat_frame()
                last_sent = clock()
            await asyncio.sleep(poll_s)
    finally:
        on_close()

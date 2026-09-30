"""Server-Sent Events generator (plain sync generator so it is testable without HTTP)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

Refresh = Callable[[int], tuple[list[tuple[int, str, dict[str, Any]]], bool]]


def frame(event_id: int, event: str, data: dict[str, Any]) -> str:
    return f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


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
            yield ": heartbeat\n\n"
            last_sent = clock()
        sleep(poll_s)

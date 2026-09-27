# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for network integration tests.

* :func:`wait_for_text_count` — poll a channel's WAL until ``EV_TEXT``
  count reaches a threshold; conversation/discussion adapters have no
  terminal event to await on.
* :func:`wait_for_delivery` — poll one recipient's captured envelopes
  until a given text lands; for per-recipient delivery assertions.
* :func:`adapter_state` — a channel's folded state, as the adapter's
  state class.
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import TypeVar

from ag2.network import EV_TEXT, Envelope, Hub

__all__ = ("_MockClock", "adapter_state", "wait_for_delivery", "wait_for_text_count")

TState = TypeVar("TState")


class _MockClock:
    """Controllable clock — returns a stored ISO timestamp; ``advance``
    pushes it forward by ``seconds``."""

    def __init__(self, start: str = "2026-01-01T00:00:00+00:00") -> None:
        self._now = datetime.fromisoformat(start)
        if self._now.tzinfo is None:
            self._now = self._now.replace(tzinfo=timezone.utc)

    def __call__(self) -> str:
        return self._now.isoformat()

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


async def wait_for_text_count(
    hub: Hub,
    channel_id: str,
    expected: int,
    *,
    timeout: float = 5.0,
) -> list[Envelope]:
    """Poll WAL until at least ``expected`` ``EV_TEXT`` envelopes appear.

    Used by adapters that have no terminal event to await on
    (``conversation``, ``discussion``). Auto-driven LLM exchanges
    settle when one side returns an empty body; tests poll rather than
    assume timing.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel_id)
        if sum(1 for e in wal if e.event_type == EV_TEXT) >= expected:
            return wal
        await asyncio.sleep(0.02)
    raise asyncio.TimeoutError(f"channel {channel_id!r} never reached {expected} EV_TEXT envelopes")


async def wait_for_delivery(
    received: "Sequence[Envelope]",
    text: str,
    *,
    timeout: float = 5.0,
) -> None:
    """Poll a capture list until an ``EV_TEXT`` envelope carrying ``text`` lands.

    The hub fans out to each recipient in turn, so one participant's copy can
    land several event-loop turns after another's; polling avoids encoding a
    guess about the slowest recipient.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if any(e.event_type == EV_TEXT and e.event_data.get("text") == text for e in received):
            return
        await asyncio.sleep(0.02)
    raise asyncio.TimeoutError(f"no EV_TEXT envelope carrying {text!r} was delivered within {timeout}s")


def adapter_state(hub: Hub, channel_id: str, kind: type[TState]) -> TState:
    """Read ``channel_id``'s folded state through ``Hub.adapter_state``, as ``kind``.

    The hub answers ``object``; a state of another class fails here rather than at an attribute read.
    """
    state = hub.adapter_state(channel_id)
    assert isinstance(state, kind), f"channel {channel_id!r} holds {state!r}, not a {kind.__name__}"
    return state

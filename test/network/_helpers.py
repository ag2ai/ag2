# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for network integration tests.

* :class:`ScriptedConfig` — ``ModelConfig`` whose script persists across
  multiple ``create()`` calls, unlike ``ag2.testing.TestConfig``
  which resets its iterator each turn (breaks multi-turn LLM tests).
* :func:`wait_for_text_count` — poll a channel's WAL until ``EV_TEXT``
  count reaches a threshold; conversation/discussion adapters have no
  terminal event to await on.
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from ag2 import Context
from ag2.config import LLMClient, ModelConfig
from ag2.events import BaseEvent, ModelMessage, ModelResponse
from ag2.network import EV_TEXT, Envelope, Hub

__all__ = ("ScriptedConfig", "_MockClock", "wait_for_text_count")


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


class ScriptedConfig(ModelConfig):
    """Test config whose reply script persists across ``create()`` calls.

    ``Agent.ask`` calls ``config.create()`` once per turn; the standard
    ``TestConfig`` builds a fresh iterator each time, so a single agent
    keeps replaying its first scripted reply on every turn. This wrapper
    feeds one shared cursor into every client it produces, so a single
    agent can answer N successive turns deterministically.

    Exhausted scripts return ``""`` — the network's default handler
    treats an empty reply body as "don't send," which halts auto-driven
    LLM exchanges cleanly.
    """

    def __init__(self, *replies: str) -> None:
        self._replies: list[str] = list(replies)
        self._cursor = 0

    def copy(self) -> "ScriptedConfig":
        return self

    def create(self) -> "_ScriptedClient":
        return _ScriptedClient(self)

    def create_files_client(self) -> None:
        raise NotImplementedError("ScriptedConfig has no Files API")

    def _next_reply(self) -> str:
        if self._cursor >= len(self._replies):
            return ""
        reply = self._replies[self._cursor]
        self._cursor += 1
        return reply


class _ScriptedClient(LLMClient):
    def __init__(self, config: ScriptedConfig) -> None:
        self._config = config

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        text = self._config._next_reply()
        message = ModelMessage(text)
        await context.send(message)
        return ModelResponse(message)


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

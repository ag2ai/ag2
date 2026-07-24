# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""In-process test doubles for ACP-backed agents.

``fake_acp_config`` wires an :class:`~.config.ACPConfig` to a scripted, in-process
agent so tests can drive the public ``Agent.run`` path without spawning a real CLI
subprocess. Each :class:`ACPTurn` describes one ``session/prompt``: the
``session/update`` notifications the agent emits and the resulting stop reason.

This module imports ``acp`` and is only usable with the ``acp`` extra installed;
keep it out of the extra-free :mod:`ag2.testing`.
"""

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import acp
from acp import schema

from .config import ACPConfig
from .types import SessionUpdate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = (
    "ACPTurn",
    "fake_acp_config",
)


@dataclass
class ACPTurn:
    """One scripted ``session/prompt`` turn.

    Attributes:
        updates: ``session/update`` notifications the agent emits during the turn.
        stop_reason: ``stop_reason`` of the resulting ``PromptResponse``.
        usage: Token usage reported for the turn (``None`` => unreported).
        hang: When ``True`` the turn blocks until ``session/cancel`` (then returns
            ``stop_reason="cancelled"``) — used to exercise ``turn_timeout``.
    """

    updates: Sequence[SessionUpdate] = field(default_factory=tuple)
    stop_reason: str = "end_turn"
    usage: "schema.Usage | None" = None
    hang: bool = False


class _FakeConnection:
    """Minimal ``ClientSideConnection`` stand-in that drives the bridge in-process.

    ``prompt`` replays one :class:`ACPTurn`'s updates back through the bound client
    (the bridge) exactly as a real agent's ``session/update`` callbacks would.
    """

    def __init__(
        self,
        client: acp.Client,
        turns: Iterator[ACPTurn],
        config_options: "Sequence[schema.SessionConfigOptionSelect] | None" = None,
        config_option_calls: "list[tuple[str, str | bool]] | None" = None,
    ) -> None:
        self._client = client
        self._turns = turns
        self._config_options = list(config_options or [])
        self._cancelled = asyncio.Event()
        self.closed = False
        self.config_option_calls: list[tuple[str, str | bool]] = (
            config_option_calls if config_option_calls is not None else []
        )

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        return schema.InitializeResponse(protocol_version=acp.PROTOCOL_VERSION)

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        return schema.NewSessionResponse(
            session_id="fake-session-1",
            config_options=self._config_options or None,
        )

    async def set_config_option(self, *, session_id: str, config_id: str, value: Any, **kwargs: Any) -> None:
        self.config_option_calls.append((config_id, value))

    async def cancel(self, **kwargs: Any) -> None:
        self._cancelled.set()

    async def prompt(self, *, session_id: str, **kwargs: Any) -> schema.PromptResponse:
        turn = next(self._turns)
        if turn.hang:
            await self._cancelled.wait()
            self._cancelled.clear()
            return schema.PromptResponse(stop_reason="cancelled")
        for update in turn.updates:
            await self._client.session_update(session_id=session_id, update=update)
        return schema.PromptResponse(stop_reason=turn.stop_reason, usage=turn.usage)


def fake_acp_config(
    *turns: ACPTurn,
    config_options: "Sequence[schema.SessionConfigOptionSelect] | None" = None,
    config_option_calls: "list[tuple[str, str | bool]] | None" = None,
    **overrides: Any,
) -> ACPConfig:
    """Build an :class:`ACPConfig` backed by an in-process scripted agent.

    No subprocess is spawned: each ``Agent.run`` model-turn consumes one ``turns``
    entry in order. ``config_options`` are advertised in the ``session/new``
    response (the agent's model picker et al.); ``session/set_config_option``
    calls are appended to the caller-supplied ``config_option_calls`` list as
    ``(config_id, value)`` tuples. ``overrides`` are forwarded to ``ACPConfig``
    (e.g. ``permission_policy=...``, ``turn_timeout=...``).
    """
    config = ACPConfig(**overrides)
    script = list(turns)

    @asynccontextmanager
    async def connect(client: acp.Client) -> "AsyncIterator[tuple[_FakeConnection, None]]":
        conn = _FakeConnection(
            client, iter(script), config_options=config_options, config_option_calls=config_option_calls
        )
        try:
            yield conn, None
        finally:
            conn.closed = True

    config._connect = connect
    return config

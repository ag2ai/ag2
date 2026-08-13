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
import socket
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import acp
from acp import schema

from .config import ACPConfig
from .types import SessionUpdate

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from acp.core import ClientSideConnection

    from ag2.context import StreamId

    from .agent import ACPAgent
    from .config import ConnectHook
    from .session import ACPSession

__all__ = (
    "ACPTurn",
    "FakeACPConfig",
    "RecordingClient",
    "connect",
    "duplex",
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
        on_prompt: Awaited at the start of the turn, before ``updates`` replay —
            lets a test act as the CLI agent mid-turn (e.g. call the MCP gateway).
    """

    updates: Sequence[SessionUpdate] = field(default_factory=tuple)
    stop_reason: str = "end_turn"
    usage: "schema.Usage | None" = None
    hang: bool = False
    on_prompt: "Callable[[], Awaitable[None]] | None" = None


class _FakeConnection:
    """Minimal ``ClientSideConnection`` stand-in that drives the bridge in-process.

    ``prompt`` replays one :class:`ACPTurn`'s updates back through the bound client
    (the bridge) exactly as a real agent's ``session/update`` callbacks would.
    """

    def __init__(
        self,
        client: acp.Client,
        turns: Iterator[ACPTurn],
        *,
        agent_capabilities: "schema.AgentCapabilities | None" = None,
        config_options: "Sequence[schema.SessionConfigOptionSelect] | None" = None,
        config_option_calls: "list[tuple[str, str | bool]] | None" = None,
    ) -> None:
        self._client = client
        self._turns = turns
        self._config_options = list(config_options or [])
        self._cancelled = asyncio.Event()
        self._agent_capabilities = agent_capabilities
        self.new_session_kwargs: dict[str, Any] | None = None
        self.closed = False
        self.config_option_calls: list[tuple[str, str | bool]] = (
            config_option_calls if config_option_calls is not None else []
        )

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        return schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=self._agent_capabilities,
        )

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        self.new_session_kwargs = kwargs
        return schema.NewSessionResponse(
            session_id="fake-session-1",
            config_options=self._config_options or None,
        )

    async def set_config_option(
        self, *, session_id: str, config_id: str, value: Any, **kwargs: Any
    ) -> schema.SetSessionConfigOptionResponse:
        """Record the call and echo back the option set with ``value`` applied.

        The real ``set_config_option`` returns the agent's full, updated option
        list — that response is how a caller could tell an agent accepted the
        call but ignored it. Returning ``None`` here would let such a bug pass
        unnoticed in tests.
        """
        self.config_option_calls.append((config_id, value))
        self._config_options = [
            option.model_copy(update={"current_value": value}) if option.id == config_id else option
            for option in self._config_options
        ]
        return schema.SetSessionConfigOptionResponse(config_options=list(self._config_options))

    async def cancel(self, **kwargs: Any) -> None:
        self._cancelled.set()

    async def prompt(self, *, session_id: str, **kwargs: Any) -> schema.PromptResponse:
        turn = next(self._turns)
        if turn.on_prompt is not None:
            await turn.on_prompt()
        if turn.hang:
            await self._cancelled.wait()
            self._cancelled.clear()
            return schema.PromptResponse(stop_reason="cancelled")
        for update in turn.updates:
            await self._client.session_update(session_id=session_id, update=update)
        return schema.PromptResponse(stop_reason=turn.stop_reason, usage=turn.usage)


@dataclass(slots=True)
class FakeACPConfig(ACPConfig):
    """:class:`ACPConfig` bound to the scripted in-process agent.

    Adds public read-only views of the run-scoped state so tests can assert on
    session lifecycle (leaks, teardown) without reaching into private fields.
    """

    @property
    def sessions(self) -> "dict[StreamId, ACPSession]":
        """Live sessions keyed by stream id (empty once ``aclose()`` ran)."""
        return self._sessions

    @property
    def connect(self) -> "ConnectHook":
        """The in-process connection opener, for driving ``ACPSession.ensure`` directly."""
        assert self._connect is not None
        return self._connect


def fake_acp_config(
    *turns: ACPTurn,
    agent_capabilities: "schema.AgentCapabilities | None" = None,
    config_options: "Sequence[schema.SessionConfigOptionSelect] | None" = None,
    config_option_calls: "list[tuple[str, str | bool]] | None" = None,
    **overrides: Any,
) -> FakeACPConfig:
    """Build an :class:`ACPConfig` backed by an in-process scripted agent.

    No subprocess is spawned: each ``Agent.run`` model-turn consumes one ``turns``
    entry in order. ``overrides`` are forwarded to ``ACPConfig`` (e.g.
    ``permission_policy=...``, ``turn_timeout=...``). ``agent_capabilities``
    shapes the fake's ``initialize`` response; by default it advertises HTTP MCP
    support like the real Claude Code / Codex / OpenCode adapters do.
    ``config_options`` are advertised in the ``session/new`` response (the
    agent's model picker et al.); ``session/set_config_option`` calls are
    appended to the caller-supplied ``config_option_calls`` list as
    ``(config_id, value)`` tuples.
    """
    if agent_capabilities is None:
        agent_capabilities = schema.AgentCapabilities(mcp_capabilities=schema.McpCapabilities(http=True, sse=True))
    config = FakeACPConfig(**overrides)
    script = list(turns)

    @asynccontextmanager
    async def connect(client: acp.Client) -> "AsyncGenerator[tuple[_FakeConnection, None]]":
        conn = _FakeConnection(
            client,
            iter(script),
            agent_capabilities=agent_capabilities,
            config_options=config_options,
            config_option_calls=config_option_calls,
        )
        try:
            yield conn, None
        finally:
            conn.closed = True

    config._connect = connect
    return config


class RecordingClient:
    """An :class:`acp.Client` that records every ``session/update`` it receives.

    The server side of the harness: pair it with :func:`connect` to assert on the
    notifications an :class:`~ag2.acp.agent.ACPAgent` actually emitted, in the
    order it emitted them.

    Client capabilities are all off — this client implements no filesystem,
    terminal or permission behaviour, so advertising any would let a test pass
    against a capability nothing here provides.
    """

    def __init__(self) -> None:
        self.updates: list[tuple[str, SessionUpdate]] = []

    def updates_for(self, session_id: str) -> "list[SessionUpdate]":
        """Only the updates belonging to ``session_id``, in arrival order."""
        return [u for sid, u in self.updates if sid == session_id]

    async def session_update(self, *, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))

    async def request_permission(self, **kwargs: Any) -> Any:
        raise NotImplementedError("RecordingClient does not implement permissions.")

    async def write_text_file(self, **kwargs: Any) -> Any:
        raise NotImplementedError("RecordingClient does not implement fs/write_text_file.")

    async def read_text_file(self, **kwargs: Any) -> Any:
        raise NotImplementedError("RecordingClient does not implement fs/read_text_file.")

    async def create_terminal(self, **kwargs: Any) -> Any:
        raise NotImplementedError("RecordingClient does not implement terminals.")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"RecordingClient does not implement ext method {method!r}.")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


@asynccontextmanager
async def connect(
    server: "ACPAgent",
    *,
    client: "RecordingClient | None" = None,
    initialize: bool = True,
) -> "AsyncGenerator[tuple[ClientSideConnection, RecordingClient]]":
    """Yield a real ACP ``ClientSideConnection`` driving ``server`` in-process.

    Both sides are the genuine SDK connection classes, wired over a connected
    socket pair inside this process — no subprocess to spawn and no port to
    bind — so tests exercise real JSON-RPC framing, dispatch and error mapping.
    The ACP analogue of :func:`ag2.mcp.testing.connect`.

    Yields the connection (call ``new_session``, ``prompt``, … on it) and the
    :class:`RecordingClient` that captured the notifications.
    """
    from acp.core import ClientSideConnection

    recorder = client or RecordingClient()

    # One socket pair carries both directions. `acp` speaks newline-delimited
    # JSON over asyncio streams, and a connected socket gives each side a real
    # ``StreamReader``/``StreamWriter`` on every platform.
    agent_end, client_end = socket.socketpair()
    agent_reader, agent_writer = await asyncio.open_connection(sock=agent_end)
    client_reader, client_writer = await asyncio.open_connection(sock=client_end)

    # ``ACPAgent`` / ``RecordingClient`` implement the SDK's Agent / Client
    # Protocols structurally; mypy cannot see that through the ``**kwargs``
    # signatures the Protocols declare.
    from .guard import serve

    agent_task = asyncio.create_task(serve(server.bind, agent_reader, agent_writer))
    conn = ClientSideConnection(cast("Any", lambda _agent: recorder), client_writer, client_reader)
    try:
        if initialize:
            await conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        yield conn, recorder
    finally:
        for writer in (client_writer, agent_writer):
            writer.close()
        agent_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await agent_task


async def duplex() -> (
    "tuple[tuple[asyncio.StreamReader, asyncio.StreamWriter], tuple[asyncio.StreamReader, asyncio.StreamWriter]]"
):
    """A connected pair of asyncio stream endpoints, one per side.

    Backed by :func:`socket.socketpair` rather than :func:`os.pipe`. An anonymous
    pipe cannot be registered with Windows' IOCP, so the proactor event loop
    rejects it — ``connect_read_pipe`` there raises
    ``OSError: [WinError 6] The handle is invalid``. A socket works on every
    platform asyncio supports.
    """
    left, right = socket.socketpair()
    return await asyncio.open_connection(sock=left), await asyncio.open_connection(sock=right)

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
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import acp
from acp import schema

from .config import ACPConfig, _dispatch_kwargs
from .types import SessionUpdate

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from acp.core import ClientSideConnection

    from ag2.context import StreamId

    from .agent import ACPAgent
    from .config import ConnectHook
    from .remote import ACPRemoteConfig
    from .session import ACPSession

FAKE_SESSION_ID = "fake-session-1"

__all__ = (
    "FAKE_SESSION_ID",
    "ACPTurn",
    "FakeACPConfig",
    "RecordingClient",
    "ScriptedElicitation",
    "connect",
    "duplex",
    "duplex_acp_config",
    "fake_acp_config",
    "fake_remote_acp_config",
)


@dataclass
class ScriptedElicitation:
    """One ``elicitation/create`` the scripted agent issues — the agent asking the user.

    Attributes:
        message: The human-readable message describing what input is needed.
        mode: The requested mode — normally one of ACP's four
            ``ElicitationMode`` models (form/url × session/request scope).
            Anything else stands in for a mode a later protocol release adds,
            which the client is expected to decline rather than error on.
        complete: When ``True`` the agent follows the answer with an
            ``elicitation/complete`` notification, as an agent may alongside a
            ``url``-mode flow. Requires a ``mode`` carrying an ``elicitation_id``.
    """

    message: str
    mode: Any
    complete: bool = False


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
        elicitations: Questions the agent puts to the user during the turn, issued
            in order after ``on_prompt`` and before ``updates`` replay — so a turn
            scripts the question *and* the reply the agent gives once answered.
            Each response the client sends back is appended to the
            ``elicitation_responses`` list passed to :func:`fake_acp_config`.
    """

    updates: Sequence[SessionUpdate] = field(default_factory=tuple)
    stop_reason: str = "end_turn"
    usage: "schema.Usage | None" = None
    hang: bool = False
    on_prompt: "Callable[[], Awaitable[None]] | None" = None
    elicitations: Sequence[ScriptedElicitation] = field(default_factory=tuple)


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
        initialize_calls: "list[schema.ClientCapabilities | None] | None" = None,
        initialize_elicitations: "Sequence[ScriptedElicitation]" = (),
        elicitation_responses: "list[schema.CreateElicitationResponse] | None" = None,
    ) -> None:
        self._client = client
        self._turns = turns
        self._config_options = list(config_options or [])
        self._cancelled = asyncio.Event()
        self._agent_capabilities = agent_capabilities
        self._initialize_elicitations = initialize_elicitations
        self.new_session_kwargs: dict[str, Any] | None = None
        self.closed = False
        self.config_option_calls: list[tuple[str, str | bool]] = (
            config_option_calls if config_option_calls is not None else []
        )
        self.initialize_calls: list[schema.ClientCapabilities | None] = (
            initialize_calls if initialize_calls is not None else []
        )
        self.elicitation_responses: list[schema.CreateElicitationResponse] = (
            elicitation_responses if elicitation_responses is not None else []
        )

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        self.initialize_calls.append(kwargs.get("client_capabilities"))
        await self._elicit(self._initialize_elicitations)
        return schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=self._agent_capabilities,
        )

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        self.new_session_kwargs = kwargs
        return schema.NewSessionResponse(
            session_id=FAKE_SESSION_ID,
            config_options=self._config_options or None,
        )

    async def _elicit(self, elicitations: "Sequence[ScriptedElicitation]") -> None:
        """Put each scripted question to the client, recording what it answered."""
        for elicitation in elicitations:
            response = await self._client.create_elicitation(
                message=elicitation.message,
                mode=elicitation.mode,
            )
            self.elicitation_responses.append(response)
            if elicitation.complete:
                await self._client.complete_elicitation(elicitation_id=elicitation.mode.elicitation_id)

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
        await self._elicit(turn.elicitations)
        if turn.hang:
            await self._cancelled.wait()
            self._cancelled.clear()
            return schema.PromptResponse(stop_reason="cancelled")
        for update in turn.updates:
            await self._client.session_update(session_id=session_id, update=update)
        return schema.PromptResponse(stop_reason=turn.stop_reason, usage=turn.usage)


class _FakeConfigViews:
    """Public read-only views of a fake config's run-scoped state.

    Lets tests assert on session lifecycle (leaks, teardown) without reaching
    into private fields. Mixed into the launch-based and remote fakes alike —
    the whole point of the design is that a behavioural test cannot tell them
    apart, so neither should the harness.
    """

    __slots__ = ()

    @property
    def sessions(self) -> "dict[StreamId, ACPSession]":
        """Live sessions keyed by stream id (empty once ``aclose()`` ran)."""
        return self._sessions  # type: ignore[attr-defined]

    @property
    def connect(self) -> "ConnectHook":
        """The in-process connection opener, for driving ``ACPSession.ensure`` directly."""
        connect = self._connect  # type: ignore[attr-defined]
        assert connect is not None
        return cast("ConnectHook", connect)


@dataclass(slots=True, kw_only=True)
class FakeACPConfig(_FakeConfigViews, ACPConfig):
    """:class:`ACPConfig` bound to the scripted in-process agent."""


@cache
def _fake_remote_config_type() -> "type[ACPRemoteConfig]":
    """The remote fake's class, built on first use.

    Deferred so importing this module does not require
    ``agent-client-protocol[http]``: a caller who only drives local subprocesses
    still gets the rest of the harness.
    """
    from .remote import ACPRemoteConfig

    @dataclass(slots=True, kw_only=True)
    class FakeACPRemoteConfig(_FakeConfigViews, ACPRemoteConfig):
        """:class:`ACPRemoteConfig` bound to the scripted in-process agent."""

    return FakeACPRemoteConfig


def _scripted_connect(
    *turns: ACPTurn,
    agent_capabilities: "schema.AgentCapabilities | None" = None,
    config_options: "Sequence[schema.SessionConfigOptionSelect] | None" = None,
    config_option_calls: "list[tuple[str, str | bool]] | None" = None,
    initialize_calls: "list[schema.ClientCapabilities | None] | None" = None,
    initialize_elicitations: "Sequence[ScriptedElicitation]" = (),
    elicitation_responses: "list[schema.CreateElicitationResponse] | None" = None,
) -> "ConnectHook":
    """A connection hook yielding the scripted in-process agent and no process."""
    if agent_capabilities is None:
        agent_capabilities = schema.AgentCapabilities(mcp_capabilities=schema.McpCapabilities(http=True, sse=True))
    script = list(turns)

    @asynccontextmanager
    async def connect(client: acp.Client) -> "AsyncGenerator[tuple[_FakeConnection, None]]":
        conn = _FakeConnection(
            client,
            iter(script),
            agent_capabilities=agent_capabilities,
            config_options=config_options,
            config_option_calls=config_option_calls,
            initialize_calls=initialize_calls,
            initialize_elicitations=initialize_elicitations,
            elicitation_responses=elicitation_responses,
        )
        try:
            yield conn, None
        finally:
            conn.closed = True

    return cast("ConnectHook", connect)


def fake_acp_config(
    *turns: ACPTurn,
    agent_capabilities: "schema.AgentCapabilities | None" = None,
    config_options: "Sequence[schema.SessionConfigOptionSelect] | None" = None,
    config_option_calls: "list[tuple[str, str | bool]] | None" = None,
    initialize_calls: "list[schema.ClientCapabilities | None] | None" = None,
    initialize_elicitations: "Sequence[ScriptedElicitation]" = (),
    elicitation_responses: "list[schema.CreateElicitationResponse] | None" = None,
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

    The elicitation seam works the same way — one field for what the agent asks,
    one list for what the client answered:

    * ``ACPTurn.elicitations`` are the questions the agent puts to the user
      mid-turn, and ``initialize_elicitations`` the ones it puts *before* any
      session exists (a pre-session auth flow, necessarily request-scoped).
    * every response the client sends back is appended to
      ``elicitation_responses`` in order.
    * the ``client_capabilities`` of each ``initialize`` is appended to
      ``initialize_calls``, which is how a test sees what the agent was told AG2
      supports.

    See :func:`fake_remote_acp_config` for the same agent behind a remote config.
    """
    config = FakeACPConfig(**overrides)
    config._connect = _scripted_connect(
        *turns,
        agent_capabilities=agent_capabilities,
        config_options=config_options,
        config_option_calls=config_option_calls,
        initialize_calls=initialize_calls,
        initialize_elicitations=initialize_elicitations,
        elicitation_responses=elicitation_responses,
    )
    return config


def fake_remote_acp_config(
    *turns: ACPTurn,
    url: str = "https://agent.example/acp",
    agent_capabilities: "schema.AgentCapabilities | None" = None,
    config_options: "Sequence[schema.SessionConfigOptionSelect] | None" = None,
    config_option_calls: "list[tuple[str, str | bool]] | None" = None,
    initialize_calls: "list[schema.ClientCapabilities | None] | None" = None,
    initialize_elicitations: "Sequence[ScriptedElicitation]" = (),
    elicitation_responses: "list[schema.CreateElicitationResponse] | None" = None,
    **overrides: Any,
) -> "ACPRemoteConfig":
    """:func:`fake_acp_config`, but behind an :class:`ACPRemoteConfig`.

    Same scripted agent, same arguments; only the config the turns run through
    differs. No socket is opened — ``url`` is there because a remote config must
    have one, and to prove behaviour does not depend on it.
    """
    config = _fake_remote_config_type()(url=url, **overrides)
    config._connect = _scripted_connect(
        *turns,
        agent_capabilities=agent_capabilities,
        config_options=config_options,
        config_option_calls=config_option_calls,
        initialize_calls=initialize_calls,
        initialize_elicitations=initialize_elicitations,
        elicitation_responses=elicitation_responses,
    )
    return config


def _duplex_connect(agent: "Callable[[acp.Client], Any] | Any") -> "ConnectHook":
    """A connection hook that reaches ``agent`` over a real ACP connection in-process.

    ``agent`` is what ``acp.run_agent`` takes: an object implementing the SDK's
    ``Agent`` protocol, or a callable handed the connection to talk back through.
    """

    @asynccontextmanager
    async def connect(client: acp.Client) -> "AsyncGenerator[tuple[ClientSideConnection, None]]":
        from acp.agent.connection import AgentSideConnection

        (agent_reader, agent_writer), (client_reader, client_writer) = await duplex()
        # `listening=False` + an explicit task: the receive loop must be owned here
        # so it is cancelled with the connection rather than outliving the test.
        agent_conn = AgentSideConnection(agent, agent_writer, agent_reader, listening=False)
        serving = asyncio.ensure_future(agent_conn.listen())
        # The same arguments the real transports pass, `_dispatch_kwargs` included:
        # a harness that connected differently would not be exercising AG2's wiring.
        conn = acp.connect_to_agent(
            client, client_writer, client_reader, use_unstable_protocol=True, **_dispatch_kwargs(client)
        )
        try:
            yield conn, None
        finally:
            await conn.close()
            serving.cancel()
            with suppress(asyncio.CancelledError):
                await serving
            await agent_conn.close()
            for writer in (client_writer, agent_writer):
                writer.close()

    return cast("ConnectHook", connect)


def duplex_acp_config(agent: "Callable[[acp.Client], Any] | Any", **overrides: Any) -> FakeACPConfig:
    """An :class:`~.config.ACPConfig` reaching ``agent`` over a real ACP connection.

    Unlike :func:`fake_acp_config`, which calls the client's methods directly, both
    ends here are the genuine SDK connection classes over a socket pair — real
    JSON-RPC framing, real receive loop, real notification queue, real routers —
    with no subprocess to spawn and no agent program to keep on disk.

    Reach for it when the transport *is* the subject: notification dispatch and the
    ordering it implies, capability negotiation through ``initialize``, whether an
    unstable route reaches the client at all. For everything else prefer
    :func:`fake_acp_config` — a scripted turn says more about behaviour with less
    machinery. What this does not cover is the launch path itself (argv, env,
    ``cwd``, a process to terminate); a test about that needs a real subprocess.
    """
    config = FakeACPConfig(**overrides)
    config._connect = _duplex_connect(agent)
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

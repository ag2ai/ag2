# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Expose an AG2 :class:`~ag2.Agent` through the Agent Client Protocol.

This is the *serving* half of AG2's ACP support: AG2 plays the ACP **Agent** role
so any ACP Client (Zed, an SDK client, an application acting on a user's behalf)
can drive an AG2 agent. The consume side — AG2 driving an external CLI agent —
lives in :mod:`ag2.acp.config` and :mod:`ag2.acp.client`.

The class is named for the role ACP itself defines. AG2 names each protocol
adapter after that protocol's own word for the serving side — MCP and A2A both
call it a *Server* (:class:`ag2.mcp.MCPServer`, :class:`ag2.a2a.A2AServer`), and
ACP calls it an *Agent*, so this is ``ACPAgent``. It is a wrapper around an AG2
:class:`~ag2.Agent`, not a subclass of one.
"""

import asyncio
import importlib.metadata
import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import acp
from acp import schema
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.stdio import stdio_streams

from ag2.agent import Agent, _get_stream_turn_lock

from .auth import AuthProvider
from .executor import STOP_CANCELLED, AgentExecutor, heal_cancelled_turn, isolate_variables, unanswered_tool_calls
from .guard import serve
from .mappers import history_to_session_updates
from .sessions import (
    AgentSession,
    ConnectionOverloadedError,
    SessionBusyError,
    SessionConfig,
    SessionLimitError,
    SessionStore,
    UnknownSessionError,
)

if TYPE_CHECKING:
    from ag2.hitl import HumanHook

    from .types import ContentBlock, McpServer

logger = logging.getLogger(__name__)

_DEFAULT_VERSION = "0.0.0"

SessionOrigin = Literal["created", "loaded"]
"""How a session came to be on a connection: minted by ``session/new``, or
reattached by ``session/load`` — whether from this connection's own registry or
rehydrated from storage."""

SessionObserver = Callable[[AgentSession, SessionOrigin], Awaitable[None] | None]
"""A host's hook into session lifetimes; see ``ACPAgent(on_session=...)``."""


@dataclass(frozen=True, slots=True)
class PromptContent:
    """Which non-text prompt content this deployment can actually handle.

    Advertised verbatim in the ``initialize`` handshake, where a Client reads it
    to decide what it may send. Whether a given block *works* depends on the
    model behind the agent, and AG2 has no registry of provider modalities to
    consult — the mapper converting ACP blocks into AG2 inputs will happily build
    an ``AudioInput`` that, say, the Anthropic mapper then rejects.

    So this is a declaration by whoever deploys the agent, not a guess. The
    defaults cover what every provider AG2 ships supports; ``audio`` is off
    because most do not. Turn on what your model actually accepts::

        ACPAgent(agent, prompt_content=PromptContent(audio=True))
    """

    image: bool = True
    audio: bool = False
    embedded_context: bool = True


def _request_meta(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request's ACP ``_meta`` map, as delivered to a handler.

    The SDK router *spreads* a request's ``_meta`` entries into the handler's
    keyword arguments rather than passing the map itself (``acp/router.py``:
    ``params.update(meta)``). Every declared parameter is bound by name, so
    whatever is left in ``**kwargs`` is exactly the metadata the Client sent.

    Kept verbatim. ``_meta`` is where applications put their own namespaced data
    — AG2 Space provenance, say — and AG2 has no business interpreting it.
    """
    return dict(kwargs)


def _package_version() -> str:
    try:
        return importlib.metadata.version("ag2")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - ag2 always installed in practice
        return _DEFAULT_VERSION


class ACPAgent:
    """Serve an AG2 :class:`Agent` as an ACP Agent.

    The simplest form serves over stdio, which is how ACP Clients normally launch
    an Agent::

        from ag2 import Agent
        from ag2.acp import ACPAgent

        await ACPAgent(Agent("workie", ...)).run_stdio()

    The instance implements the :class:`acp.Agent` protocol, so it can also be
    handed to :func:`acp.run_agent` directly with your own streams.

    Each ``session/new`` gets an isolated conversation with its own history;
    sessions never see each other's messages. Concurrent prompts on *one* session
    queue rather than interleave (bounded by ``sessions.max_queued``), matching
    how :class:`ag2.mcp.MCPServer` already treats one session's overlapping calls.

    ``session/cancel`` names a session, so it cancels that session's running turn
    *and* everything queued behind it, leaving other sessions untouched. Events
    already streamed stay in the session's history.

    ``session/load`` reattaches a Client to a session by id and replays its
    history. The id is enough on its own — it is the key the history is stored
    under — so a session can be loaded from a later connection, or a later
    process, as long as its history is still in the ``sessions.storage``.
    Pair ``SessionConfig(retain_history=True)`` with a durable storage for that;
    by default a session's history goes when its connection does.

    Args:
        agent: The AG2 agent to expose.
        name: Advertised agent name (defaults to ``agent.name``).
        version: Advertised version (defaults to the installed ``ag2`` version).
        title: Optional human-readable title for the ``initialize`` handshake.
        sessions: Session registry bounds — ``True`` for defaults, or a
            :class:`SessionConfig` to tune the LRU cap, idle TTL, history backend
            and per-session prompt queue depth.
        auth: Authentication provider. ``None`` (the default) advertises no auth
            methods and rejects ``authenticate`` — appropriate for local stdio,
            where the Client already launched the process. A provider decides
            whether a Client may connect, not which Client it is: no principal
            reaches the turn, so serve one tenant per instance. See
            :mod:`ag2.acp.auth` for the full limitation.
        stream_thoughts: Whether to project the agent's reasoning as ACP
            ``agent_thought_chunk`` updates. Off by default: reasoning is
            internal, and an ACP Client may be an external audience. Even when
            on, thoughts are only sent to a Client that advertised support.
        prompt_content: Which non-text prompt content to advertise as supported.
            See :class:`PromptContent` — it depends on the model behind the
            agent, so it is declared rather than guessed.
        hitl_hook: How to answer a turn that calls ``context.input()``. ``None``
            (the default) fails the turn with
            :class:`~ag2.acp.executor.HumanInputUnsupportedError`, because ACP
            elicitation is not wired on this side and there is genuinely nobody
            to ask. An application that *can* reach a human by its own means —
            its own chat surface, a queue, an approval UI — passes a hook here
            and its return value becomes the answer. Same shape as
            ``Agent(hitl_hook=...)``, dependency injection included; it replaces
            the served agent's own hook for ACP-driven turns, so bind it per
            connection (see :meth:`bind`) if the human differs per Client. This
            is deliberately invisible on the wire: no capability is advertised
            and no ACP method is called.
        on_session: Called with ``(session, origin)`` each time a session comes
            to exist on a connection — ``"created"`` after ``session/new``,
            ``"loaded"`` after ``session/load`` (before the replay). The seam
            for a host that keeps its own record per conversation and needs to
            tie an ACP session id to it, both when the id is minted and when a
            Client comes back with one. Sync or async. An exception fails the
            request rather than being swallowed, and leaves no session behind:
            a refused load is as if it never happened, with the conversation
            untouched in storage. A host that could not correlate a session —
            or will not hand it to this caller — is never left serving it.
    """

    __slots__ = (
        "_agent",
        "_executor",
        "_name",
        "_title",
        "_version",
        "_auth",
        "_stream_thoughts",
        "_prompt_content",
        "_session_config",
        "_on_session",
        "_scope",
    )

    def __init__(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        version: str | None = None,
        title: str | None = None,
        sessions: "bool | SessionConfig" = True,
        auth: AuthProvider | None = None,
        stream_thoughts: bool = False,
        prompt_content: PromptContent | None = None,
        hitl_hook: "HumanHook | None" = None,
        on_session: SessionObserver | None = None,
    ) -> None:
        self._agent = agent
        self._name = name or agent.name
        self._version = version or _package_version()
        self._title = title
        self._auth = auth
        self._stream_thoughts = stream_thoughts
        self._prompt_content = prompt_content or PromptContent()
        config = sessions if isinstance(sessions, SessionConfig) else SessionConfig()
        self._executor = AgentExecutor(agent, stream_thoughts=stream_thoughts, hitl_hook=hitl_hook)
        self._session_config = config
        self._on_session = on_session
        # The most recently opened connection, so ``sessions`` has something to
        # point at. Authorization and sessions live on the scope, never here.
        self._scope: _ConnectionScope | None = None

    @property
    def agent(self) -> Agent:
        return self._agent

    async def _observe(self, session: AgentSession, origin: SessionOrigin) -> None:
        """Tell the host's ``on_session`` a session exists on a connection now."""
        if self._on_session is None:
            return
        result = self._on_session(session, origin)
        if inspect.isawaitable(result):
            await result

    @property
    def sessions(self) -> SessionStore:
        """The current connection's session registry (for advanced wiring and tests).

        Sessions belong to a connection, so this is the newest one's store. Before
        any Client has connected it is an empty store belonging to nothing.
        """
        if self._scope is None:
            self._scope = self.bind(None)
        return self._scope.sessions

    def bind(self, client: "acp.Client | None") -> "_ConnectionScope":
        """Open a fresh authorization and session scope for one Client.

        Handed to the SDK as a factory so that *every* connection gets its own
        state. Sharing one mutable set of flags across connections cannot be made
        safe: a request carries no connection identity, so a handler reading
        shared flags is reading whichever connection touched them last.
        """
        scope = _ConnectionScope(self, client)
        self._scope = scope
        return scope

    async def run_stdio(self, *, buffer_limit_bytes: int | None = None) -> None:  # pragma: no cover - real stdio pipes
        """Serve over stdin/stdout until the Client disconnects.

        ``buffer_limit_bytes`` caps a single incoming ACP frame. The default is
        the SDK's own (50 MiB), not asyncio's 64 KiB: ACP frames are one line of
        JSON, so a prompt carrying an inline image or embedded document is one
        very long line. At asyncio's default an image of about 48 KiB — well
        under anything a user would think twice about — overruns the reader and
        drops the connection, while this agent advertises image and embedded
        content as supported.
        """
        limit = DEFAULT_STDIO_BUFFER_LIMIT_BYTES if buffer_limit_bytes is None else buffer_limit_bytes
        reader, writer = await stdio_streams(limit=limit)
        await serve(self.bind, reader, writer)


class _ConnectionScope:
    """One Client's slice of an :class:`ACPAgent`: its authorization and sessions.

    A request reaching a handler carries no hint of which connection sent it, so
    authorization state cannot live on an object shared between connections —
    whichever connection wrote the flags last would be speaking for all of them.
    Giving each connection its own scope is what makes "authenticated" mean
    "*this* Client authenticated".

    Created through :meth:`ACPAgent.bind`, which the SDK calls once per
    connection.
    """

    __slots__ = ("_owner", "_client", "_sessions", "_initialized", "_authenticated", "_client_capabilities")

    def __init__(self, owner: "ACPAgent", client: "acp.Client | None") -> None:
        self._owner = owner
        self._client = client
        self._sessions = SessionStore.from_config(owner._session_config)
        self._initialized = False
        self._authenticated = owner._auth is None
        self._client_capabilities: schema.ClientCapabilities | None = None

    @property
    def sessions(self) -> SessionStore:
        return self._sessions

    async def aclose(self) -> None:
        """Drop this connection's sessions — its scope is over."""
        await self._sessions.aclose()

    def on_connect(self, conn: acp.Client) -> None:
        """Record the Client handle this scope pushes ``session/update`` through."""
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: "schema.ClientCapabilities | None" = None,
        client_info: "schema.Implementation | None" = None,
        **kwargs: Any,
    ) -> schema.InitializeResponse:
        """Negotiate the protocol version and advertise implemented capabilities.

        Capabilities are derived from what is actually wired, never aspirational:
        a Client must be able to trust that anything advertised here works.

        Marks this connection as having completed its handshake; every session
        method below refuses to run before that. ACP requires initialize first
        but the SDK router does not enforce it, so this class does.
        """
        self._initialized = True
        self._client_capabilities = client_capabilities
        return schema.InitializeResponse(
            protocol_version=min(protocol_version, acp.PROTOCOL_VERSION),
            agent_capabilities=self._capabilities(),
            auth_methods=self._owner._auth.methods() if self._owner._auth is not None else [],
            agent_info=schema.Implementation(
                name=self._owner._name, title=self._owner._title, version=self._owner._version
            ),
        )

    def _capabilities(self) -> schema.AgentCapabilities:
        return schema.AgentCapabilities(
            # ``session/load`` is wired (see :meth:`load_session`). Advertised
            # unconditionally: an id whose history is gone gets a clean
            # ``resource_not_found``, which is what Clients expect from an Agent
            # that supports loading but no longer has that conversation.
            load_session=True,
            prompt_capabilities=schema.PromptCapabilities(
                image=self._owner._prompt_content.image,
                audio=self._owner._prompt_content.audio,
                embedded_context=self._owner._prompt_content.embedded_context,
            ),
            # Client-declared MCP servers are captured but never connected, so
            # every transport stays off.
            mcp_capabilities=schema.McpCapabilities(http=False, sse=False, acp=False),
            # Session operations are advertised by *presence*: an absent field
            # means unsupported. Everything past new / load / prompt / cancel —
            # list, delete, fork, resume, close — is unimplemented. (``resume``,
            # ``fork`` and ``close`` are also gated behind the SDK's unstable
            # protocol flag, so a stable v1 Client could not call them even if
            # they were wired.)
            session_capabilities=schema.SessionCapabilities(),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> schema.AuthenticateResponse:
        """Delegate to the configured provider; reject when none is configured."""
        if not self._initialized:
            raise acp.RequestError.invalid_request({"reason": "Call initialize before authenticate."})
        if self._owner._auth is None:
            raise acp.RequestError.invalid_request({
                "reason": "This ACP agent does not require or support authentication."
            })
        await self._owner._auth.authenticate(method_id, **kwargs)
        self._authenticated = True
        return schema.AuthenticateResponse()

    async def new_session(
        self,
        cwd: str,
        additional_directories: "list[str] | None" = None,
        mcp_servers: "list[McpServer] | None" = None,
        **kwargs: Any,
    ) -> schema.NewSessionResponse:
        """Create an isolated session and return its id.

        ``cwd``, ``additional_directories`` and ``mcp_servers`` are recorded as
        session context. They are *not* acted on: a path or MCP server named by a
        Client is context, not authorization for the agent to reach it. An
        embedding application decides what to honour.
        """
        self._require_session_scope()
        try:
            session = await self._sessions.create(
                cwd=cwd,
                additional_directories=list(additional_directories or []),
                mcp_servers=list(mcp_servers or []),
                meta=_request_meta(kwargs),
                # Seeded by value: this conversation owns its variables from here on.
                variables=isolate_variables(self._owner._agent._agent_variables),
            )
        except SessionLimitError as exc:
            raise acp.RequestError.invalid_request({"reason": str(exc)}) from exc
        try:
            await self._owner._observe(session, "created")
        except BaseException:
            # The Client never learns this id, but a registry entry the host
            # refused to account for must not outlive the refusal either.
            await self._sessions.forget(session.session_id)
            raise
        return schema.NewSessionResponse(session_id=session.session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: "list[McpServer] | None" = None,
        additional_directories: "list[str] | None" = None,
        **kwargs: Any,
    ) -> None:
        """Reattach to a session by id and replay its history to the Client.

        The id names the session's history directly (it is the stream id, in
        hex), so a session can be loaded that this connection never issued: one
        from an earlier connection, or an earlier process, whose history is
        still in storage. What is *not* in storage — ``cwd`` and the rest of the
        request context — the request supplies again, and context variables are
        re-seeded from the agent's defaults. A session still live on this
        connection is reused as it is, variables and all; only the request
        context is refreshed.

        Per ACP, every stored entry is replayed as a ``session/update`` before
        this returns — user turns as ``user_message_chunk``, replies as
        ``agent_message_chunk``, tool calls and their results — and nothing is
        sent after. A Client may therefore treat the response as the line
        between history and whatever comes next.

        A session whose last turn was cut short (the process died mid-tool)
        has its transcript repaired first, so the next prompt sends a valid
        conversation to the provider. A session with a turn in progress is
        refused rather than replayed underneath it.

        Any authenticated connection may load any id it holds. The id is the
        credential: a ``uuid4``, unguessable, and never more than that. A host
        that needs to bind sessions to a principal does so at its transport,
        or in ``on_session``.
        """
        self._require_session_scope()
        client = self._require_client()
        stream_id = self._parse_session_id(session_id)
        session_id = stream_id.hex  # the canonical form, whatever the Client sent

        context: dict[str, Any] = {
            "cwd": cwd,
            "additional_directories": list(additional_directories or []),
            "mcp_servers": list(mcp_servers or []),
        }
        meta = _request_meta(kwargs)

        # Not live here? Then only storage can vouch for the id. An id with no
        # history is indistinguishable from one never issued, and is answered
        # the same way — including a session opened and dropped before its
        # first prompt, which has nothing to resume anyway.
        try:
            await self._sessions.get(session_id)
        except UnknownSessionError:
            if not list(await self._sessions.storage.get_history(stream_id)):
                raise acp.RequestError.resource_not_found(f"acp-session:{session_id}") from None

        try:
            session, adopted = await self._sessions.get_or_adopt(
                stream_id,
                **context,
                meta=meta,
                variables=isolate_variables(self._owner._agent._agent_variables),
            )
        except SessionLimitError as exc:
            raise acp.RequestError.invalid_request({"reason": str(exc)}) from exc

        # Everything from here on can still refuse the load — the host's hook,
        # a busy check, a replay that cannot be delivered. An *adopted* session
        # was registered only so those questions could be asked about it; if
        # the answer is no, it must be unregistered again, or the refusal is
        # cosmetic: ``session/prompt`` would find the session and run.
        loaded = False
        try:
            if not adopted:
                # Refresh what the request carries; leave what it does not. The
                # variables are this conversation's own state and a load is not
                # a reason to lose them, and ``_meta`` absent means "unchanged",
                # not "none".
                session.cwd = cwd
                session.additional_directories = context["additional_directories"]
                session.mcp_servers = context["mcp_servers"]
                if meta:
                    session.meta = meta
            if not session.is_idle:
                raise acp.RequestError.invalid_request({
                    "reason": f"ACP session {session_id!r} has a turn in progress; load it once that finishes.",
                })

            await self._owner._observe(session, "loaded")

            stream = self._sessions.stream(session)
            # The session lock keeps the next prompt out; the stream lock keeps
            # out a turn driving the *same stream id* from another object —
            # a second load of this id on another connection in this process.
            # Same order as the prompt path (session, then stream), so the two
            # cannot deadlock against each other.
            async with session.recovery():
                turn_lock = _get_stream_turn_lock(stream)
                if turn_lock.locked():
                    raise acp.RequestError.invalid_request({
                        "reason": f"ACP session {session_id!r} is in use by another connection.",
                    })
                async with turn_lock:
                    events = list(await stream.history.get_events())
                    if unanswered_tool_calls(events):
                        closed = await heal_cancelled_turn(stream)
                        logger.debug("closed %d unanswered tool call(s) while loading %s", closed, session_id)
                        events = list(await stream.history.get_events())
                    updates = history_to_session_updates(events, session_id=session_id)
                    await self._owner._executor.deliver(updates, client=client, session_id=session_id)
            loaded = True
        except acp.RequestError:
            raise
        except Exception as exc:
            logger.exception("ACP session/load failed for session %s", session_id)
            raise acp.RequestError.internal_error({
                "reason": str(exc) or exc.__class__.__name__,
                "type": exc.__class__.__name__,
            }) from exc
        finally:
            if adopted and not loaded:
                await self._sessions.forget(session_id)
        return None

    async def prompt(
        self,
        session_id: str,
        prompt: "list[ContentBlock]",
        **kwargs: Any,
    ) -> schema.PromptResponse:
        """Run one agent turn on ``session_id`` and report how it ended.

        A prompt arriving while the session is busy waits its turn. If the
        session is cancelled — before or during the turn — the response is
        ``stop_reason="cancelled"`` and the agent is never reached.

        A turn that fails for any other reason becomes a JSON-RPC internal error
        carrying the cause. The Client is in a different process, so an error
        without its reason leaves whoever is debugging with nothing at all.
        """
        self._require_session_scope()
        session = await self._get_session(session_id)
        client = self._require_client()
        meta = _request_meta(kwargs) or dict(session.meta)

        abandoned = False
        try:
            # Connection-wide admission first: a prompt waiting its turn on a busy
            # session still occupies a request handler, so it has to count.
            async with self._sessions.admit(), session.turn(), self._sessions.running_turn():
                task = asyncio.create_task(
                    self._owner._executor.run_turn(
                        session=session,
                        store=self._sessions,
                        client=client,
                        blocks=prompt,
                        meta=meta,
                    )
                )
                session.turn_task = task
                try:
                    # ``asyncio.wait`` reports the turn's own cancellation instead
                    # of re-raising it, which is what separates the two very
                    # different cancellations that reach the handler below: this
                    # one can only be *the request* being cancelled.
                    await asyncio.wait([task])
                except asyncio.CancelledError:
                    abandoned = True
                    # Nothing else is going to stop the turn now, and its history
                    # writes have to be finished before the store is torn down.
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task
                    raise
                stop_reason = task.result()
        except asyncio.CancelledError:
            if abandoned:
                # The request itself was cancelled, which happens when the
                # connection is being closed underneath it. There is no Client
                # left to answer, and the SDK stops its sender before it unwinds
                # handlers — so replying here would park on a closed sender
                # forever, and the close it is unwinding for would never finish.
                raise
            # Cancellation is a normal ACP outcome, not a failure: report it as a
            # stop reason rather than tearing down the JSON-RPC request. Repair
            # the transcript first — a cancel can land between a tool call and
            # its result, and the session has to stay usable afterwards.
            await self._heal(session)
            return schema.PromptResponse(stop_reason=STOP_CANCELLED)
        except (SessionBusyError, ConnectionOverloadedError) as exc:
            raise acp.RequestError.invalid_request({"reason": str(exc)}) from exc
        except acp.RequestError:
            raise  # already a protocol error with its own payload
        except Exception as exc:
            logger.exception("ACP prompt turn failed for session %s", session_id)
            raise acp.RequestError.internal_error({
                "reason": str(exc) or exc.__class__.__name__,
                "type": exc.__class__.__name__,
            }) from exc

        return schema.PromptResponse(stop_reason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel ``session_id``'s running turn and everything queued behind it.

        A notification, not a request: there is no response channel, so an
        unknown session id — or a caller with no right to touch it — is logged
        and ignored rather than raised.

        Cancelling is a mutation of someone's conversation, so it needs the same
        connection scope as prompting. Without that check a reconnect could stop
        work belonging to the connection it replaced, just by remembering an id.
        """
        try:
            self._require_session_scope()
        except acp.RequestError:
            logger.debug("session/cancel outside an authorized connection scope; ignored")
            return
        try:
            session = await self._sessions.get(session_id)
        except UnknownSessionError:
            logger.debug("session/cancel for unknown session %s; ignored", session_id)
            return
        await session.cancel()

    async def _heal(self, session: AgentSession) -> None:
        """Make a cancelled session's history valid to send to a provider again.

        Runs under the session's turn lock so the next prompt cannot start
        mid-repair — the repair rewrites the whole transcript, and a turn racing
        it would either read an unanswered tool call or have its own events
        overwritten by a stale snapshot. Several cancelled prompts may each land
        here; the lock serializes them and the repair is idempotent, so the ones
        after the first find nothing left to close.

        Best-effort: if it fails the session is still cancelled, and reporting
        that matters more than the repair.
        """
        try:
            async with session.recovery():
                closed = await heal_cancelled_turn(self._sessions.stream(session))
        except Exception:  # pragma: no cover - defensive; storage failures only
            logger.warning("could not repair history for cancelled session %s", session.session_id, exc_info=True)
            return
        if closed:
            logger.debug("closed %d unanswered tool call(s) after cancelling %s", closed, session.session_id)

    async def _get_session(self, session_id: str) -> AgentSession:
        try:
            return await self._sessions.get(session_id)
        except UnknownSessionError as exc:
            raise acp.RequestError.resource_not_found(f"acp-session:{session_id}") from exc

    @staticmethod
    def _parse_session_id(session_id: str) -> UUID:
        """The stream id a session id names, or ``resource_not_found`` if it names none.

        A malformed id is not a malformed *request* — the Client sent a
        well-formed ``session/load`` for a session that does not exist, which
        is the same answer an unknown id gets.
        """
        try:
            return UUID(session_id)
        except ValueError:
            raise acp.RequestError.resource_not_found(f"acp-session:{session_id}") from None

    def _require_client(self) -> acp.Client:
        if self._client is None:  # pragma: no cover - set by the SDK before any request
            raise acp.RequestError.internal_error({"reason": "ACP connection is not established."})
        return self._client

    def _require_session_scope(self) -> None:
        """Gate every session operation on an initialized, authenticated connection.

        ACP requires ``initialize`` before anything else, but the SDK router does
        not enforce that — so we do. Without it, a reconnecting Client could skip
        the handshake and, with it, the scope reset that revokes the previous
        connection's authentication.
        """
        if not self._initialized:
            raise acp.RequestError.invalid_request({
                "reason": "Call initialize before any session method.",
            })
        if not self._authenticated:
            raise acp.RequestError.auth_required({"reason": "Call authenticate first."})

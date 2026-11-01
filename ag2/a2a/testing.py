# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, ExitStack
from typing import TYPE_CHECKING, Any

import httpx
from a2a.types import AgentCard, TaskState
from typing_extensions import Self

from ag2.agent import Agent
from ag2.annotations import Context
from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig, ModelProvider
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelResponse, ToolResultsEvent
from ag2.observers import Observer
from ag2.testing import TestConfig, Turn

from .card import build_card
from .events import A2ATaskStatusUpdate
from .executor import AgentExecutor
from .server import A2AServer

if TYPE_CHECKING:
    from ag2.files.protocol import FilesClient

__all__ = (
    "ObservedAgentExecutor",
    "TaskStatusRecorder",
    "ToolRoundsConfig",
    "TurnGate",
    "make_test_client_factory",
    "make_test_rest_client_factory",
    "pick_free_port",
)


class ToolRoundsConfig(ModelConfig):
    """Script a served agent one tool round at a time.

    ``A2AServer`` builds a fresh LLM client for every request — the first
    message and each continuation carrying client-tool results — so a single
    :class:`~ag2.testing.TestConfig` would replay its first turn on every
    continuation. This config reads the round from the history instead: the
    model call that follows the N-th ``ToolResultsEvent`` is answered by
    ``rounds[N]``. Being stateless, it also behaves the same when the
    continuation lands on another server replica.

    A round is a single :data:`~ag2.testing.Turn`, or a ``TestConfig`` when it
    publishes events (chunks, reasoning) before the turn that ends it:

    .. code-block:: python

        config = ToolRoundsConfig(
            TestConfig(ModelMessageChunk("Checking..."), ToolCallEvent(name="lookup")),
            "Found it.",
        )
    """

    def __init__(self, *rounds: "Turn | TestConfig") -> None:
        self.rounds = [r if isinstance(r, TestConfig) else TestConfig(r) for r in rounds]

    @property
    def provider(self) -> ModelProvider:
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError

    def copy(self) -> Self:
        return self

    def create(self) -> "_ToolRoundsClient":
        return _ToolRoundsClient(self.rounds)

    def create_files_client(self) -> "FilesClient":
        raise NotImplementedError(f"{type(self).__name__} does not support Files API.")


class _ToolRoundsClient(LLMClient):
    def __init__(self, rounds: Sequence[TestConfig]) -> None:
        self.rounds = rounds

    async def __call__(
        self, messages: Sequence[BaseEvent], context: ConversationContext, **kwargs: Any
    ) -> ModelResponse:
        index = sum(isinstance(m, ToolResultsEvent) for m in messages)
        if index >= len(self.rounds):
            raise RuntimeError(
                f"ToolRoundsConfig scripts {len(self.rounds)} round(s); the model was called for round {index + 1}"
            )
        return await self.rounds[index].create()(messages, context, **kwargs)


class ObservedAgentExecutor(AgentExecutor):
    """``AgentExecutor`` that runs ``observers`` over each request's whole task lifecycle.

    Observers on the served ``Agent`` only run during the model turn, so they miss ``WORKING``
    and the terminal status. These see every ``A2ATaskStatusUpdate`` the request puts on the
    wire, in wire order. Serve it with ``A2AServer(agent, executor=...)``:

    .. code-block:: python

        statuses = TaskStatusRecorder()
        server = A2AServer(agent, executor=ObservedAgentExecutor(agent, statuses))
        ...
        assert statuses.states == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_COMPLETED]
    """

    def __init__(self, agent: Agent, *observers: Observer) -> None:
        super().__init__(agent)
        self._observers = observers

    def _observe(self, stack: ExitStack, context: ConversationContext) -> None:
        for observer in self._observers:
            observer.register(stack, context)


class TaskStatusRecorder(Observer):
    """Observer that records the state of every ``A2ATaskStatusUpdate``, in delivery order."""

    def __init__(self) -> None:
        self.states: list[TaskState] = []
        self._reached: dict[TaskState, asyncio.Event] = {}

    def register(self, stack: ExitStack | AsyncExitStack, context: Context) -> None:
        stack.enter_context(context.stream.where(A2ATaskStatusUpdate).sub_scope(self._record))

    async def wait_for(self, state: TaskState) -> None:
        """Wait until an update with ``state`` has been delivered."""
        await self._event(state).wait()

    async def _record(self, event: A2ATaskStatusUpdate) -> None:
        self.states.append(event.state)
        self._event(event.state).set()

    def _event(self, state: TaskState) -> asyncio.Event:
        return self._reached.setdefault(state, asyncio.Event())


class TurnGate:
    """A server tool, ``hold``, that keeps the turn running until :meth:`release` is called.

    It holds the task in ``working``, so a test can act on it mid-turn:

    .. code-block:: python

        gate = TurnGate()
        agent = Agent("server", config=TestConfig(ToolCallEvent(name="hold"), "done"), tools=[gate.hold])
        asking = asyncio.create_task(client.ask("go"))
        await gate.entered.wait()
        ...  # cancel, inspect the task
        gate.release()
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self._released = asyncio.Event()

    async def hold(self) -> str:
        self.entered.set()
        await self._released.wait()
        return "released"

    def release(self) -> None:
        self._released.set()


def make_test_client_factory(
    server: A2AServer,
    *,
    url: str = "http://test",
    card: AgentCard | None = None,
    timeout: float = 30.0,
) -> Callable[[], httpx.AsyncClient]:
    """Build an ``httpx.AsyncClient`` factory that talks to ``server`` in-process.

    Uses ``httpx.ASGITransport`` to dispatch directly into the Starlette
    app produced by ``server.build_jsonrpc(url=url)`` — no real socket,
    no port binding, no SSE proxy in the way. Use it as the
    ``httpx_client_factory`` on ``A2AConfig`` for end-to-end tests:

    .. code-block:: python

        server = A2AServer(agent)
        factory = make_test_client_factory(server, url="http://test")
        remote = Agent(
            "remote",
            config=A2AConfig(card_url="http://test", httpx_client_factory=factory),
        )
        await remote.ask("ping")

    The transport is created **once** and shared by every client the
    factory hands out, which matches how httpx.ASGITransport is meant to
    be reused. Each client returned by the factory is independent and
    closed by the caller (``A2AClient`` runs ``aclose()`` after each
    ``ask``); ``ASGITransport`` itself doesn't need explicit cleanup.

    ``card`` overrides the card the server publishes. Pass it when the
    default ``build_card(agent, url=url)`` can't express what the test
    needs — a custom ``extensions`` list, for instance. ``None`` keeps
    the default.
    """
    app = server.build_jsonrpc(url=url, card=card)
    transport = httpx.ASGITransport(app=app)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url=url, timeout=timeout)

    return factory


def make_test_rest_client_factory(
    server: A2AServer,
    *,
    url: str = "http://test",
    card: AgentCard | None = None,
    timeout: float = 30.0,
) -> Callable[[], httpx.AsyncClient]:
    """Build an ``httpx.AsyncClient`` factory talking REST to ``server`` in-process.

    REST counterpart to :func:`make_test_client_factory`. Builds the
    Starlette REST app via ``server.build_rest(url=url)`` and dispatches
    through ``httpx.ASGITransport``. The card served at ``url`` declares
    only the REST interface, so an ``A2AConfig(card_url=url, prefer="rest")``
    client connects through this factory without leaking to other
    transports.

    ``card`` overrides that default REST card; it must still declare a
    REST interface, or the client will pick a different transport.
    """
    rest_card = card if card is not None else build_card(server.agent, url=url, transports=("rest",))
    app = server.build_rest(url=url, card=rest_card)
    transport = httpx.ASGITransport(app=app)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url=url, timeout=timeout)

    return factory


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Find an available TCP port on ``host``.

    Used by gRPC tests that need a real listening socket — there is no
    in-process equivalent of ``httpx.ASGITransport`` for gRPC, so the
    test must bind a real port. Race-prone in theory (the port can be
    snatched between the probe and the gRPC server's ``start()``) but
    fine in practice for sequential local test runs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port: int = sock.getsockname()[1]
        return port

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from a2a.server.tasks import (
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard

from autogen.beta.agent import Agent

from .card import build_card
from .executor import AgentExecutor
from .transports.grpc import build_grpc_server
from .transports.jsonrpc import (
    DEFAULT_AGENT_CARD_PATH,
    LEGACY_AGENT_CARD_PATH,
    CardModifier,
    ExtendedCardModifier,
    build_jsonrpc_asgi,
)
from .transports.rest import build_rest_asgi

if TYPE_CHECKING:
    from grpc.aio import Server
    from starlette.applications import Starlette


class A2AServer:
    """Wrap an AG2 ``Agent`` as an A2A endpoint.

    The class holds transport-agnostic state — the executor, optional
    task/push stores, the extended card, and per-card modifier hooks.
    Transport-specific parameters (URL, ports, paths) live on the
    ``build_*`` methods, since one server can be exposed on different
    URLs through different transports.

    ``extended_card``, when supplied, is served via the JSON-RPC
    ``GetExtendedAgentCard`` method (auth-aware extra metadata). The
    public card automatically flips ``capabilities.extended_agent_card``
    when an extended card is provided.

    Each builder returns a ready-to-serve transport object:

    - :py:meth:`build_jsonrpc` returns a Starlette ASGI app.
    - :py:meth:`build_rest` returns a Starlette ASGI app.
    - :py:meth:`build_grpc` returns a ``grpc.aio.Server``.

    A2A spec doesn't define middleware. To add cross-cutting concerns
    (CORS, auth, tracing) attach them to the returned transport object
    directly — Starlette has ``app.add_middleware(...)``, gRPC has
    interceptors on ``grpc.aio.Server``.

    Example::

        server = A2AServer(agent)
        app = server.build_jsonrpc(url="http://localhost:8000")
        app.add_middleware(CORSMiddleware, allow_origins=["*"])
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """

    __slots__ = (
        "_agent",
        "_card_modifier",
        "_executor",
        "_extended_card",
        "_extended_card_modifier",
        "_push_config_store",
        "_push_sender",
        "_task_store",
    )

    def __init__(
        self,
        agent: Agent,
        *,
        extended_card: AgentCard | None = None,
        card_modifier: CardModifier | None = None,
        extended_card_modifier: ExtendedCardModifier | None = None,
        task_store: TaskStore | None = None,
        push_config_store: PushNotificationConfigStore | None = None,
        push_sender: PushNotificationSender | None = None,
    ) -> None:
        self._agent = agent
        self._extended_card = extended_card
        self._card_modifier = card_modifier
        self._extended_card_modifier = extended_card_modifier
        self._task_store = task_store
        self._push_config_store = push_config_store
        self._push_sender = push_sender
        self._executor = AgentExecutor(agent)

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def extended_card(self) -> AgentCard | None:
        return self._extended_card

    def build_jsonrpc(
        self,
        *,
        url: str,
        card: AgentCard | None = None,
        rpc_url: str = "/",
        card_url: str = DEFAULT_AGENT_CARD_PATH,
        legacy_card_url: str | None = LEGACY_AGENT_CARD_PATH,
    ) -> "Starlette":
        """Build a Starlette ASGI app exposing JSON-RPC routes + agent card.

        ``url`` is the public base URL the server will be reached at.
        ``card`` defaults to a freshly-built single-transport card for
        this URL — override when you need to declare additional
        transports (REST/gRPC) on the same card. ``legacy_card_url``
        registers a second card route on the v0.x path; pass ``None`` to
        disable.
        """
        resolved_card = card or build_card(self._agent, url=url, transports=("jsonrpc",))
        return build_jsonrpc_asgi(
            agent_executor=self._executor,
            agent_card=resolved_card,
            extended_agent_card=self._extended_card,
            card_modifier=self._card_modifier,
            extended_card_modifier=self._extended_card_modifier,
            task_store=self._task_store,
            push_config_store=self._push_config_store,
            push_sender=self._push_sender,
            rpc_url=rpc_url,
            card_url=card_url,
            legacy_card_url=legacy_card_url,
        )

    def build_rest(
        self,
        *,
        url: str,
        card: AgentCard | None = None,
        path_prefix: str = "",
        card_url: str = DEFAULT_AGENT_CARD_PATH,
        legacy_card_url: str | None = LEGACY_AGENT_CARD_PATH,
    ) -> "Starlette":
        """Build a Starlette ASGI app exposing REST routes + agent card.

        ``path_prefix`` mounts the REST routes under a sub-path (e.g.
        ``"/v1"``); both the AgentCard interface URL and the dispatcher
        respect it. See :py:meth:`build_jsonrpc` for the ``card`` /
        ``legacy_card_url`` semantics.
        """
        resolved_card = card or build_card(
            self._agent,
            url=url,
            transports=("rest",),
            rest_path_prefix=path_prefix,
        )
        return build_rest_asgi(
            agent_executor=self._executor,
            agent_card=resolved_card,
            extended_agent_card=self._extended_card,
            card_modifier=self._card_modifier,
            extended_card_modifier=self._extended_card_modifier,
            task_store=self._task_store,
            push_config_store=self._push_config_store,
            push_sender=self._push_sender,
            path_prefix=path_prefix,
            card_url=card_url,
            legacy_card_url=legacy_card_url,
        )

    def build_grpc(
        self,
        *,
        bind: str,
        grpc_url: str,
        card: AgentCard | None = None,
        options: Sequence[tuple[str, Any]] = (),
    ) -> "Server":
        """Build a ``grpc.aio.Server`` bound to ``bind``.

        ``bind`` is the listener address (e.g. ``"0.0.0.0:50051"``).
        ``grpc_url`` is the public URL clients will connect to (used in
        the AgentCard interface entry — usually identical to bind, but
        not when you're behind a load balancer). Insecure binding only.
        Caller is responsible for ``await server.start()`` and
        ``await server.wait_for_termination()``.
        """
        resolved_card = card or build_card(
            self._agent,
            url=grpc_url,
            transports=("grpc",),
            grpc_url=grpc_url,
        )
        return build_grpc_server(
            agent_executor=self._executor,
            agent_card=resolved_card,
            bind=bind,
            extended_agent_card=self._extended_card,
            extended_card_modifier=self._extended_card_modifier,
            task_store=self._task_store,
            push_config_store=self._push_config_store,
            push_sender=self._push_sender,
            options=options,
        )

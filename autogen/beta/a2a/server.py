# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from a2a.server.tasks import (
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard
from starlette.applications import Starlette

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
    import grpc as _grpc_t

    from autogen.beta.agent import Agent

TransportName = Literal["jsonrpc", "rest", "grpc"]


class A2AServer:
    """Wrap an AG2 ``Agent`` as an A2A endpoint.

    Defaults to JSON-RPC + SSE on the ``/`` path with the agent card
    served at ``/.well-known/agent-card.json`` and the legacy v0.x path
    ``/.well-known/agent.json`` (for backward compatibility). Pass
    ``legacy_card_url=None`` to disable the legacy route.

    ``extended_card``, when supplied, is served via the JSON-RPC
    ``GetExtendedAgentCard`` method (auth-aware extra metadata). The
    public card automatically flips ``capabilities.extended_agent_card``
    when an extended card is provided.

    ``transports`` selects which protocol bindings this server exposes.
    Each transport gets its own builder method:

    - ``"jsonrpc"`` → :py:meth:`build_jsonrpc` returns a Starlette ASGI app.
    - ``"rest"`` → :py:meth:`build_rest` returns a Starlette ASGI app.
    - ``"grpc"`` → :py:meth:`build_grpc` returns a ``grpc.aio.Server``.

    The three builders are independent; if a caller wants JSON-RPC and REST
    on the same port they assemble the routes themselves (uncommon).

    Example::

        server = A2AServer(agent, url="http://localhost:8000")
        server.add_middleware(CORSMiddleware, allow_origins=["*"])
        uvicorn.run(server.build_jsonrpc(), host="0.0.0.0", port=8000)
    """

    __slots__ = (
        "_agent",
        "_card",
        "_card_modifier",
        "_executor",
        "_extended_card",
        "_extended_card_modifier",
        "_grpc_url",
        "_legacy_card_url",
        "_middlewares",
        "_push_config_store",
        "_push_sender",
        "_rest_path_prefix",
        "_task_store",
        "_transports",
        "_url",
    )

    def __init__(
        self,
        agent: "Agent",
        *,
        url: str = "http://localhost:8000",
        card: AgentCard | None = None,
        extended_card: AgentCard | None = None,
        card_modifier: CardModifier | None = None,
        extended_card_modifier: ExtendedCardModifier | None = None,
        task_store: TaskStore | None = None,
        push_config_store: PushNotificationConfigStore | None = None,
        push_sender: PushNotificationSender | None = None,
        legacy_card_url: str | None = LEGACY_AGENT_CARD_PATH,
        transports: Sequence[TransportName] = ("jsonrpc",),
        rest_path_prefix: str = "",
        grpc_url: str | None = None,
    ) -> None:
        transports_tuple: tuple[TransportName, ...] = tuple(transports)
        if not transports_tuple:
            raise ValueError("transports must contain at least one of 'jsonrpc', 'rest', 'grpc'")
        if "grpc" in transports_tuple and grpc_url is None:
            raise ValueError("grpc_url is required when 'grpc' is in transports")
        self._agent = agent
        self._url = url
        self._card = card or build_card(
            agent,
            url=url,
            transports=transports_tuple,
            rest_path_prefix=rest_path_prefix,
            grpc_url=grpc_url,
        )
        self._extended_card = extended_card
        self._card_modifier = card_modifier
        self._extended_card_modifier = extended_card_modifier
        self._task_store = task_store
        self._push_config_store = push_config_store
        self._push_sender = push_sender
        self._legacy_card_url = legacy_card_url
        self._transports = transports_tuple
        self._rest_path_prefix = rest_path_prefix
        self._grpc_url = grpc_url
        self._middlewares: list[tuple[type, dict[str, Any]]] = []
        self._executor = AgentExecutor(agent)

    @property
    def agent(self) -> "Agent":
        return self._agent

    @property
    def card(self) -> AgentCard:
        return self._card

    @property
    def extended_card(self) -> AgentCard | None:
        return self._extended_card

    @property
    def url(self) -> str:
        return self._url

    @property
    def transports(self) -> tuple[TransportName, ...]:
        return self._transports

    def add_middleware(self, middleware_class: type, **kwargs: Any) -> None:
        """Register a Starlette middleware class to be applied at ASGI build time.

        The class is instantiated by Starlette per request when the ASGI
        app is built — ``A2AServer`` only stores the registration here,
        no side effects until ``build_jsonrpc()`` / ``build_rest()``.
        """
        self._middlewares.append((middleware_class, dict(kwargs)))

    def build_jsonrpc(
        self,
        *,
        rpc_url: str = "/",
        card_url: str = DEFAULT_AGENT_CARD_PATH,
    ) -> Starlette:
        """Build a Starlette ASGI app exposing JSON-RPC routes + agent card."""
        self._require_transport("jsonrpc")
        return build_jsonrpc_asgi(
            agent_executor=self._executor,
            agent_card=self._card,
            extended_agent_card=self._extended_card,
            card_modifier=self._card_modifier,
            extended_card_modifier=self._extended_card_modifier,
            middlewares=self._snapshot_middlewares(),
            task_store=self._task_store,
            push_config_store=self._push_config_store,
            push_sender=self._push_sender,
            rpc_url=rpc_url,
            card_url=card_url,
            legacy_card_url=self._legacy_card_url,
        )

    def build_rest(
        self,
        *,
        card_url: str = DEFAULT_AGENT_CARD_PATH,
    ) -> Starlette:
        """Build a Starlette ASGI app exposing REST routes + agent card."""
        self._require_transport("rest")
        return build_rest_asgi(
            agent_executor=self._executor,
            agent_card=self._card,
            extended_agent_card=self._extended_card,
            card_modifier=self._card_modifier,
            extended_card_modifier=self._extended_card_modifier,
            middlewares=self._snapshot_middlewares(),
            task_store=self._task_store,
            push_config_store=self._push_config_store,
            push_sender=self._push_sender,
            path_prefix=self._rest_path_prefix,
            card_url=card_url,
            legacy_card_url=self._legacy_card_url,
        )

    def build_grpc(
        self,
        *,
        bind: str,
        options: Sequence[tuple[str, Any]] = (),
    ) -> "_grpc_t.aio.Server":
        """Build a ``grpc.aio.Server`` bound to ``bind`` (e.g. ``"0.0.0.0:50051"``).

        Insecure binding only. Caller is responsible for ``await server.start()``
        and ``await server.wait_for_termination()`` — usually composed with the
        HTTP server's lifecycle via ``asyncio.gather``.
        """
        self._require_transport("grpc")
        return build_grpc_server(
            agent_executor=self._executor,
            agent_card=self._card,
            bind=bind,
            extended_agent_card=self._extended_card,
            extended_card_modifier=self._extended_card_modifier,
            task_store=self._task_store,
            push_config_store=self._push_config_store,
            push_sender=self._push_sender,
            options=options,
        )

    def _require_transport(self, name: TransportName) -> None:
        if name not in self._transports:
            raise RuntimeError(
                f"transport {name!r} not enabled — pass transports={(name,)!r} (or include {name!r}) to A2AServer"
            )

    def _snapshot_middlewares(self) -> list[tuple[type, Mapping[str, Any]]]:
        return [(cls, dict(kwargs)) for cls, kwargs in self._middlewares]

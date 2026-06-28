# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard

from ag2.agent import Agent

from .card import build_card
from .executor import AgentExecutor
from .transports._common import (
    DEFAULT_AGENT_CARD_PATH,
    LEGACY_AGENT_CARD_PATH,
    CardModifier,
    ExtendedCardModifier,
)
from .transports.grpc import build_grpc_server
from .transports.jsonrpc import build_jsonrpc_asgi
from .transports.rest import build_rest_asgi

if TYPE_CHECKING:
    from grpc.aio import Server
    from starlette.applications import Starlette


class A2AServer:
    """Wrap an AG2 ``Agent`` as an A2A endpoint.

    Holds transport-agnostic state (executor, task/push stores, extended
    card, per-card modifier hooks). Transport-specific parameters (URL,
    ports, paths) live on the ``build_*`` methods — one server can be
    exposed on different URLs through different transports.

    ``extended_card``, when supplied, is served via the JSON-RPC
    ``GetExtendedAgentCard`` method; the public card automatically flips
    ``capabilities.extended_agent_card`` when an extended card is provided.

    Each builder returns a ready-to-serve transport object:
    :py:meth:`build_jsonrpc` and :py:meth:`build_rest` return a Starlette
    ASGI app; :py:meth:`build_grpc` returns a ``grpc.aio.Server``.

    A2A spec doesn't define middleware — attach cross-cutting concerns
    (CORS, auth, tracing) to the returned transport object directly.
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
        executor: A2AAgentExecutorBase | None = None,
    ) -> None:
        self._agent = agent
        self._extended_card = extended_card
        self._card_modifier = card_modifier
        self._extended_card_modifier = extended_card_modifier
        # Materialise the store eagerly so multi-transport setups (same
        # server exposed via JSON-RPC + REST + gRPC) all share one task
        # store. Otherwise each builder defaults to its own.
        self._task_store = task_store or InMemoryTaskStore()
        self._push_config_store = push_config_store
        self._push_sender = push_sender
        # ``executor`` is escape-hatch for tests / advanced use cases that
        # need a custom ``AgentExecutor``. Default wraps the supplied agent.
        self._executor = executor if executor is not None else AgentExecutor(agent)

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def extended_card(self) -> AgentCard | None:
        return self._extended_card

    @property
    def task_store(self) -> TaskStore:
        """The shared task store used across all transport builders."""
        return self._task_store

    def _shared_kwargs(self, *, include_card_modifier: bool) -> dict[str, Any]:
        """Wiring shared by every ``build_*`` method.

        gRPC has no HTTP card route, so ``card_modifier`` is dropped there.
        """
        kwargs: dict[str, Any] = {
            "agent_executor": self._executor,
            "extended_agent_card": self._extended_card,
            "extended_card_modifier": self._extended_card_modifier,
            "task_store": self._task_store,
            "push_config_store": self._push_config_store,
            "push_sender": self._push_sender,
        }
        if include_card_modifier:
            kwargs["card_modifier"] = self._card_modifier
        return kwargs

    def build_jsonrpc(
        self,
        *,
        url: str,
        card: AgentCard | None = None,
        rpc_url: str = "/",
        card_url: str = DEFAULT_AGENT_CARD_PATH,
        legacy_card_url: str | None = LEGACY_AGENT_CARD_PATH,
    ) -> "Starlette":
        """Starlette ASGI app exposing JSON-RPC routes + agent card."""
        resolved_card = card or build_card(
            self._agent,
            url=url,
            transports=("jsonrpc",),
            push_notifications=self._push_config_store is not None,
        )
        return build_jsonrpc_asgi(
            agent_card=resolved_card,
            rpc_url=rpc_url,
            card_url=card_url,
            legacy_card_url=legacy_card_url,
            **self._shared_kwargs(include_card_modifier=True),
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
        """Starlette ASGI app exposing REST routes + agent card.

        ``path_prefix`` mounts REST under a sub-path (e.g. ``"/v1"``); both
        the AgentCard interface URL and the dispatcher respect it.
        """
        resolved_card = card or build_card(
            self._agent,
            url=url,
            transports=("rest",),
            rest_path_prefix=path_prefix,
            push_notifications=self._push_config_store is not None,
        )
        return build_rest_asgi(
            agent_card=resolved_card,
            path_prefix=path_prefix,
            card_url=card_url,
            legacy_card_url=legacy_card_url,
            **self._shared_kwargs(include_card_modifier=True),
        )

    def build_grpc(
        self,
        *,
        bind: str,
        grpc_url: str,
        card: AgentCard | None = None,
        options: Sequence[tuple[str, Any]] = (),
    ) -> "Server":
        """``grpc.aio.Server`` bound to ``bind``; caller starts/awaits it.

        ``bind`` is the listener address (e.g. ``"0.0.0.0:50051"``).
        ``grpc_url`` is the public URL clients will connect to (used in
        the AgentCard interface entry — usually identical to ``bind``,
        but not when behind a load balancer). Insecure binding only.

        ``card_modifier`` does not apply: A2A v1.x has no ``GetAgentCard``
        gRPC method — the public card is served over HTTP only.
        ``extended_card_modifier`` does apply (gRPC has ``GetExtendedAgentCard``).
        """
        resolved_card = card or build_card(
            self._agent,
            url=grpc_url,
            transports=("grpc",),
            grpc_url=grpc_url,
            push_notifications=self._push_config_store is not None,
        )
        return build_grpc_server(
            agent_card=resolved_card,
            bind=bind,
            options=options,
            **self._shared_kwargs(include_card_modifier=False),
        )

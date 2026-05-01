# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from a2a.server.request_handlers import DefaultRequestHandler, RequestHandler
from a2a.server.tasks import (
    InMemoryTaskStore,
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentExtension

from .cards import build_card
from .executor import AG2AgentExecutor
from .server_middleware import ExecutorMiddleware
from .transports import (
    build_asgi as _build_asgi,
)
from .transports import (
    build_grpc as _build_grpc,
)
from .transports import (
    build_rest as _build_rest,
)

if TYPE_CHECKING:
    from grpc.aio import Server as GrpcServer
    from starlette.applications import Starlette

    from autogen.beta import Agent


class A2AServer:
    """Configure once, build any transport.

    Concurrent transports against the same ``A2AServer`` instance share the
    same in-process ``AG2AgentExecutor`` and (by default) the same
    ``InMemoryTaskStore`` — useful for serving JSON-RPC and gRPC simultaneously.
    """

    __slots__ = (
        "_agent",
        "_card",
        "_executor",
        "_extended_card",
        "_push_config_store",
        "_push_sender",
        "_task_store",
        "_url",
    )

    def __init__(
        self,
        agent: "Agent",
        *,
        url: str = "http://localhost:8000",
        # Card customisation (all ignored when `card=` is passed):
        card: AgentCard | None = None,
        version: str = "0.1.0",
        description: str | None = None,
        capabilities: AgentCapabilities | None = None,
        default_input_modes: Sequence[str] | None = None,
        default_output_modes: Sequence[str] | None = None,
        extensions: Sequence[AgentExtension] | None = None,
        extended_card: AgentCard | None = None,
        supports_client_tools: bool = True,
        # Executor pipeline:
        executor_middleware: Iterable[ExecutorMiddleware] = (),
        # Storage:
        task_store: "TaskStore | None" = None,
        push_config_store: "PushNotificationConfigStore | None" = None,
        push_sender: "PushNotificationSender | None" = None,
    ) -> None:
        self._agent = agent
        self._url = url
        if card is not None:
            _warn_ignored_card_kwargs(
                version=version,
                description=description,
                capabilities=capabilities,
                default_input_modes=default_input_modes,
                default_output_modes=default_output_modes,
                extensions=extensions,
            )
            self._card = card
        else:
            self._card = build_card(
                agent,
                url=url,
                version=version,
                description=description,
                capabilities=capabilities,
                default_input_modes=default_input_modes,
                default_output_modes=default_output_modes,
                extensions=extensions,
                supports_extended=extended_card is not None,
                supports_client_tools=supports_client_tools,
            )
        if extended_card is not None and not self._card.capabilities.extended_agent_card:
            warnings.warn(
                "extended_card was provided but the supplied `card` does not advertise "
                "`capabilities.extended_agent_card=True`; A2A clients will not fetch the "
                "extended card. Set the flag on `card.capabilities` to expose it.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._extended_card = extended_card
        self._task_store = task_store
        self._push_config_store = push_config_store
        self._push_sender = push_sender
        self._executor = AG2AgentExecutor(agent, middleware=executor_middleware)

    @property
    def agent(self) -> "Agent":
        return self._agent

    @property
    def url(self) -> str:
        return self._url

    @property
    def card(self) -> AgentCard:
        return self._card

    @property
    def extended_card(self) -> AgentCard | None:
        return self._extended_card

    @property
    def executor(self) -> AG2AgentExecutor:
        return self._executor

    def build_request_handler(
        self,
        *,
        task_store: "TaskStore | None" = None,
        push_config_store: "PushNotificationConfigStore | None" = None,
        push_sender: "PushNotificationSender | None" = None,
    ) -> "RequestHandler":
        """Build the ``DefaultRequestHandler`` shared by all built-in transports.

        Caller can hand the handler to a custom transport adapter (e.g. an HTTP
        framework not in this package's transports list) without rebuilding the
        executor or task store. Per-call overrides take precedence over the
        instance defaults set in ``__init__``.

        If ``push_config_store`` / ``push_sender`` are missing on both the call
        and the instance, the underlying handler returns
        ``UnsupportedOperationError`` for push-notification ops — same wire
        behaviour clients see when an A2A server doesn't support webhooks.
        """
        return DefaultRequestHandler(
            agent_executor=self._executor,
            task_store=task_store or self._task_store or InMemoryTaskStore(),
            agent_card=self._card,
            push_config_store=push_config_store or self._push_config_store,
            push_sender=push_sender or self._push_sender,
            extended_agent_card=self._extended_card,
        )

    def build_asgi(self, **kwargs: Any) -> "Starlette":
        """Starlette ASGI app speaking JSON-RPC."""
        return _build_asgi(self, **kwargs)

    def build_rest(self, **kwargs: Any) -> "Starlette":
        """Starlette ASGI app speaking the HTTP+JSON/REST binding (A2A spec §11)."""
        return _build_rest(self, **kwargs)

    def build_grpc(self, **kwargs: Any) -> "GrpcServer":
        """``grpc.aio.Server`` speaking the gRPC binding (A2A spec §10).

        With no kwargs, returns a fresh server without bound ports — caller
        adds ``add_insecure_port`` / ``add_secure_port`` and ``await server.start()``.
        Pass ``host`` + ``port`` to bind, or ``grpc_server=existing`` to register
        the A2A service on an already-built server.
        """
        return _build_grpc(self, **kwargs)


def _warn_ignored_card_kwargs(
    *,
    version: str,
    description: str | None,
    capabilities: AgentCapabilities | None,
    default_input_modes: Sequence[str] | None,
    default_output_modes: Sequence[str] | None,
    extensions: Sequence[AgentExtension] | None,
) -> None:
    """Warn when ``A2AServer(card=..., <other kwarg>)`` would silently drop kwargs."""
    ignored: list[str] = []
    if version != "0.1.0":
        ignored.append("version")
    if description is not None:
        ignored.append("description")
    if capabilities is not None:
        ignored.append("capabilities")
    if default_input_modes is not None:
        ignored.append("default_input_modes")
    if default_output_modes is not None:
        ignored.append("default_output_modes")
    if extensions is not None:
        ignored.append("extensions")
    if ignored:
        warnings.warn(
            f"A2AServer received `card=` together with card-customisation kwargs "
            f"({', '.join(ignored)}); these kwargs are ignored — they only apply when "
            f"the server builds the card itself.",
            UserWarning,
            stacklevel=3,
        )

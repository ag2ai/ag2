# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

from .card import build_card
from .executor import AG2AgentExecutor

if TYPE_CHECKING:
    from a2a.server.apps.jsonrpc.jsonrpc_app import CallContextBuilder
    from a2a.server.request_handlers import RequestHandler
    from a2a.server.tasks import TaskStore
    from starlette.applications import Starlette
    from starlette.middleware import Middleware

    from autogen.beta import Agent


class A2AServer:
    __slots__ = ("_agent", "_card", "_executor", "_extended_card", "_task_store", "_url")

    def __init__(
        self,
        agent: "Agent",
        *,
        card: AgentCard | None = None,
        extended_card: AgentCard | None = None,
        url: str = "http://localhost:8000",
        task_store: "TaskStore | None" = None,
    ) -> None:
        self._agent = agent
        self._url = url
        self._card = card or build_card(agent, url=url, supports_extended=extended_card is not None)
        if extended_card is not None and not self._card.supports_authenticated_extended_card:
            warnings.warn(
                "extended_card was provided but the supplied `card` does not advertise "
                "`supports_authenticated_extended_card=True`; A2A clients will not fetch the "
                "extended card. Set the flag on `card` to expose it.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._extended_card = extended_card
        self._task_store = task_store
        self._executor = AG2AgentExecutor(agent)

    @property
    def card(self) -> AgentCard:
        return self._card

    @property
    def extended_card(self) -> AgentCard | None:
        return self._extended_card

    @property
    def executor(self) -> AG2AgentExecutor:
        return self._executor

    def build_asgi(
        self,
        *,
        middleware: Iterable["Middleware"] | None = None,
        request_handler: "RequestHandler | None" = None,
        context_builder: "CallContextBuilder | None" = None,
    ) -> "Starlette":
        """Construct a Starlette ASGI application that serves the A2A protocol."""
        handler = request_handler or DefaultRequestHandler(
            agent_executor=self._executor,
            task_store=self._task_store or InMemoryTaskStore(),
        )
        app = A2AStarletteApplication(
            agent_card=self._card,
            extended_agent_card=self._extended_card,
            http_handler=handler,
            context_builder=context_builder,
        ).build()
        for mw in middleware or ():
            app.add_middleware(mw.cls, **mw.options)
        return app

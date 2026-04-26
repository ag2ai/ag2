# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

from .card import build_card
from .executor import AgentExecutor

if TYPE_CHECKING:
    from a2a.server.tasks import TaskStore
    from starlette.applications import Starlette
    from starlette.middleware import Middleware

    from autogen.beta import Agent


class A2AServer:
    __slots__ = ("_agent", "_card", "_executor", "_task_store", "_url")

    def __init__(
        self,
        agent: "Agent",
        *,
        card: AgentCard | None = None,
        url: str = "http://localhost:8000",
        task_store: "TaskStore | None" = None,
    ) -> None:
        self._agent = agent
        self._url = url
        self._card = card or build_card(agent, url=url)
        self._task_store = task_store
        self._executor = AgentExecutor(agent)

    @property
    def card(self) -> AgentCard:
        return self._card

    @property
    def executor(self) -> AgentExecutor:
        return self._executor

    def build_asgi(
        self,
        *,
        middleware: Iterable["Middleware"] | None = None,
        request_handler: Any | None = None,
        context_builder: Any | None = None,
    ) -> "Starlette":
        """Construct a Starlette ASGI application that serves the A2A protocol."""
        handler = request_handler or DefaultRequestHandler(
            agent_executor=self._executor,
            task_store=self._task_store or InMemoryTaskStore(),
        )
        app = A2AStarletteApplication(
            agent_card=self._card,
            http_handler=handler,
            context_builder=context_builder,
        ).build()
        for mw in middleware or ():
            app.add_middleware(mw.cls, **mw.options)
        return app

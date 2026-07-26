# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any

from nlip_sdk.nlip import NLIP_Message
from nlip_server.server import NLIP_Application, NLIP_Session, setup_server

from ag2.agent import Agent

from .executor import NlipExecutor

logger = logging.getLogger(__name__)


class _AgentNlipSession(NLIP_Session):
    """NLIP session that executes a single turn of an AG2 ``Agent``."""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent
        self._executor = NlipExecutor(agent)

    async def start(self) -> None:
        await super().start()
        logger.info(f"Started NLIP session for agent: {self._agent.name}")

    async def execute(self, msg: NLIP_Message) -> NLIP_Message:
        logger.info(f"Executing agent {self._agent.name} with NLIP message")
        return await self._executor.execute(msg)

    async def stop(self) -> None:
        await super().stop()


class NlipServer:
    """Wrap an AG2 ``Agent`` as a NLIP endpoint.

    NLIP has no discovery card, no transports, and no task lifecycle, so
    there is exactly one way to serve it. The instance is itself an ASGI
    callable (built eagerly via ``nlip_server.setup_server`` so lifespan
    hooks are wired up before any request arrives); hand it directly to
    any ASGI server::

        import uvicorn
        from ag2 import Agent
        from ag2.extensions.nlip import NlipServer

        agent = Agent("assistant", config=...)
        server = NlipServer(agent)
        uvicorn.run(server, host="0.0.0.0", port=8000)
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._application = _AgentNlipApplication(agent)
        self._asgi_app = setup_server(self._application)
        self._lifespan_started = asyncio.Event()
        self._lifespan_shutdown = asyncio.Event()
        self._lifespan_task: asyncio.Task[None] | None = None
        self._lifespan_lock = asyncio.Lock()

    @property
    def agent(self) -> Agent:
        return self._agent

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self._asgi_app(scope, receive, send)
            return

        # ``nlip_server``'s FastAPI app stashes ``client_app`` on
        # ``app.state`` from its lifespan startup hook (see
        # ``nlip_server.server.create_app``). Real ASGI servers
        # (uvicorn/hypercorn) always send the lifespan handshake before any
        # request; ``httpx.ASGITransport`` (used by in-process tests) never
        # does. Drive the lifespan ourselves in the background the first
        # time a non-lifespan scope arrives so both paths work without
        # requiring callers to wire up a separate lifespan manager.
        if not self._lifespan_started.is_set():
            async with self._lifespan_lock:
                if self._lifespan_task is None:
                    self._lifespan_task = asyncio.ensure_future(self._drive_lifespan())
            await self._lifespan_started.wait()

        await self._asgi_app(scope, receive, send)

    async def _drive_lifespan(self) -> None:
        startup_requested = asyncio.Event()
        shutdown_requested = self._lifespan_shutdown

        async def receive() -> dict[str, Any]:
            if not startup_requested.is_set():
                startup_requested.set()
                return {"type": "lifespan.startup"}
            await shutdown_requested.wait()
            return {"type": "lifespan.shutdown"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] in ("lifespan.startup.complete", "lifespan.startup.failed"):
                self._lifespan_started.set()

        await self._asgi_app({"type": "lifespan"}, receive, send)


class _AgentNlipApplication(NLIP_Application):
    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent

    async def startup(self) -> None:
        logger.info(f"Starting NLIP application for agent: {self._agent.name}")

    async def shutdown(self) -> None:
        logger.info(f"Shutting down NLIP application for agent: {self._agent.name}")

    def create_session(self) -> NLIP_Session:
        return _AgentNlipSession(self._agent)


__all__: tuple[str, ...] = ("NlipServer",)

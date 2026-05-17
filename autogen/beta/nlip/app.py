# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Beta-Agent NLIP session and application wrappers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...import_utils import optional_import_block, require_optional_import

with optional_import_block() as _nlip_available:
    from nlip_sdk.nlip import NLIP_Factory, NLIP_Message
    from nlip_server.server import NLIP_Application, NLIP_Session, setup_server

if not _nlip_available.is_successful:
    NLIP_Session = object  # type: ignore[assignment,misc]
    NLIP_Application = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

__all__ = ("BetaNlipApplication", "BetaNlipSession")


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
class BetaNlipSession(NLIP_Session):
    """NLIP Session backed by a beta :class:`~autogen.beta.Agent`."""

    def __init__(self, agent: Agent[Any]) -> None:
        super().__init__()
        self._agent = agent

    async def start(self) -> None:
        await super().start()
        logger.info("Started NLIP session for beta agent: %s", self._agent.name)

    async def execute(self, msg: NLIP_Message) -> NLIP_Message:
        """Run the agent and return an NLIP response.

        The top-level text of ``msg`` is extracted and fed to the agent via
        :meth:`~autogen.beta.Agent.ask`.  The agent's reply body is returned
        as the top-level content of the response ``NLIP_Message``.
        """
        text = msg.extract_text() or ""
        logger.info("Executing beta agent %s with NLIP message (len=%d)", self._agent.name, len(text))

        reply = await self._agent.ask(text)
        response_text = reply.body or ""
        return NLIP_Factory.create_text(response_text, language="english")

    async def correlated_execute(self, msg: NLIP_Message) -> dict[str, Any]:  # type: ignore[override]
        response: NLIP_Message = await super().correlated_execute(msg)
        return response.model_dump(exclude_none=True)

    async def stop(self) -> None:
        logger.info("Stopping NLIP session for beta agent: %s", self._agent.name)
        await super().stop()


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
class BetaNlipApplication(NLIP_Application):
    """NLIP Application that serves a beta :class:`~autogen.beta.Agent`.

    Both an :class:`nlip_server.server.NLIP_Application` (so ``nlip-server``
    can call :meth:`create_session` / :meth:`startup` / :meth:`shutdown` on
    it) **and** an ASGI callable.  Pass directly to any ASGI server::

        import uvicorn
        from autogen.beta import Agent
        from autogen.beta.nlip import NLIPPlugin

        plugin = NLIPPlugin()
        agent = Agent("assistant", plugins=[plugin])
        uvicorn.run(plugin.build_asgi(), host="0.0.0.0", port=8000)
    """

    def __init__(self, agent: Agent[Any]) -> None:
        super().__init__()
        self._agent = agent
        self._asgi_app = setup_server(self)

    @property
    def asgi_app(self) -> Any:
        return self._asgi_app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """ASGI entrypoint — delegates to the wrapped nlip-server app."""
        await self._asgi_app(scope, receive, send)

    async def startup(self) -> None:
        logger.info("Starting NLIP application for beta agent: %s", self._agent.name)

    async def shutdown(self) -> None:
        logger.info("Shutting down NLIP application for beta agent: %s", self._agent.name)

    def create_session(self) -> BetaNlipSession:
        return BetaNlipSession(self._agent)

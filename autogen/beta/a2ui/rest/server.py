# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ..agent import A2UIAgent
from .asgi import build_jsonl_app, build_sse_app

if TYPE_CHECKING:
    from starlette.applications import Starlette


class A2UIServer:
    """Serve an :class:`~autogen.beta.a2ui.A2UIAgent` over HTTP as canonical A2UI.

    A transport-neutral REST/SSE adapter that depends only on Starlette (no
    ``ag-ui`` / ``a2a-sdk``). It mirrors ``autogen.beta.a2a.A2AServer``: hold the
    agent, expose ``build_*`` methods that each return a ready-to-serve Starlette
    ASGI app. Bring your own ``uvicorn`` (or any ASGI server) to run it.

    The server is **stateless** — clients send the full conversation on every
    request (see :func:`autogen.beta.a2ui.rest.parse_request` for the JSON body
    contract). Each turn runs on a fresh stream and the A2UI messages the
    validation middleware emits are streamed out as the canonical wire format.

    A2UI's wire is transport-agnostic, so two encodings are offered:

    - :meth:`build_sse_app` — Server-Sent Events (``text/event-stream``).
    - :meth:`build_jsonl_app` — canonical A2UI NDJSON (``application/x-ndjson``).

    Example::

        from autogen.beta.a2ui import A2UIAgent
        from autogen.beta.a2ui.rest import A2UIServer

        agent = A2UIAgent(name="ui", config=...)
        app = A2UIServer(agent).build_sse_app()  # serve with `uvicorn module:app`
    """

    __slots__ = ("_agent",)

    def __init__(self, agent: A2UIAgent) -> None:
        self._agent = agent

    @property
    def agent(self) -> A2UIAgent:
        return self._agent

    def build_sse_app(self, *, path: str = "/a2ui") -> "Starlette":
        """Starlette ASGI app serving the turn as SSE at ``path`` (POST)."""
        return build_sse_app(self._agent, path=path)

    def build_jsonl_app(self, *, path: str = "/a2ui") -> "Starlette":
        """Starlette ASGI app serving the turn as A2UI NDJSON at ``path`` (POST)."""
        return build_jsonl_app(self._agent, path=path)


__all__ = ("A2UIServer",)

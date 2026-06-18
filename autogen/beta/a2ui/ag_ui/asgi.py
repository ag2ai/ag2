# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Starlette ASGI app that serves :class:`A2UIAGUIServer` over HTTP. The turn is
streamed as AG-UI events (SSE by default, negotiated via the ``accept`` header).
Importing this module requires Starlette.
"""

import functools

from ag_ui.core import RunAgentInput
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from .server import A2UIAGUIServer


def build_app(server: "A2UIAGUIServer", *, path: str = "/") -> Starlette:
    """Build a Starlette app that streams the turn as AG-UI events at ``path`` (POST)."""
    endpoint = functools.partial(_endpoint, server)
    return Starlette(routes=[Route(path, endpoint, methods=["POST"])])


async def _endpoint(server: "A2UIAGUIServer", request: Request) -> Response:
    try:
        body = await request.body()
        incoming = RunAgentInput.model_validate_json(body)
    except Exception:  # noqa: BLE001 - bad/short body or disconnect → 400, not 500
        return Response('{"error": "invalid AG-UI RunAgentInput body"}', status_code=400, media_type="application/json")

    accept = request.headers.get("accept")
    return StreamingResponse(server.dispatch(incoming, accept=accept))


__all__ = ("build_app",)

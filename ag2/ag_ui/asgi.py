# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag_ui.core import RunAgentInput

try:
    from starlette.endpoints import HTTPEndpoint
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
except ImportError as e:
    raise ImportError("starlette is not installed. Please install it with:\npip install starlette") from e

if TYPE_CHECKING:
    from .stream import AGUIStream


def build_asgi(stream: "AGUIStream") -> type[HTTPEndpoint]:
    class AGUIEndpoint(HTTPEndpoint):
        async def get(
            endpoint,  # noqa: N805
            request: Request,
        ) -> JSONResponse:
            """Tell a client what this agent can do, before it starts a run.

            The protocol describes the capabilities document but defines neither
            an event nor a transport that carries it, so it is served from the
            same route the runs are posted to.
            """
            return JSONResponse(stream.capabilities().model_dump(by_alias=True, exclude_none=True))

        async def post(
            endpoint,  # noqa: N805
            request: Request,
        ) -> StreamingResponse:
            return StreamingResponse(
                stream.dispatch(
                    RunAgentInput.model_validate_json(await request.body()),
                    accept=request.headers.get("accept"),
                )
            )

    return AGUIEndpoint

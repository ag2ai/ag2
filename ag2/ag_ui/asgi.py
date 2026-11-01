# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag_ui.encoder import EventEncoder

from .run_input import read_run_input

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
            """Tell a client what this agent can do, before it starts a run."""
            return JSONResponse(stream.capabilities().model_dump(by_alias=True, exclude_none=True))

        async def post(
            endpoint,  # noqa: N805
            request: Request,
        ) -> StreamingResponse | JSONResponse:
            try:
                incoming = read_run_input(await request.body())
            except ValueError:
                # Refused before any stream: a run that never started has no
                # event stream for a RUN_ERROR to travel on.
                return JSONResponse({"error": "invalid AG-UI RunAgentInput body"}, status_code=400)
            accept = request.headers.get("accept")
            return StreamingResponse(
                stream.dispatch(incoming, accept=accept),
                # The encoder's own type, never the client's `Accept` copied back.
                media_type=EventEncoder(accept=accept).get_content_type(),  # type: ignore[arg-type]
            )

    return AGUIEndpoint

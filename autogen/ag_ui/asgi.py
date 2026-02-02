# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag_ui.core import RunAgentInput

try:
    from starlette.endpoints import HTTPEndpoint
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
except ImportError as e:
    raise ImportError("starlette is not installed. Please install it with:\npip install starlette") from e

if TYPE_CHECKING:
    from .adapter import AGUIStream


def build_asgi(stream: "AGUIStream") -> HTTPEndpoint:
    class AGUIEndpoint(HTTPEndpoint):
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

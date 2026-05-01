# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from fastapi import FastAPI
from starlette.routing import BaseRoute

if TYPE_CHECKING:
    from a2a.server.request_handlers import RequestHandler
    from a2a.server.routes.common import ServerCallContextBuilder

    from ..server import A2AServer


def build_fastapi(
    server: "A2AServer",
    *,
    rpc_url: str = "/",
    request_handler: "RequestHandler | None" = None,
    context_builder: "ServerCallContextBuilder | None" = None,
    enable_v0_3_compat: bool = False,
    app: FastAPI | None = None,
) -> FastAPI:
    """Construct (or augment) a FastAPI app speaking the JSON-RPC binding.

    Pass ``app=existing_fastapi`` to mount the A2A routes onto an existing app.
    """
    handler = request_handler or server.build_request_handler()
    routes: list[BaseRoute] = [
        *create_agent_card_routes(server.card),
        *create_jsonrpc_routes(
            handler,
            rpc_url=rpc_url,
            context_builder=context_builder,
            enable_v0_3_compat=enable_v0_3_compat,
        ),
    ]
    target = app or FastAPI()
    target.router.routes.extend(routes)
    return target

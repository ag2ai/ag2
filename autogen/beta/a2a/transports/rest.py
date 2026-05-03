# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import TYPE_CHECKING

from a2a.server.routes import create_agent_card_routes, create_rest_routes
from starlette.applications import Starlette

if TYPE_CHECKING:
    from a2a.server.request_handlers import RequestHandler
    from a2a.server.routes.common import ServerCallContextBuilder
    from starlette.middleware import Middleware

    from ..server import A2AServer


def build_rest_factory(
    server: "A2AServer",
    *,
    path_prefix: str = "",
    middleware: Iterable["Middleware"] | None = None,
    request_handler: "RequestHandler | None" = None,
    context_builder: "ServerCallContextBuilder | None" = None,
    enable_v0_3_compat: bool = False,
) -> Starlette:
    """Construct a Starlette app speaking the REST binding."""
    handler = request_handler or server.build_request_handler()
    routes = [
        *create_agent_card_routes(server.card),
        *create_rest_routes(
            handler,
            context_builder=context_builder,
            enable_v0_3_compat=enable_v0_3_compat,
            path_prefix=path_prefix,
        ),
    ]
    return Starlette(routes=routes, middleware=list(middleware) if middleware else None)

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import httpx

from .server import A2AServer


def make_test_client_factory(
    server: A2AServer,
    *,
    url: str = "http://test",
    timeout: float = 30.0,
) -> Callable[[], httpx.AsyncClient]:
    """Build an ``httpx.AsyncClient`` factory that talks to ``server`` in-process.

    Uses ``httpx.ASGITransport`` to dispatch directly into the Starlette
    app produced by ``server.build_jsonrpc(url=url)`` — no real socket,
    no port binding, no SSE proxy in the way. Use it as the
    ``httpx_client_factory`` on ``A2AConfig`` for end-to-end tests:

    .. code-block:: python

        server = A2AServer(agent)
        factory = make_test_client_factory(server, url="http://test")
        remote = Agent(
            "remote",
            config=A2AConfig(card_url="http://test", httpx_client_factory=factory),
        )
        await remote.ask("ping")

    The transport is created **once** and shared by every client the
    factory hands out, which matches how httpx.ASGITransport is meant to
    be reused. Each client returned by the factory is independent and
    closed by the caller (``A2AClient`` runs ``aclose()`` after each
    ``ask``); ``ASGITransport`` itself doesn't need explicit cleanup.
    """
    app = server.build_jsonrpc(url=url)
    transport = httpx.ASGITransport(app=app)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url=url, timeout=timeout)

    return factory

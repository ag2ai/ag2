# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import socket
from collections.abc import Callable

import httpx

from .card import build_card
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


def make_test_rest_client_factory(
    server: A2AServer,
    *,
    url: str = "http://test",
    timeout: float = 30.0,
) -> Callable[[], httpx.AsyncClient]:
    """Build an ``httpx.AsyncClient`` factory talking REST to ``server`` in-process.

    REST counterpart to :func:`make_test_client_factory`. Builds the
    Starlette REST app via ``server.build_rest(url=url)`` and dispatches
    through ``httpx.ASGITransport``. The card served at ``url`` declares
    only the REST interface, so an ``A2AConfig(card_url=url, prefer="rest")``
    client connects through this factory without leaking to other
    transports.
    """
    rest_card = build_card(server.agent, url=url, transports=("rest",))
    app = server.build_rest(url=url, card=rest_card)
    transport = httpx.ASGITransport(app=app)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url=url, timeout=timeout)

    return factory


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Find an available TCP port on ``host``.

    Used by gRPC tests that need a real listening socket — there is no
    in-process equivalent of ``httpx.ASGITransport`` for gRPC, so the
    test must bind a real port. Race-prone in theory (the port can be
    snatched between the probe and the gRPC server's ``start()``) but
    fine in practice for sequential local test runs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]

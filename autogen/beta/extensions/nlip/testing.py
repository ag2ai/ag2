# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import httpx

from .server import NlipServer


def make_test_client_factory(
    server: NlipServer,
    *,
    url: str = "http://test",
    timeout: float = 30.0,
) -> Callable[[], httpx.AsyncClient]:
    """Build an ``httpx.AsyncClient`` factory that talks to ``server`` in-process.

    Uses ``httpx.ASGITransport`` to dispatch directly into ``server`` — no
    real socket, no port binding. Use it as the ``httpx_client_factory`` on
    ``NlipConfig`` for end-to-end tests:

    .. code-block:: python

        server = NlipServer(agent)
        factory = make_test_client_factory(server)
        remote = Agent("remote", config=NlipConfig(url="http://test", httpx_client_factory=factory))
        await remote.ask("ping")

    The transport is created **once** and shared by every client the
    factory hands out; each client returned by the factory is independent
    and closed by the caller (``NlipClient`` closes it after each request).

    Note: this bypasses the FastAPI lifespan that ``nlip_server`` relies on
    to populate ``app.state.client_app`` — callers that need the lifespan
    to run (e.g. to exercise ``Agent`` startup/shutdown hooks) should drive
    ``server`` through ``asgi_lifespan.LifespanManager`` instead.
    """
    transport = httpx.ASGITransport(app=server)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url=url, timeout=timeout)

    return factory


__all__: tuple[str, ...] = ("make_test_client_factory",)

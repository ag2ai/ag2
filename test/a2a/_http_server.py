# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serving an A2A app over a real socket.

Kept out of ``_helpers`` because it is the only thing in ``test/a2a`` that
needs ``uvicorn``, which no A2A extra pulls in — importing it from
``_helpers`` would make an unrelated optional dependency a hard
requirement for *collecting* the whole package.

The skip guard lives here rather than in the importing test, so it covers
whoever imports this module next without them having to know why.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("uvicorn")

import uvicorn

from ag2.a2a.testing import pick_free_port


@asynccontextmanager
async def serve_over_http(build_app: Callable[[str], Any], host: str = "127.0.0.1") -> AsyncGenerator[str]:
    """Serve an ASGI app on a real socket, yielding the URL it is reachable at.

    Almost every test should prefer ``make_test_client_factory`` — it is
    faster and binds nothing. This exists for the contract a factory would
    hide: supplying ``httpx_client_factory`` is precisely what makes
    ``A2AConfig`` skip building its own httpx client, so anything AG2
    configures *on* that client is only observable when AG2 builds it
    itself, against a real URL.

    ``build_app`` receives the resolved URL, since the AgentCard has to
    advertise the port the app ends up on.
    """
    port = pick_free_port(host)
    url = f"http://{host}:{port}"
    server = uvicorn.Server(uvicorn.Config(build_app(url), host=host, port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if serving.done():
                serving.result()  # Surface startup's own error rather than hang.
                raise RuntimeError(f"uvicorn exited before serving {url}")
            await asyncio.sleep(0.005)
        yield url
    finally:
        server.should_exit = True
        await serving

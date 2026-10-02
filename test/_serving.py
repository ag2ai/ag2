# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serving an ASGI app under ``uvicorn`` on a loopback port.

Kept out of ``test/_helpers.py`` because it needs ``uvicorn``, which ships with
``ag2[acp]`` rather than ``ag2[mcp]`` — importing it from there would make an
unrelated optional dependency a hard requirement for *collecting* every package
that uses those helpers. The skip guard lives here rather than in the importing
test, so it covers whoever imports this module next.
"""

import asyncio
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("sse_starlette")

import uvicorn
from sse_starlette.sse import AppStatus


@asynccontextmanager
async def serving(app: Any) -> AsyncGenerator[str]:
    """Run ``app`` under ``uvicorn`` on a loopback port, yielding its base URL."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    # Bound here so the port is known before start-up, but `bind_socket` only
    # binds — the socket is not accepting until `serve()` reaches `startup()`,
    # which is why `started` below is what has to be waited on. Connecting
    # earlier is refused outright on Linux and merely slow elsewhere.
    sock = config.bind_socket()
    # `bind_socket` leaves the listener's `proto` at 0, and `asyncio` only sets
    # `TCP_NODELAY` on an accepted socket whose `proto` is `IPPROTO_TCP`. Nagle
    # therefore stays on server-side, and because a response goes out as two
    # writes (headers, then body) the second one waits for the client's ACK —
    # 40ms of delayed-ACK per response on Linux, ~0ms elsewhere. Set here
    # because accepted sockets inherit it from the listener.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server = uvicorn.Server(config)
    serving_task = asyncio.create_task(server.serve(sockets=[sock]))
    drain = AppStatus.should_exit
    try:
        while not server.started:
            if serving_task.done():
                serving_task.result()  # Surface start-up's own error rather than hang.
                raise RuntimeError("uvicorn exited before it began serving")
            await asyncio.sleep(0.005)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await serving_task
        sock.close()
        # `sse_starlette` watches for shutdown from a task that outlives a stream
        # the client walked away from, and on seeing one it latches `should_exit`
        # *process-wide*. Left set, every later SSE response in this process
        # drains the moment it opens, which surfaces as "SSE stream ended without
        # a response" in whichever test serves next — a module away from the
        # cause. Each server leaves the flag as it found it.
        AppStatus.should_exit = drain

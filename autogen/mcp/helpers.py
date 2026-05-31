# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import signal
from asyncio.subprocess import PIPE, Process, create_subprocess_exec
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


@asynccontextmanager
async def run_streamable_http_client(
    *, mcp_server_path: str, env_vars: dict[str, str] | None = None, startup_wait_secs: float = 5.0
) -> AsyncGenerator[Process, None]:
    """Async context manager to run a Python subprocess for streamable-http with custom env vars.

    Args:
        mcp_server_path: Path to the Python script to run.
        env_vars: Environment variables to export to the subprocess.
        startup_wait_secs: Time to wait for the server to start (in seconds).

    Yields:
        An asyncio.subprocess.Process object.
    """
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    process = await create_subprocess_exec(
        "python", mcp_server_path, "streamable-http", env=env, stdout=PIPE, stderr=PIPE
    )

    # Optional startup delay to let the server initialize
    await asyncio.sleep(startup_wait_secs)

    try:
        yield process
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()

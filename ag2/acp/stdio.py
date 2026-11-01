# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Launching an ACP agent as a subprocess and talking to it over its stdio."""

import asyncio
import sys
from asyncio.subprocess import Process
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import TracebackType

import acp
from acp.core import ClientSideConnection
from acp.transports import spawn_stdio_transport


class StdioConnection:
    """Connection to an ACP agent launched as a subprocess; an async context manager.

    Does what ``acp.spawn_agent_process`` does, as a plain class instead of a generator.
    That function is a generator wrapped around another generator, and a connection left
    open when the event loop shuts down has both finalized at once, which makes them fail
    ("asynchronous generator is already running"). Here only the SDK's transport generator
    remains, so there is nothing to race and an unclosed session ends quietly.
    """

    def __init__(
        self,
        client: acp.Client,
        command: str,
        *args: str,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._client = client
        self._launch = (command, *args)
        self._env = env
        self._cwd = cwd
        self._transport: (
            AbstractAsyncContextManager[tuple[asyncio.StreamReader, asyncio.StreamWriter, Process]] | None
        ) = None
        self._conn: ClientSideConnection | None = None

    async def __aenter__(self) -> tuple[ClientSideConnection, Process]:
        command, *args = self._launch
        self._transport = spawn_stdio_transport(command, *args, env=self._env, cwd=self._cwd)
        reader, writer, process = await self._transport.__aenter__()
        try:
            # `use_unstable_protocol` is what registers the `elicitation/*` routes on the
            # client side; without it the SDK answers the agent's question with
            # method-not-found before the bridge ever sees it. Elicitation is the only
            # unstable route a Client serves, so this enables nothing else.
            self._conn = ClientSideConnection(self._client, writer, reader, use_unstable_protocol=True)
        except BaseException:
            await self._transport.__aexit__(*sys.exc_info())
            raise
        return self._conn, process

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._transport is not None and self._conn is not None, "closed before it was opened"
        try:
            await self._conn.close()
        finally:
            await self._transport.__aexit__(exc_type, exc, tb)

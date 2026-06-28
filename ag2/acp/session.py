# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Persistent ACP session bound to one AG2 agent run.

The subprocess + ACP session are created on first use and reused across turns;
only the *new* human input since the last turn is sent to the live session,
tracked by a high-water mark over the run's ``ModelRequest`` events.
"""

from asyncio.subprocess import Process
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import acp

from ag2.events import BaseEvent, ModelRequest, TextInput

if TYPE_CHECKING:
    from .bridge import ACPBridge
    from .config import ConnectHook


def new_prompt_text(messages: Sequence[BaseEvent], sent_count: int) -> tuple[str, int]:
    """Return (text of ``ModelRequest`` turns beyond ``sent_count``, new request count).

    The user's input arrives as ``ModelRequest`` events carrying ``TextInput``
    parts. We forward only the requests not yet sent to the live ACP session,
    tracked by ``sent_count`` (a high-water mark over the run's request events).
    """
    requests = [m for m in messages if isinstance(m, ModelRequest)]
    new = requests[sent_count:]
    parts = [p.content for req in new for p in req.parts if isinstance(p, TextInput)]
    return "\n".join(parts), len(requests)


class ACPSession:
    """Live ACP connection + session id for one agent run.

    Lazily spawns the subprocess and creates the session on first ``ensure``;
    subsequent calls are no-ops. ``close`` tears the subprocess down.
    """

    def __init__(self) -> None:
        self.conn: acp.core.ClientSideConnection | None = None
        self.proc: Process | None = None
        self.bridge: ACPBridge | None = None  # the bridge bound to this connection
        self.session_id: str | None = None
        self.sent_count: int = 0
        # the spawn_agent_process async context manager
        self._cm: AbstractAsyncContextManager[tuple[acp.core.ClientSideConnection, Process]] | None = None

    @property
    def started(self) -> bool:
        return self.session_id is not None

    async def ensure(
        self,
        client: acp.Client,
        command: list[str],
        *,
        cwd: str,
        env: Mapping[str, str] | None,
        protocol_version: int,
        client_capabilities: acp.schema.ClientCapabilities | None = None,
        additional_directories: list[str] | None = None,
        connect: "ConnectHook | None" = None,
    ) -> None:
        """Spawn + initialize + create the session on first use; no-op afterwards.

        ``connect`` overrides how the connection is opened (tests inject an
        in-process agent); when ``None`` the real subprocess is spawned.

        Not concurrency-safe: callers rely on model-turns within a run being
        sequential (and on the per-stream session registry in ``ACPClient``) to
        avoid spawning two subprocesses for the same session.
        """
        if self.started:
            return

        if connect is not None:
            self._cm = connect(client)
        else:
            executable, *args = command
            self._cm = acp.spawn_agent_process(client, executable, *args, env=env, cwd=cwd)
        self.conn, self.proc = await self._cm.__aenter__()
        try:
            await self.conn.initialize(
                protocol_version=protocol_version,
                client_capabilities=client_capabilities,
            )
            session = await self.conn.new_session(
                cwd=cwd,
                additional_directories=additional_directories or None,
            )
        except BaseException:
            # initialize/new_session failed after the subprocess was spawned;
            # tear it down so a retry doesn't orphan this process.
            await self.close()
            raise
        self.session_id = session.session_id

    async def close(self) -> None:
        """Terminate the subprocess and reset the handle."""
        cm, self._cm = self._cm, None
        self.conn = None
        self.proc = None
        self.bridge = None
        self.session_id = None
        self.sent_count = 0
        if cm is not None:
            await cm.__aexit__(None, None, None)

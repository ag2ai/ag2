# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Persistent ACP session bound to one AG2 agent run.

The connection + ACP session are created on first use and reused across turns;
only the *new* human input since the last turn is sent to the live session,
tracked by a high-water mark over the run's ``ModelRequest`` events.

Transport-blind throughout: whether the agent is a local subprocess or a remote
endpoint is settled by the connection hook the config hands in, and the process
handle a session holds is optional for exactly that reason.
"""

import logging
from asyncio.subprocess import Process
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager, suppress
from difflib import get_close_matches
from typing import TYPE_CHECKING

import acp
from acp import schema

from ag2.events import BaseEvent, ModelRequest, TextInput

from .tool_gateway import MCPCapabilityError

if TYPE_CHECKING:
    from .bridge import ACPBridge
    from .config import ConnectHook
    from .tool_gateway import ToolGateway

logger = logging.getLogger(__name__)


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


def _model_option(session: schema.NewSessionResponse) -> schema.SessionConfigOptionSelect | None:
    """The agent's model picker, if it advertised one in ``session/new``."""
    for option in session.config_options or []:
        if isinstance(option, schema.SessionConfigOptionSelect) and option.id == "model":
            return option
    return None


def current_model(session: schema.NewSessionResponse) -> str | None:
    """The model the agent reports it is running, if it advertised a picker."""
    option = _model_option(session)
    return option.current_value if option is not None else None


def _model_values(option: schema.SessionConfigOptionSelect) -> list[str]:
    """The option's selectable values (entries are plain options or groups)."""
    values: list[str] = []
    for entry in option.options:
        if isinstance(entry, schema.SessionConfigSelectGroup):
            values.extend(choice.value for choice in entry.options)
        else:
            values.append(entry.value)
    return values


async def select_model(
    conn: "acp.core.ClientSideConnection",
    session: schema.NewSessionResponse,
    model: str,
) -> None:
    """Apply ``model`` via ``session/set_config_option``.

    No-op when the agent advertises no model picker (``model`` then stays
    response metadata, as before) or when it already runs the requested model.
    A value the agent does not offer raises ``ValueError`` up front rather
    than failing on the wire.
    """
    option = _model_option(session)
    if option is None:
        # Not an error: `model` legitimately degrades to response metadata on an
        # agent without a picker. Logged because the *same* typo hard-errors on
        # an agent that has one, and silence here is the confusing half of that.
        logger.debug("ACP agent advertises no model picker; model=%r not applied to the session", model)
        return
    if option.current_value == model:
        return
    values = _model_values(option)
    if model not in values:
        # Agents can offer hundreds of ids (Kilo advertises 320), so list close
        # matches rather than the whole set.
        close = get_close_matches(model, values, n=3)
        hint = f"; closest offered: {close}" if close else ""
        raise ValueError(f"model {model!r} is not offered by the ACP agent ({len(values)} offered){hint}")
    await conn.set_config_option(session_id=session.session_id, config_id=option.id, value=model)


class ACPSession:
    """Live ACP connection + session id for one agent run.

    Lazily opens the connection and creates the session on first ``ensure``;
    subsequent calls are no-ops. ``close`` tears the connection down, along with
    the subprocess behind it when there is one.
    """

    def __init__(self) -> None:
        self.conn: acp.core.ClientSideConnection | None = None
        # None for a connection with no subprocess behind it — a remote agent,
        # or the in-process double tests inject.
        self.proc: Process | None = None
        self.bridge: ACPBridge | None = None  # the bridge bound to this connection
        self.session_id: str | None = None
        # What the agent reports it is actually running: the selected model, or
        # the picker's current_value when nothing was selected. None when the
        # agent advertises no picker (nothing to report beyond config.model).
        self.model: str | None = None
        self.sent_count: int = 0
        self.gateway: ToolGateway | None = None
        self.external_servers: list[acp.schema.HttpMcpServer] = []
        # the config's connection hook, entered
        self._cm: AbstractAsyncContextManager[tuple[acp.core.ClientSideConnection, Process | None]] | None = None

    @property
    def started(self) -> bool:
        return self.session_id is not None

    async def ensure(
        self,
        client: acp.Client,
        *,
        connect: "ConnectHook",
        cwd: str,
        protocol_version: int,
        client_capabilities: acp.schema.ClientCapabilities | None = None,
        additional_directories: list[str] | None = None,
        model: str | None = None,
        mcp_servers: "Sequence[acp.schema.HttpMcpServer] | None" = None,
        agent_label: str = "acp-agent",
    ) -> None:
        """Connect + initialize + create the session on first use; no-op afterwards.

        ``connect`` is how the connection is opened — the config's own hook,
        which spawns a subprocess, dials a remote agent, or (in tests) yields an
        in-process double. This method never learns which.

        ``mcp_servers`` (when non-empty) requires the agent to advertise
        HTTP MCP capability in ``initialize``; otherwise
        :class:`~ag2.acp.MCPCapabilityError` is raised and the connection is
        torn down. ``agent_label`` names the agent in that error until the agent
        introduces itself in ``initialize``.

        Not concurrency-safe: callers rely on model-turns within a run being
        sequential (and on the per-stream session registry in ``ACPClient``) to
        avoid opening two connections for the same session.
        """
        if self.started:
            return

        self._cm = connect(client)
        self.conn, self.proc = await self._cm.__aenter__()
        try:
            init = await self.conn.initialize(
                protocol_version=protocol_version,
                client_capabilities=client_capabilities,
            )
            if mcp_servers:
                caps = init.agent_capabilities.mcp_capabilities if init.agent_capabilities else None
                if caps is None or not caps.http:
                    agent_name = (init.agent_info.name if init.agent_info else None) or agent_label
                    raise MCPCapabilityError(agent_name)
            session = await self.conn.new_session(
                cwd=cwd,
                additional_directories=additional_directories or None,
                mcp_servers=list(mcp_servers) if mcp_servers else None,
            )
            if model is not None:
                await select_model(self.conn, session, model)
            self.model = model or current_model(session)
        except BaseException:
            # initialize/new_session/select_model failed after the connection was
            # opened; tear it down so a retry doesn't orphan a process or socket.
            await self.close()
            raise
        self.session_id = session.session_id

    async def close(self) -> None:
        """Close the connection, shut down the tool gateway, reset the handles.

        The connection hook owns whatever is behind the connection, so this is
        the same call whether that is a subprocess to terminate or a socket to
        close; the ``proc`` backstop below simply has nothing to do when there
        is no process.

        Ordinary teardown failures are absorbed so they cannot mask the error
        that triggered the close (``ensure`` calls this from an ``except``
        block). Cancellation still propagates: ``CancelledError`` is a
        ``BaseException``, so it escapes the ``except Exception`` below, and
        ``gateway.close()`` re-raises it deliberately.
        """
        gateway, self.gateway = self.gateway, None
        cm, self._cm = self._cm, None
        proc, self.proc = self.proc, None
        self.conn = None
        self.bridge = None
        self.session_id = None
        self.model = None
        self.sent_count = 0
        self.external_servers = []
        try:
            # Connection first: dropping it kills any in-flight tools/call HTTP
            # requests, so the gateway shutdown that follows doesn't wait on them.
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    # Teardown noise — typically the connection's receive loop tripping
                    # over a notification that was in flight when the queue closed
                    # ("mssage queue already closed", sic, from `acp`). Routine enough
                    # to swallow: it fires on essentially every Kilo session teardown,
                    # and re-raising here would mask the error that caused the close.
                    # Still logged, or a genuinely wedged subprocess leaves no trace.
                    # The transport's own cleanup (wait → terminate → kill) has already
                    # run by the time it propagates here; the terminate is a backstop.
                    logger.debug("closing the ACP agent connection failed", exc_info=True)
                    if proc is not None and proc.returncode is None:
                        with suppress(ProcessLookupError):
                            proc.terminate()
        finally:
            if gateway is not None:
                await gateway.close()

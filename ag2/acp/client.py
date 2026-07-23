# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""``ACPClient`` — the :class:`LLMClient` that drives a CLI agent over ACP.

One AG2 model turn maps to one ACP ``session/prompt``. The agent's own tool loop
runs inside that single call; ``session/update`` notifications stream onto the
AG2 event stream via the bridge, and the accumulated text becomes a
``ModelResponse``.

Lifecycle: the framework calls ``config.create()`` once per ``AgentRun``, so the
live ACP session is keyed by ``context.stream.id`` in a per-config registry and
reused across the run's internal model-turns. A ``weakref.finalize`` on the
stream terminates the subprocess if the run is dropped without an explicit
``config.aclose()``. When the run's tools include function tools, a per-session
:class:`~.tool_gateway.ToolGateway` serves them to the agent over HTTP MCP; it
lives and dies with the session.
"""

import asyncio
import logging
import weakref
from asyncio.subprocess import Process
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING
from uuid import uuid4

import acp
from acp import schema

from ag2.context import ConversationContext
from ag2.events import BaseEvent
from ag2.events.types import ModelMessage, ModelResponse
from ag2.response import ResponseProto
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

from .bridge import make_bridge
from .mappers import map_usage
from .session import ACPSession, new_prompt_text
from .tool_gateway import ToolGateway, partition_tools

if TYPE_CHECKING:
    from fast_depends.library.serializer import SerializerProto

    from .config import ACPConfig

logger = logging.getLogger(__name__)


def _terminate_proc(proc: Process | None) -> None:
    """Best-effort synchronous subprocess termination (finalizer safety net)."""
    try:
        if proc is not None and proc.returncode is None:
            proc.terminate()
    except ProcessLookupError:
        pass


class ACPClient:
    """ACP client implementing :class:`LLMClient`, one live session per run."""

    def __init__(self, config: "ACPConfig") -> None:
        self.config = config

    def _client_capabilities(self) -> schema.ClientCapabilities:
        return schema.ClientCapabilities(
            fs=schema.FileSystemCapabilities(read_text_file=True, write_text_file=True),
            terminal=bool(self.config.allow_terminal),
        )

    async def _session_for(self, context: ConversationContext, tools: Sequence[ToolSchema]) -> ACPSession:
        key = context.stream.id
        session = self.config._sessions.get(key)
        if session is not None and session.started:
            if self.config.expose_tools:
                _refresh_tools(session, tools)
            return session

        session = ACPSession()
        session.bridge = make_bridge(self.config)

        mcp_servers: list[schema.HttpMcpServer] = []
        functions: list[FunctionToolSchema] = []
        external: list[schema.HttpMcpServer] = []
        if self.config.expose_tools:
            functions, external = partition_tools(tools)
            mcp_servers.extend(external)

        try:
            if functions:
                session.gateway = ToolGateway(
                    session.bridge.state, functions, startup_timeout=self.config.startup_timeout
                )
                await session.gateway.start()
                mcp_servers.insert(0, session.gateway.as_acp_server())
            await session.ensure(
                session.bridge,
                self.config.command,
                cwd=self.config.cwd,
                env=self.config.env,
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=self._client_capabilities(),
                additional_directories=self.config.additional_directories,
                mcp_servers=mcp_servers or None,
                connect=self.config._connect,
            )
        except BaseException:
            # ensure() closes itself on failure, but a gateway startup that
            # raised (or was cancelled) never reached ensure() — make
            # teardown unconditional so the HTTP server can't outlive this.
            await session.close()
            raise

        session.external_servers = external
        self.config._sessions[key] = session
        # Safety net: terminate the subprocess if the stream is dropped without
        # an explicit aclose(). Keyed on the stream, not the (per-run) client.
        weakref.finalize(context.stream, _terminate_proc, session.proc)
        return session

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: "ResponseProto | None",
        serializer: "SerializerProto",
    ) -> ModelResponse:
        session = await self._session_for(context, list(tools))
        bridge = session.bridge
        conn = session.conn
        session_id = session.session_id
        assert bridge is not None and conn is not None and session_id is not None  # ensured by _session_for
        state = bridge.state
        state.context = context
        state.begin_turn()

        text, new_count = new_prompt_text(messages, session.sent_count)

        async def _run_turn() -> schema.PromptResponse:
            return await conn.prompt(
                prompt=[acp.text_block(text)],
                session_id=session_id,
                message_id=str(uuid4()),
            )

        timed_out = False
        response: schema.PromptResponse | None = None
        if self.config.turn_timeout is not None:
            # Prefer cooperative cancellation: signal session/cancel and let the
            # agent return the in-flight prompt with stop_reason="cancelled".
            # Cancelling the coroutine outright would corrupt the JSON-RPC stream.
            task = asyncio.ensure_future(_run_turn())
            done, _ = await asyncio.wait({task}, timeout=self.config.turn_timeout)
            if task in done:
                response = await task
            else:
                timed_out = True
                await _cancel_quietly(session)
                # Bounded grace for the agent to honor the cancel.
                done, _ = await asyncio.wait({task}, timeout=self.config.cancel_timeout)
                if task in done:
                    response = await task
                else:
                    # Agent ignored the cancel; hard-stop so we never block the
                    # turn forever. The session is torn down and the next turn
                    # re-spawns it.
                    task.cancel()
                    # Drain the cancelled/broken prompt before tearing down.
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.debug("draining the hard-stopped prompt raised", exc_info=True)
                    await session.close()
        else:
            response = await _run_turn()

        if response is not None:
            session.sent_count = new_count

        finish_reason = "timeout" if timed_out else (response.stop_reason if response is not None else None)

        return ModelResponse(
            message=ModelMessage(state.turn_text),
            usage=map_usage(response.usage if response is not None else None),
            files=state.turn_files,
            finish_reason=finish_reason,
            provider="acp",
            model=self.config.model,
        )


def _refresh_tools(session: ACPSession, tools: Sequence[ToolSchema]) -> None:
    """Re-validate this turn's tools against the session created on turn one.

    Function tools hot-update the gateway's served list (``tools/list`` reads it
    live). Anything ACP cannot change mid-session is a hard error: the external
    ``mcp_servers`` set is fixed at ``session/new``, and a gateway cannot be
    added after the fact.
    """
    functions, external = partition_tools(tools)
    if external != session.external_servers:
        raise ValueError(
            "the run's MCPServerTool set changed after the ACP session was created; "
            "ACP fixes mcp_servers at session/new — keep the set stable for the whole run."
        )
    if session.gateway is not None:
        session.gateway.tools = functions
    elif functions:
        raise ValueError(
            "function tools appeared after the ACP session was created without a tool "
            "gateway; ACP fixes mcp_servers at session/new — provide the tools on the first turn."
        )


async def _cancel_quietly(session: ACPSession) -> None:
    try:
        if session.conn is not None and session.session_id is not None:
            await session.conn.cancel(session_id=session.session_id)
    except Exception as e:  # noqa: BLE001 — cancellation is best-effort
        logger.debug("session/cancel failed (best-effort): %s", e)

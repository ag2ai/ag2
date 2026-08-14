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
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import acp
from acp import schema
from fast_depends.library.serializer import SerializerProto

from ag2.context import ConversationContext
from ag2.events import BaseEvent
from ag2.events.types import ModelMessage, ModelResponse
from ag2.response import ResponseProto
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

from .bridge import make_bridge
from .mappers import map_usage
from .session import ACPSession, new_prompt_text
from .tool_gateway import GATEWAY_SERVER_NAME, ToolGateway, partition_tools
from .transport import ACPTransportError

if TYPE_CHECKING:
    from .config import ACPConfig, ElicitationPolicy

logger = logging.getLogger(__name__)

# Ceiling on waiting for received `session/update`s to finish being handled. Only
# approached if a handler is wedged — the normal wait is a scheduling round or
# two — and exceeding it costs the tail of the turn's text, never a hung turn.
_UPDATE_SETTLE_TIMEOUT = 10.0


def _elicitation_capabilities(policy: "ElicitationPolicy") -> schema.ElicitationCapabilities | None:
    """What ``initialize`` advertises for elicitation, per policy.

    ``"decline"`` advertises nothing at all rather than advertising support and
    refusing every request: the protocol already has a way to say "don't ask me",
    and using it saves the agent a round trip and a branch on the refusal.
    """
    if policy == "decline":
        return None
    return schema.ElicitationCapabilities(
        form=schema.ElicitationFormCapabilities(),
        url=schema.ElicitationUrlCapabilities(),
    )


def _terminate_proc(proc: Process | None) -> None:
    """Best-effort synchronous subprocess termination (finalizer safety net)."""
    try:
        if proc is not None and proc.returncode is None:
            proc.terminate()
    except ProcessLookupError:
        pass


class ACPClient:
    """ACP client implementing :class:`LLMClient`, one live session per run.

    Transport-blind: every difference between a locally-launched agent and a
    remote one is settled by the config, through the connection hook it opens
    and the gateway address it nominates.
    """

    def __init__(self, config: "ACPConfig") -> None:
        self.config = config

    def _client_capabilities(self) -> schema.ClientCapabilities:
        return schema.ClientCapabilities(
            fs=schema.FileSystemCapabilities(read_text_file=True, write_text_file=True),
            terminal=bool(self.config.allow_terminal),
            elicitation=_elicitation_capabilities(self.config.elicitation_policy),
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
        # Before `ensure`, not after: an elicitation scoped to a *request* rather
        # than a session (a pre-session auth flow) arrives during initialize, and
        # the bridge needs a context to reach the human with it.
        session.bridge.state.context = context

        mcp_servers: list[schema.HttpMcpServer] = []
        functions: list[FunctionToolSchema] = []
        external: list[schema.HttpMcpServer] = []
        if self.config.expose_tools:
            functions, external = partition_tools(tools)
            mcp_servers.extend(external)

        try:
            if functions:
                # Two identically-named entries in mcp_servers are not resolvable:
                # ACP does not define precedence, and agents namespace tools by
                # server name (mcp__<name>__<tool>), so one set would silently
                # shadow the other. Checked here, not in partition_tools, because
                # an external server named "ag2" is fine when no gateway is built.
                if any(server.name == GATEWAY_SERVER_NAME for server in external):
                    raise ValueError(
                        f"MCPServerTool server_label {GATEWAY_SERVER_NAME!r} collides with the name AG2 "
                        "uses for its own tool gateway in mcp_servers; rename that server."
                    )
                # Asked for before the server starts: a config whose agent could
                # never reach the gateway refuses here rather than handing out an
                # address that does not work.
                session.gateway = ToolGateway(
                    session.bridge.state,
                    functions,
                    address=self.config._gateway_address(),
                    startup_timeout=self.config.startup_timeout,
                )
                await session.gateway.start()
                mcp_servers.insert(0, session.gateway.as_acp_server())
            await session.ensure(
                session.bridge,
                connect=self.config._open_connection,
                cwd=self.config.cwd,
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=self._client_capabilities(),
                additional_directories=self.config.additional_directories,
                model=self.config.model,
                mcp_servers=mcp_servers or None,
                agent_label=self.config._agent_label,
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
        # A connection with no process behind it makes this a no-op.
        weakref.finalize(context.stream, _terminate_proc, session.proc)
        return session

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: "ResponseProto | None",
        serializer: SerializerProto,
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
            try:
                # ACP 0.11 dropped PromptRequest.message_id; extra kwargs would only
                # end up in the request's `_meta`, so send none.
                return await conn.prompt(
                    prompt=[acp.text_block(text)],
                    session_id=session_id,
                )
            except self.config._transport_errors as e:
                # A connection that dropped mid-turn must not read as an agent
                # with nothing to say; name the transport and fail the turn.
                raise ACPTransportError(self.config._transport_label, e) from e

        try:
            timed_out, response = await self._drive_turn(session, _run_turn)
        except ACPTransportError:
            # The connection is gone, so the session it carried is gone too:
            # drop it rather than leaving a started session with a dead
            # connection (and, for a remote agent, an open client) that the next
            # turn on this stream would reuse. Nothing is resumed or replayed —
            # a later turn starts a new session, as it does after a hard stop.
            self.config._sessions.pop(context.stream.id, None)
            await session.close()
            raise

        # The prompt response arriving does not mean the `session/update`s that
        # preceded it on the wire have been handled — see `dispatch`. Reading the
        # turn now would cut off its tail, so wait for them first. Bounded only
        # against a wedged update handler: the normal wait is the time it takes to
        # drain a queue whose items are already there.
        try:
            await asyncio.wait_for(state.updates.settle(), _UPDATE_SETTLE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out after %.0fs waiting for streamed updates to be handled; "
                "the reply may be missing its tail (agent=%r).",
                _UPDATE_SETTLE_TIMEOUT,
                self.config._agent_label,
            )

        if response is not None:
            session.sent_count = new_count

        finish_reason = "timeout" if timed_out else (response.stop_reason if response is not None else None)

        # What the agent says it is running beats what was asked for: with no
        # `model` set, config.model is None while the agent may be on a default
        # that cannot answer at all — exactly the case the warning below is for.
        model = session.model or self.config.model

        if finish_reason == "end_turn" and not state.turn_text and not state.turn_files and not state.turn_worked:
            # The agent reported a clean finish yet emitted nothing at all. Some
            # CLI agents end a turn this way when the provider call failed on
            # their side (an unauthorized model, or one that cannot do text) —
            # nothing reaches the ACP wire, so the empty reply would otherwise
            # be the only clue.
            logger.warning(
                "ACP agent ended the turn with stop_reason='end_turn' but produced no output "
                "(agent=%r, model=%r). The agent may have failed the provider call silently — "
                "check its own logs, and that the model is spelled right and authorized.",
                self.config._agent_label,
                model,
            )

        return ModelResponse(
            message=ModelMessage(state.turn_text),
            usage=map_usage(response.usage if response is not None else None),
            files=state.turn_files,
            finish_reason=finish_reason,
            provider="acp",
            model=model,
        )

    async def _drive_turn(
        self,
        session: ACPSession,
        run_turn: "Callable[[], Awaitable[schema.PromptResponse]]",
    ) -> "tuple[bool, schema.PromptResponse | None]":
        """Run one prompt turn under the configured timeout; report (timed out, response)."""
        timed_out = False
        response: schema.PromptResponse | None = None
        if self.config.turn_timeout is not None:
            # Prefer cooperative cancellation: signal session/cancel and let the
            # agent return the in-flight prompt with stop_reason="cancelled".
            # Cancelling the coroutine outright would corrupt the JSON-RPC stream.
            task = asyncio.ensure_future(run_turn())
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
                    # re-opens it — closing the connection where a remote agent
                    # leaves no process to kill.
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
            response = await run_turn()
        return timed_out, response


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

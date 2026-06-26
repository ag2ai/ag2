# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one A2UI agent turn on a fresh per-turn stream and yield it as
transport-neutral frames: one :class:`A2UIProseFrame` (conversational text)
followed by one :class:`A2UIMessageFrame` per A2UI message. Shared core under
the SSE / NDJSON wire encoders.
"""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ag2.agent import Agent
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelRequest, TextInput
from ag2.stream import MemoryStream

from ._runtime import _A2UIRuntime
from ._types import ServerToClientMessage
from .actions import A2UIAction
from .events import A2UIMessageEvent
from .incoming import A2UIIncomingActionResult
from .middleware import A2UIInboundMiddleware
from .request import A2UIServerRequest
from .server_action import build_server_action_context, run_server_action


@dataclass(slots=True)
class A2UIProseFrame:
    """The turn's conversational prose (A2UI-free assistant text)."""

    text: str


@dataclass(slots=True)
class A2UIMessageFrame:
    """A single canonical A2UI server→client message."""

    message: ServerToClientMessage


A2UIFrame = A2UIProseFrame | A2UIMessageFrame

# Shared immutable default so the keyword arg never aliases a mutable {}.
_NO_SERVER_ACTIONS: Mapping[str, A2UIAction] = MappingProxyType({})


async def stream_turn(
    agent: Agent,
    runtime: _A2UIRuntime,
    request: A2UIServerRequest,
    *,
    server_actions: Mapping[str, A2UIAction] = _NO_SERVER_ACTIONS,
) -> AsyncIterator[A2UIFrame]:
    """Execute one turn and yield its prose then A2UI message frames.

    Server-side actions are handled first and never invoke the agent: each
    incoming click whose name maps to a ``server_actions`` entry runs that
    action and its result is yielded as A2UI message frames. The agent then
    runs only if the turn still has input for it (a user message, or a click on
    a button with no registered action) — a turn carrying *only* server-side
    clicks skips the agent entirely.

    Args:
        agent: The plain ``Agent`` to run. Must have ``config`` set (unless the
            turn carries only server-side clicks, in which case it is not run).
        runtime: The A2UI runtime supplying the prompt section, validation
            middleware, and catalog/capabilities helpers.
        request: The parsed turn (history, current inputs, prompt, variables).
        server_actions: Action name → :class:`A2UIAction` for ``@a2ui_action``
            buttons, executed on click without invoking the agent.

    Yields:
        Any server-action :class:`A2UIMessageFrame`s first, then (when the agent
        runs) an :class:`A2UIProseFrame` for its prose followed by an
        :class:`A2UIMessageFrame` per A2UI message it produced.

    Raises:
        RuntimeError: If the agent must run but has no ``config`` to create an
            LLM client.
    """
    # Run server-side click actions and emit their messages. These never reach
    # the agent (the prompt rewriter already skipped registered actions).
    handled_server = False
    if server_actions:
        # Server actions resolve their dependencies against the agent's DI
        # surface (built once for the turn, only when a click actually runs).
        action_context = build_server_action_context(agent, variables=request.variables)
        for interaction in request.client_interactions:
            if not isinstance(interaction, A2UIIncomingActionResult):
                continue
            server_action = server_actions.get(interaction.action.name)
            if server_action is None:
                continue
            handled_server = True
            for message in await run_server_action(
                server_action,
                interaction.action,
                version=runtime.version_string,
                context=action_context,
            ):
                yield A2UIMessageFrame(message)

    # Run the agent only when the turn has real input for it. A turn whose only
    # content was server-side clicks is complete already; don't fabricate a
    # blank agent turn for it (but keep the blank-turn fallback otherwise).
    if not request.current_inputs and handled_server:
        return

    if agent.config is None:
        raise RuntimeError("Agent.config is not set; cannot serve over REST")
    client = agent.config.create()

    stream = MemoryStream()
    if request.history:
        await stream.history.replace(request.history)

    # The validation middleware emits one A2UIMessageEvent per validated A2UI
    # message onto this turn's stream. Collect them as the single source of UI
    # content (the event seam) — consistent with the A2A executor.
    a2ui_messages: list[ServerToClientMessage] = []

    @stream.subscribe
    async def _collect_a2ui_messages(event: BaseEvent) -> None:
        if isinstance(event, A2UIMessageEvent):
            a2ui_messages.append(event.message)

    # Apply A2UI behaviour to the plain agent for this turn: prepend the A2UI
    # prompt section, fold in negotiated client capabilities so the LLM only
    # targets components the client can render, and inject the validation
    # middleware that emits the A2UIMessageEvents collected above.
    caps_prompt = runtime.capabilities_prompt(request.client_capabilities)
    extra_prompt = [runtime.system_prompt_section, *([caps_prompt] if caps_prompt else [])]

    merged_variables = {**dict(agent._agent_variables), **request.variables}
    ctx = ConversationContext(
        stream,
        prompt=[*agent._system_prompt, *extra_prompt, *request.prompt],
        dependencies=dict(agent._agent_dependencies),
        variables=merged_variables,
        dependency_provider=agent.dependency_provider,
    )

    # Surface each incoming client→server interaction as an A2UIClientEvent on
    # the turn's stream (alongside the validation middleware), so observers see
    # client clicks/responses — not just the LLM via the rewritten prompt.
    extra_middleware = list(runtime.middleware_factories())
    if request.client_interactions:
        extra_middleware.append(A2UIInboundMiddleware(request.client_interactions))

    initial_event: BaseEvent = ModelRequest(request.current_inputs or [TextInput("")])
    reply = await agent._execute(
        initial_event,
        context=ctx,
        client=client,
        additional_middleware=extra_middleware,
    )

    response = reply.response
    prose = response.message.content if response.message else ""
    if prose:
        yield A2UIProseFrame(prose)
    for message in a2ui_messages:
        yield A2UIMessageFrame(message)


@dataclass(slots=True)
class _A2UITurnCore:
    """Transport-neutral turn engine shared by every transport.

    Bundles the plain ``Agent``, the configured ``_A2UIRuntime``, and the
    ``server_actions`` for clickable actions, so a transport can run one turn
    via :meth:`run_turn` without knowing how A2UI is wired. A click on a
    registered action runs its handler on the server without invoking the agent;
    a click on any other button is rewritten into a prompt for the agent.
    """

    agent: Agent
    runtime: _A2UIRuntime
    server_actions: Mapping[str, A2UIAction] = field(default_factory=dict)

    def run_turn(self, request: A2UIServerRequest) -> AsyncIterator[A2UIFrame]:
        """Run one turn and yield its prose then A2UI message frames."""
        return stream_turn(
            self.agent,
            self.runtime,
            request,
            server_actions=self.server_actions,
        )


__all__ = ("A2UIFrame", "A2UIMessageFrame", "A2UIProseFrame", "stream_turn")

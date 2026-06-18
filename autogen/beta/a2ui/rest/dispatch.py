# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one A2UI agent turn on a fresh per-turn stream and yield it as
transport-neutral frames: one :class:`A2UIProseFrame` (conversational text)
followed by one :class:`A2UIMessageFrame` per A2UI message. Shared core under
the SSE / NDJSON wire encoders.
"""

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

from autogen.beta.agent import Agent
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelRequest, TextInput
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.tool import Tool

from .._runtime import _A2UIRuntime
from .._types import ServerToClientMessage
from ..events import A2UIMessageEvent
from ..middleware import A2UIInboundMiddleware
from .request import A2UIServerRequest


@dataclass(slots=True)
class A2UIProseFrame:
    """The turn's conversational prose (A2UI-free assistant text)."""

    text: str


@dataclass(slots=True)
class A2UIMessageFrame:
    """A single canonical A2UI server→client message."""

    message: ServerToClientMessage


A2UIFrame = A2UIProseFrame | A2UIMessageFrame


async def stream_turn(
    agent: Agent,
    runtime: _A2UIRuntime,
    request: A2UIServerRequest,
    *,
    additional_tools: Iterable[Tool] = (),
) -> AsyncIterator[A2UIFrame]:
    """Execute one turn and yield its prose then A2UI message frames.

    Args:
        agent: The plain ``Agent`` to run. Must have ``config`` set.
        runtime: The A2UI runtime supplying the prompt section, validation
            middleware, and catalog/capabilities helpers.
        request: The parsed turn (history, current inputs, prompt, variables).
        additional_tools: Clickable A2UI actions to expose for this turn only
            (the ``A2UIActionTool``s passed to ``A2UIServer(actions=[...])``).
            The agent itself stays plain — the tools ride along just for the turn.

    Yields:
        An :class:`A2UIProseFrame` first (only when there is prose), then an
        :class:`A2UIMessageFrame` for each collected A2UI message.

    Raises:
        RuntimeError: If the agent has no ``config`` to create an LLM client.
    """
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
        additional_tools=additional_tools,
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
    clickable ``actions`` (``A2UIActionTool``s) so a transport can run one turn
    via :meth:`run_turn` without knowing how A2UI is wired. The actions are
    injected per-turn as ``additional_tools`` — the agent stays plain.
    """

    agent: Agent
    runtime: _A2UIRuntime
    action_tools: tuple[Tool, ...] = field(default_factory=tuple)

    def run_turn(self, request: A2UIServerRequest) -> AsyncIterator[A2UIFrame]:
        """Run one turn and yield its prose then A2UI message frames."""
        return stream_turn(self.agent, self.runtime, request, additional_tools=self.action_tools)


__all__ = ("A2UIFrame", "A2UIMessageFrame", "A2UIProseFrame", "stream_turn")

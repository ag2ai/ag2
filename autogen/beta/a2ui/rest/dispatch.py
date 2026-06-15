# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one A2UI agent turn and stream the result as transport-neutral frames.

This is the shared core under the REST/SSE adapter: it builds a fresh per-turn
``MemoryStream`` (the server is stateless), dispatches into the agent the same
way :class:`autogen.beta.a2a.AgentExecutor` does, and collects the
:class:`~autogen.beta.a2ui.A2UIMessageEvent`s the validation middleware emits —
the Phase A *event seam* — rather than re-parsing the model's text.

It yields :class:`A2UIProseFrame` (the conversational text) followed by one
:class:`A2UIMessageFrame` per A2UI message. Because Phase A emits per *whole*
message on the final, validated response (level A.1), the frames are produced
after the turn completes — this is not yet token-level streaming. The async
generator shape keeps the wire encoders (SSE / NDJSON) uniform and leaves room
for progressive (A.2) emission later.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelRequest, TextInput
from autogen.beta.stream import MemoryStream

from .._types import ServerToClientMessage
from ..agent import A2UIAgent
from ..events import A2UIMessageEvent
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


async def stream_turn(agent: A2UIAgent, request: A2UIServerRequest) -> AsyncIterator[A2UIFrame]:
    """Execute one turn and yield its prose then A2UI message frames.

    Args:
        agent: The A2UI agent to run. Must have ``config`` set.
        request: The parsed turn (history, current inputs, prompt, variables).

    Yields:
        An :class:`A2UIProseFrame` first (only when there is prose), then an
        :class:`A2UIMessageFrame` for each collected A2UI message.

    Raises:
        RuntimeError: If the agent has no ``config`` to create an LLM client.
    """
    if agent.config is None:
        raise RuntimeError("A2UIAgent.config is not set; cannot serve over REST")
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

    merged_variables = {**dict(agent._agent_variables), **request.variables}
    ctx = ConversationContext(
        stream,
        prompt=[*agent._system_prompt, *request.prompt],
        dependencies=dict(agent._agent_dependencies),
        variables=merged_variables,
        dependency_provider=agent.dependency_provider,
    )

    initial_event: BaseEvent = ModelRequest(request.current_inputs or [TextInput("")])
    reply = await agent._execute(initial_event, context=ctx, client=client)

    response = reply.response
    prose = response.message.content if response.message else ""
    if prose:
        yield A2UIProseFrame(prose)
    for message in a2ui_messages:
        yield A2UIMessageFrame(message)


__all__ = ("A2UIFrame", "A2UIMessageFrame", "A2UIProseFrame", "stream_turn")

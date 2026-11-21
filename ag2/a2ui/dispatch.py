# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one A2UI agent turn on a fresh per-turn stream and yield it as
transport-neutral frames: one :class:`A2UIProseFrame` (conversational text)
followed by one :class:`A2UIMessageFrame` per A2UI message. Shared core under
the SSE / NDJSON wire encoders.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Union

from ag2.agent import Agent
from ag2.annotations import Context
from ag2.context import ConversationContext, strip_reserved_variables
from ag2.events import (
    BaseEvent,
    HumanInputRequest,
    ModelRequest,
    TaskCancelled,
    TaskCompleted,
    TaskExpired,
    TaskFailed,
    TaskStarted,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from ag2.stream import MemoryStream
from ag2.usage import collect_usage_events

from ._runtime import _A2UIRuntime
from ._turn import open_turn
from ._types import ServerToClientMessage
from .request import A2UIServerRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class A2UIProseFrame:
    """The turn's conversational prose (A2UI-free assistant text)."""

    text: str


@dataclass(slots=True)
class A2UIMessageFrame:
    """A single canonical A2UI server→client message."""

    message: ServerToClientMessage


A2UIFrame = A2UIProseFrame | A2UIMessageFrame

# What a transport hands in to answer the agent's ``context.input()`` itself.
# Spelled structurally rather than imported: the transport that has one is the
# optional AG-UI one, and this module must import without it.
Interrupter = Callable[[HumanInputRequest, Context], Awaitable[BaseEvent | None]]

# A delegation starting or ending. ``Union``, not ``|``: on event classes ``|``
# builds a stream Condition, which is no type for a subscriber's annotation.
TaskEvent = Union[TaskStarted, TaskCompleted, TaskFailed, TaskCancelled, TaskExpired]  # noqa: UP007

# What a transport hands in to hear of each delegation starting and ending.
TaskObserver = Callable[[TaskEvent], Awaitable[None]]
ToolCallObserver = Callable[[ToolCallEvent], Awaitable[None]]
ToolResultObserver = Callable[[ToolResultEvent], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _A2UITurnCore:
    """The transport-neutral turn engine: the plain ``Agent`` and its A2UI runtime."""

    agent: Agent
    runtime: _A2UIRuntime

    async def run_turn(
        self,
        request: A2UIServerRequest,
        *,
        usage_records: list[UsageEvent] | None = None,
        interrupter: Interrupter | None = None,
        on_task: TaskObserver | None = None,
        on_tool_call: ToolCallObserver | None = None,
        on_tool_result: ToolResultObserver | None = None,
    ) -> AsyncIterator[A2UIFrame]:
        """Execute one turn and yield its prose then A2UI message frames.

        Registered ``@a2ui_action`` clicks and v1.0 ``callAgentFunction`` calls are answered first,
        without the agent. The agent runs only if the turn still has input for it.

        Args:
            request: The parsed turn (history, current inputs, prompt, variables).
            usage_records: Filled with this turn's :class:`~ag2.events.UsageEvent`
                events as they are sent, so the caller has them even when the turn raises.
            interrupter: Where a question the agent asks goes. Supplied only by a
                transport that can put it to whoever is connected, and only when the
                agent has no hook of its own.
            on_task: Called with each ``TaskStarted`` / ``TaskCompleted`` /
                ``TaskFailed`` / ``TaskCancelled`` / ``TaskExpired`` on the turn's
                stream, for a transport that reports delegations.
            on_tool_call: Called with each tool call on the turn's stream.
            on_tool_result: Called with each tool result on the turn's stream.

        Yields:
            Any server-action :class:`A2UIMessageFrame`s first, then (when the agent
            runs) an :class:`A2UIProseFrame` for its prose followed by an
            :class:`A2UIMessageFrame` per A2UI message it produced.

        Raises:
            RuntimeError: If the agent must run but has no ``config`` to create an
                LLM client.
        """
        agent = self.agent
        # Only the catalogs the renderer supports are offered; with none in common the turn has no A2UI.
        turn = open_turn(self.runtime, request.renderer)
        # Answer the interactions the server handles itself (registered clicks and
        # callAgentFunction calls) and emit their messages. These never reach the
        # agent (the prompt rewriter already skipped them).
        answers = await turn.answer_interactions(agent, request.variables)
        for message in answers.messages:
            yield A2UIMessageFrame(message)

        # Run the agent only when the turn has real input for it. A turn whose only
        # content was server-side clicks is complete already; don't fabricate a
        # blank agent turn for it (but keep the blank-turn fallback otherwise).
        if answers.complete_without_agent(has_input=bool(request.current_inputs)):
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
        a2ui_messages = turn.collect(stream)

        if usage_records is not None:
            stream.where(UsageEvent).subscribe(collect_usage_events(usage_records))
        if on_task is not None:
            stream.where((TaskStarted, TaskCompleted, TaskFailed, TaskCancelled, TaskExpired)).subscribe(on_task)
        if on_tool_call is not None:
            stream.where(ToolCallEvent).subscribe(on_tool_call)
        if on_tool_result is not None:
            stream.where(ToolResultEvent).subscribe(on_tool_result)

        merged_variables = {
            **dict(agent._agent_variables),
            **strip_reserved_variables(request.variables, source="an inbound A2UI request"),
        }
        # Apply A2UI behaviour to the plain agent for this turn: the turn's A2UI prompt
        # fragments follow the agent's own prompt, and its middleware validates the A2UI
        # output (emitting the A2UIMessageEvents collected above) and surfaces each
        # incoming client→server interaction as an A2UIClientEvent.
        ctx = ConversationContext(
            stream,
            prompt=[*agent._system_prompt, *turn.system_prompt, *request.prompt],
            dependencies=dict(agent._agent_dependencies),
            variables=merged_variables,
            dependency_provider=agent.dependency_provider,
        )

        initial_event: BaseEvent = ModelRequest(request.current_inputs or [TextInput("")])
        with ExitStack() as stack:
            if interrupter is not None:
                stack.enter_context(stream.where(HumanInputRequest).sub_scope(interrupter, interrupt=True))

            reply = await agent._execute(
                initial_event,
                context=ctx,
                client=client,
                additional_middleware=turn.middleware,
            )

        response = reply.response
        prose = response.message.content if response.message else ""
        if prose:
            yield A2UIProseFrame(prose)
        for message in turn.deliverable(a2ui_messages):
            yield A2UIMessageFrame(message)


__all__ = (
    "A2UIFrame",
    "A2UIMessageFrame",
    "A2UIProseFrame",
    "Interrupter",
    "TaskEvent",
    "TaskObserver",
    "ToolCallObserver",
    "ToolResultObserver",
)

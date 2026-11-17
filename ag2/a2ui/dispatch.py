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

from ._runtime import A2UIMessageCollector, _A2UIRuntime
from ._types import ServerToClientMessage
from .data_model import data_model_to_prompt
from .events import A2UIMessageEvent
from .request import A2UIServerRequest
from .server_action import run_client_interactions

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
        agent, runtime = self.agent, self.runtime
        # Only the catalogs the renderer supports are offered; with none in common the turn has no A2UI.
        offer = runtime.offer_for(request.client_capabilities)
        # Answer the interactions the server handles itself (registered clicks and
        # callAgentFunction calls) and emit their messages. These never reach the
        # agent (the prompt rewriter already skipped them).
        server_messages, handled_server = await run_client_interactions(
            request.client_interactions,
            runtime,
            agent,
            offer=offer,
            variables=request.variables,
        )
        for message in server_messages:
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
        stream.where(A2UIMessageEvent).subscribe(A2UIMessageCollector(a2ui_messages))

        if usage_records is not None:
            stream.where(UsageEvent).subscribe(collect_usage_events(usage_records))
        if on_task is not None:
            stream.where((TaskStarted, TaskCompleted, TaskFailed, TaskCancelled, TaskExpired)).subscribe(on_task)
        if on_tool_call is not None:
            stream.where(ToolCallEvent).subscribe(on_tool_call)
        if on_tool_result is not None:
            stream.where(ToolResultEvent).subscribe(on_tool_result)

        # Apply A2UI behaviour to the plain agent for this turn: prepend the A2UI
        # prompt section for the catalogs this renderer supports, add the client's data model, and inject the validation
        # middleware that emits the A2UIMessageEvents collected above.
        caps_prompt = runtime.capabilities_prompt(request.client_capabilities, offer)
        data_model_prompt = data_model_to_prompt(request.client_data_model)
        extra_prompt = [
            *([offer.system_prompt_section] if offer.system_prompt_section else []),
            *([caps_prompt] if caps_prompt else []),
            *([data_model_prompt] if data_model_prompt else []),
        ]

        merged_variables = {
            **dict(agent._agent_variables),
            **strip_reserved_variables(request.variables, source="an inbound A2UI request"),
        }
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
        extra_middleware = runtime.turn_middleware(request.client_interactions, offer)

        initial_event: BaseEvent = ModelRequest(request.current_inputs or [TextInput("")])
        with ExitStack() as stack:
            if interrupter is not None:
                stack.enter_context(stream.where(HumanInputRequest).sub_scope(interrupter, interrupt=True))

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
        if a2ui_messages and not offer.sends_ui:
            logger.warning("The renderer offers no A2UI this turn; dropping %d A2UI message(s).", len(a2ui_messages))
            return
        for message in a2ui_messages:
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

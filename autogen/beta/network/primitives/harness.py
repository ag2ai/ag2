# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Context Harness — defines what events the LLM sees and how they're formatted.

The Harness is the bridge between the event stream (ALL events) and the LLM
context window. It makes the Actor truly general-purpose: the same stream
infrastructure, same observer system, same middleware chain — but a different
harness transforms a chat agent into a network coordinator or a domain-specific
processor.

Implemented as a middleware (HarnessMiddleware) that wraps on_llm_call. This
bridges primitives to core via the existing middleware chain — zero changes to
Layer 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from autogen.beta.context import Context
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse
from autogen.beta.events.base import Field
from autogen.beta.events.tool_events import (
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.middleware.base import BaseMiddleware, LLMCall


@runtime_checkable
class ContextHarness(Protocol):
    """Defines how stream events are assembled into LLM context.

    The harness controls what the LLM sees. Different actors need different
    context — a conversation agent sees chat history, a coordinator sees
    delegation results, a monitor sees system events.
    """

    def select(self, events: list[BaseEvent]) -> list[BaseEvent]:
        """Select which events from the stream history are relevant for the LLM.

        Called before each LLM invocation. Receives the full stream history.
        Returns the subset of events that should form the LLM context.
        """
        ...

    def format(self, event: BaseEvent) -> str | None:
        """Format an event into an LLM-readable message.

        Return None to skip events this harness doesn't know how to format.
        Unknown events fall through to the LLM client's default formatting.
        """
        ...


class FormattedEvent(BaseEvent):
    """Wraps a harness-formatted string as a synthetic event.

    Used by HarnessMiddleware to pass formatted events through to the LLM
    client while preserving the original event for reference.
    """

    content: str
    role: str = Field(default="system")
    original: BaseEvent | None = Field(default=None)


# ---------------------------------------------------------------------------
# Middleware bridge
# ---------------------------------------------------------------------------


class HarnessMiddleware(BaseMiddleware):
    """Bridges ContextHarness into the middleware chain.

    Sits in the middleware chain like any other middleware. Intercepts
    on_llm_call to filter and format events before they reach the LLM client.
    This means the harness composes naturally with all other middleware —
    logging, token limiting, retry, signal injection — without special hooks.

    Middleware ordering in Actor._execute()::

        1. HarnessMiddleware(harness)         — outermost: filters events
        2. SignalInjectionMiddleware(queue)    — injects alerts
        3. User-provided middleware            — logging, token limiting, etc.
        4. LLM client call                     — innermost: sends to model
    """

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        harness: ContextHarness,
    ) -> None:
        super().__init__(event, context)
        self._harness = harness

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        # 1. Select: filter stream history to relevant events
        selected = self._harness.select(list(events))

        # 2. Format: transform events the harness knows about
        formatted: list[BaseEvent] = []
        for event in selected:
            custom_format = self._harness.format(event)
            if custom_format is not None:
                formatted.append(FormattedEvent(content=custom_format, original=event))
            else:
                # Pass through — LLM client's default formatting
                formatted.append(event)

        # 3. Delegate to next middleware / LLM client with filtered events
        return await call_next(formatted, context)


# ---------------------------------------------------------------------------
# Built-in harness implementations
# ---------------------------------------------------------------------------

# Event types that are always part of conversation context
_CONVERSATION_TYPES = (
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    ToolErrorEvent,
)


class ConversationHarness:
    """Standard chat harness. Only conversation and tool events reach the LLM.

    This is the default — preserves current Agent behavior exactly.
    """

    def select(self, events: list[BaseEvent]) -> list[BaseEvent]:
        return [e for e in events if isinstance(e, _CONVERSATION_TYPES)]

    def format(self, event: BaseEvent) -> str | None:
        return None  # Use LLM client's default formatting


class NetworkHarness:
    """Includes network events in the LLM context.

    The actor sees delegation results, signals, and scheduler events
    alongside normal conversation — enabling network-aware reasoning.
    """

    def select(self, events: list[BaseEvent]) -> list[BaseEvent]:
        # Import here to avoid circular imports — these are Layer 3 events
        # that may not exist when harness.py is first loaded
        from ..events import DelegationResult, SchedulerTriggerFired
        from .signal import Signal

        network_types = _CONVERSATION_TYPES + (Signal, DelegationResult, SchedulerTriggerFired)

        # Also include any FormattedEvent (from other harness layers)
        return [e for e in events if isinstance(e, (*network_types, FormattedEvent))]

    def format(self, event: BaseEvent) -> str | None:
        from ..events import DelegationResult, SchedulerTriggerFired
        from .signal import Signal

        if isinstance(event, Signal):
            level = event.severity.upper() if isinstance(event.severity, str) else str(event.severity)
            return f"[SIGNAL/{level}] ({event.source}): {event.message}"
        if isinstance(event, DelegationResult):
            return f"[DELEGATION RESULT] {event.source} → {event.target}: {event.result}"
        if isinstance(event, SchedulerTriggerFired):
            return f"[SCHEDULED] Trigger '{event.watch_id}' fired for {event.target}"
        if isinstance(event, FormattedEvent):
            return event.content
        return None

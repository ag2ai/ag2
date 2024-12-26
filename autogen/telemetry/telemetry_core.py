# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Global context variable to store the current telemetry manager
_current_telemetry: ContextVar[Optional["InstrumentationManager"]] = ContextVar("current_telemetry", default=None)

# MS DEBUGGING - FOR ENSURING CORRECT NUMBER OF SPANS - CAN BE REMOVED WHEN PR FINALISED
_spans_started: ContextVar[int] = ContextVar("spans_started", default=0)
_spans_ended: ContextVar[int] = ContextVar("spans_ended", default=0)


def get_spans_started() -> int:
    return _spans_started.get()


def get_spans_ended() -> int:
    return _spans_ended.get()


# MS TEMP


def get_current_telemetry() -> Optional["InstrumentationManager"]:
    """Get the current telemetry manager from context."""
    return _current_telemetry.get()


class SpanKind(Enum):
    """Enumeration of span kinds"""

    WORKFLOW = "workflow"
    CHATS = "chats"
    CHAT = "chat"
    NESTED_CHAT = "nested_chat"
    ROUND = "round"
    REPLY = "reply"
    REPLY_FUNCTION = "reply_function"
    SUMMARY = "summary"
    REASONING = "reasoning"
    SWARM_ON_CONDITION = "swarm_on_condition"


class EventKind(Enum):
    """Enumeration of event kinds"""

    AGENT_TRANSITION = "agent_transition"
    AGENT_CREATION = "agent_creation"
    GROUPCHAT_CREATION = "groupchat_creation"
    LLM_CREATE = "llm_create"
    AGENT_SEND_MSG = "agent_send_msg"
    TOOL_EXECUTION = "tool_execution"
    COST = "cost"  # All costs should create a COST event
    CONSOLE_PRINT = "console_print"


@dataclass
class SpanContext:
    """Data class for span context"""

    kind: SpanKind
    trace_id: str
    span_id: str
    timestamp: datetime = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = None
    core_span_id: Optional[str] = None

    def __post_init__(self):
        self.timestamp = datetime.now()

        if self.attributes is None:
            self.attributes = {}

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single attribute on the span"""
        self.attributes[key] = value


@dataclass
class EventContext:
    """Data class for event context"""

    kind: EventKind
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class TelemetryProvider(ABC):
    """Base class for telemetry providers"""

    @abstractmethod
    def start_trace(self, name: str, core_span_id: str, attributes: Dict[str, Any] = None) -> SpanContext:
        pass

    @abstractmethod
    def start_span(
        self,
        kind: SpanKind,
        core_span_id: str,
        parent_context: Optional[SpanContext] = None,
        attributes: Dict[str, Any] = None,
    ) -> SpanContext:
        pass

    @abstractmethod
    def set_span_attribute(self, context: SpanContext, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def end_span(self, context: SpanContext) -> None:
        pass

    @abstractmethod
    def record_event(
        self, span_context: SpanContext, event_name: str, kind: EventKind, attributes: Dict[str, Any] = None
    ) -> None:
        pass

    @abstractmethod
    def convert_attribute_value(self, value: Any) -> Any:
        pass


class InstrumentationManager:
    """Manager for telemetry providers"""

    def __init__(self):
        # Attached providers
        self._providers: List[TelemetryProvider] = []

        # Thread-local storage for the context stack ensures each chat sequence
        # (sync or async) maintains its own independent context.
        self._thread_local = threading.local()

        # Master set of spans
        self._active_spans: Dict[str, SpanContext] = {}

    def register_provider(self, provider: TelemetryProvider) -> None:
        self._providers.append(provider)

    def get_current_span(self) -> Optional[SpanContext]:
        if not hasattr(self._thread_local, "context_stack"):
            self._thread_local.context_stack = []
        return self._thread_local.context_stack[-1] if self._thread_local.context_stack else None

    def start_trace(self, name: str, attributes: Dict[str, Any] = None) -> SpanContext:
        """Start a new trace and create a master workflow span"""
        if attributes is None:
            attributes = {}

        # Create a single trace context
        trace_id = format(uuid.uuid4().int & ((1 << 128) - 1), "032x")
        core_span_id = format(uuid.uuid4().int & ((1 << 64) - 1), "016x")

        # Create the central span context
        context = SpanContext(
            kind=SpanKind.WORKFLOW,
            trace_id=trace_id,
            span_id=core_span_id,
            core_span_id=core_span_id,
            attributes=attributes,
        )

        # Store the span centrally
        self._active_spans[core_span_id] = context

        # Start trace with each provider
        for provider in self._providers:
            try:
                provider.start_trace(name=name, core_span_id=core_span_id, attributes=attributes)
            except Exception as e:
                print(f"Error in provider {provider.__class__.__name__}: {e}")

        # Store master context in thread local stack
        if not hasattr(self._thread_local, "context_stack"):
            self._thread_local.context_stack = []
        self._thread_local.context_stack.append(context)

        return context

    def start_span(
        self, kind: SpanKind, parent_context: Optional[SpanContext] = None, attributes: Dict[str, Any] = None
    ) -> SpanContext:
        """
        Start a new span with an optional explicit parent context.
        If parent_context is not provided, uses the current span from the stack.
        """

        # DEBUGGING - FOR ENSURING CORRECT NUMBER OF SPANS - CAN BE REMOVED WHEN PR FINALISED
        _spans_started.set(_spans_started.get() + 1)

        # Use provided parent_context or get from stack
        effective_parent = parent_context if parent_context is not None else self.get_current_span()

        # Generate unified IDs
        # span_id = format(uuid.uuid4().int & ((1 << 64) - 1), "016x")
        core_span_id = format(uuid.uuid4().int & ((1 << 64) - 1), "016x")

        # Create central span context
        context = SpanContext(
            kind=kind,
            trace_id=effective_parent.trace_id if effective_parent else None,
            span_id=core_span_id,
            parent_span_id=effective_parent.span_id if effective_parent else None,
            attributes=attributes,
            core_span_id=core_span_id,
        )

        # Store span centrally
        self._active_spans[core_span_id] = context

        # Start span in each provider
        for provider in self._providers:
            try:
                provider.start_span(
                    kind=kind, core_span_id=core_span_id, parent_context=effective_parent, attributes=attributes
                )
            except Exception as e:
                print(f"Error in provider {provider.__class__.__name__}: {e}")

        # Add to thread local stack
        if not hasattr(self._thread_local, "context_stack"):
            self._thread_local.context_stack = []
        self._thread_local.context_stack.append(context)

        return context

    def set_attribute(self, span_context: SpanContext, key: str, value: Any) -> None:
        """Set an attribute on the provided span"""
        if span_context:
            # Update master span
            span = self._active_spans.get(span_context.span_id)
            if span:
                span.set_attribute(key, value)

            # Propagate to each provider
            for provider in self._providers:
                try:
                    provider.set_span_attribute(span_context, key, value)
                except Exception as e:
                    print(f"Error in provider {provider.__class__.__name__}: {e}")

    def end_span(self) -> None:
        if not hasattr(self._thread_local, "context_stack"):
            return

        # DEBUGGING - FOR ENSURING CORRECT NUMBER OF SPANS - CAN BE REMOVED WHEN PR FINALISED
        _spans_ended.set(_spans_ended.get() + 1)

        if self._thread_local.context_stack:
            context = self._thread_local.context_stack.pop()

            # Remove from central management
            if context.core_span_id in self._active_spans:
                del self._active_spans[context.core_span_id]

            # End in each provider
            for provider in self._providers:
                try:
                    provider.end_span(context)
                except Exception as e:
                    print(f"Error in provider {provider.__class__.__name__}: {e}")

        """
        if not hasattr(self._thread_local, "context_stack"):
            return

        _spans_ended.set(_spans_ended.get() + 1)

        if self._thread_local.context_stack:
            context = self._thread_local.context_stack.pop()
            for provider in self._providers:
                provider.end_span(context)
        """

    def record_event(self, event_kind: EventKind, attributes: Dict[str, Any] = None) -> None:
        current_span = self.get_current_span()
        if current_span:
            for provider in self._providers:
                try:
                    provider.record_event(current_span, event_kind, attributes)
                except Exception as e:
                    print(f"Error in provider {provider.__class__.__name__}: {e}")


@contextmanager
def telemetry_context(manager: InstrumentationManager, trace_name: str = None, trace_attributes: Dict[str, Any] = None):
    """Context manager that automatically starts and ends a trace.

    Args:
        manager: The InstrumentationManager instance
        trace_name: Optional name for the trace. Defaults to 'AG2 Workflow'
        trace_attributes: Optional attributes for the trace
    """
    # Set the current telemetry manager in the context
    token = _current_telemetry.set(manager)
    try:
        # Start the trace automatically
        trace_name = trace_name or "AG2 Workflow"
        trace_attributes = trace_attributes or {}
        manager.start_trace(trace_name, trace_attributes)
        yield manager
    finally:
        # End all remaining spans in the stack
        while hasattr(manager._thread_local, "context_stack") and manager._thread_local.context_stack:
            manager.end_span()
        # Reset the context
        _current_telemetry.reset(token)


# STATIC METHODS


@staticmethod
def _is_list_of_string_dicts(item: Any) -> bool:
    """Check for a list of dictionaries with string values, for messages"""
    if not isinstance(item, list):
        return False
    if not all(isinstance(d, dict) for d in item):
        return False
    return all(isinstance(key, str) and isinstance(value, str) for d in item for key, value in d.items())

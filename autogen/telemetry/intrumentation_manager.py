# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from .base_telemetry import (
    SPAN_ID_BITS,
    TRACE_ID_BITS,
    EventKind,
    SpanContext,
    SpanKind,
    TelemetryProvider,
    generate_id,
)


class InstrumentationManager:
    """Manager for telemetry providers.

    The InstrumentationManager is the central hub for all telemetry providers. It manages the providers
    and orchestrates the starting, ending, and recording of traces, spans, and events.

    The InstrumentationManager is to be used with the telemetry_context context manager.

    Example:
    ```python
    from autogen.telemetry.telemetry_core import InstrumentationManager, telemetry_context
    instrumentation_manager = InstrumentationManager()
    ... register providers with manager ...

    with telemetry_context(instrumentation_manager, "My Program Name", {"my_program_id": "456"}) as telemetry:
        ... AG2 workflow code here ...

    ```
    """

    def __init__(self):
        # Attached providers
        self._providers: List[TelemetryProvider] = []

        # Thread-local storage for the context stack
        self._thread_local = threading.local()

        # Master set of spans
        self._active_spans: Dict[str, SpanContext] = {}

    def register_provider(self, provider: TelemetryProvider) -> None:
        """Register a telemetry provider with the manager.

        Args:
            provider: The TelemetryProvider instance to register
        """
        self._providers.append(provider)

    def deregister_provider(self, provider: TelemetryProvider) -> bool:
        """Remove a telemetry provider from the manager.

        Args:
            provider: The TelemetryProvider instance to remove

        Returns:
            bool: True if provider was found and removed, False if not found
        """
        try:
            self._providers.remove(provider)
            return True
        except ValueError:
            return False

    def get_current_span(self) -> Optional[SpanContext]:
        """Get the current span from the thread-local stack.

        Returns:
            Optional[SpanContext]: The current span or None if not found
        """
        if not hasattr(self._thread_local, "context_stack"):
            self._thread_local.context_stack = []
        return self._thread_local.context_stack[-1] if self._thread_local.context_stack else None

    def start_trace(self, name: str, attributes: Dict[str, Any] = None) -> SpanContext:
        """Start a new trace and workflow span across all providers.

        Args:
            name: User-provided name of the trace
            attributes: Optional attributes for the trace

        Returns:
            SpanContext: The trace-level span
        """
        if attributes is None:
            attributes = {}

        # Create a single trace context
        trace_id = generate_id(TRACE_ID_BITS)
        core_span_id = generate_id(SPAN_ID_BITS)

        # Create the central span context
        context = SpanContext(
            kind=SpanKind.WORKFLOW,
            trace_id=trace_id,
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
        """Start a new span,with an optional explicit parent context, across all providers.

        If parent_context is not provided, uses the current span from the stack.

        Args:
            kind (SpanKind): The kind of span
            parent_context (SpanContext): Optional parent span context
            attributes (dict): Optional attributes for the span
        """

        # DEBUGGING - FOR ENSURING CORRECT NUMBER OF SPANS - CAN BE REMOVED WHEN PR FINALISED
        _spans_started.set(_spans_started.get() + 1)

        # Use provided parent_context or get from stack
        effective_parent = parent_context if parent_context is not None else self.get_current_span()

        # Generate unified IDs
        core_span_id = generate_id(SPAN_ID_BITS)

        # Create central span context
        context = SpanContext(
            kind=kind,
            trace_id=effective_parent.trace_id if effective_parent else None,
            parent_span_id=effective_parent.core_span_id if effective_parent else None,
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
        """Set an attribute on a span across all providers.

        Args:
            span_context: The span context to set the attribute on
            key: The attribute key
            value: The attribute value
        """
        if span_context:
            # Update master span
            span = self._active_spans.get(span_context.core_span_id)
            if span:
                span.set_attribute(key, value)

            # Propagate to each provider
            for provider in self._providers:
                try:
                    provider.set_span_attribute(span_context, key, value)
                except Exception as e:
                    print(f"Error in provider {provider.__class__.__name__}: {e}")

    def end_span(self) -> None:
        """End the current span across all providers."""
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

    def record_event(self, event_kind: EventKind, attributes: Dict[str, Any] = None) -> None:
        """Record an event in the current span across all providers.

        Args:
            event_kind (EventKind): The kind of event
            attributes (dict): Optional attributes for the event
        """
        current_span = self.get_current_span()
        if current_span:
            for provider in self._providers:
                try:
                    provider.record_event(current_span, event_kind, attributes)
                except Exception as e:
                    print(f"Error in provider {provider.__class__.__name__}: {e}")


# Global context variable to store the current telemetry manager
_current_telemetry: ContextVar[Optional[InstrumentationManager]] = ContextVar("current_telemetry", default=None)


def get_current_telemetry() -> Optional[InstrumentationManager]:
    """Get the current telemetry manager from the global context.

    Used throughout the code to access the current telemetry manager (returning None if not established)

    Returns:
        Optional[InstrumentationManager]: The current telemetry manager or None if not

    E.g.
        telemetry = get_current_telemetry()
        if telemetry:
            reply_span_context = telemetry.start_span(
                kind=SpanKind.REPLY,
                ...
    """
    return _current_telemetry.get()


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


# DEBUG - FOR ENSURING CORRECT NUMBER OF SPANS STARTED AND ENDED - CAN BE REMOVED WHEN PR REVIEWED
_spans_started: ContextVar[int] = ContextVar("spans_started", default=0)
_spans_ended: ContextVar[int] = ContextVar("spans_ended", default=0)


def get_spans_started() -> int:
    return _spans_started.get()


def get_spans_ended() -> int:
    return _spans_ended.get()


# DEBUG

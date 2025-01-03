# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

# Number of bits for trace and span IDs
TRACE_ID_BITS = 128
SPAN_ID_BITS = 64


class SpanKind(Enum):
    """Enumeration of span kinds

    Spans represent a unit of work within a trace. Typically spans have child spans and events.

    Descriptions:
        WORKFLOW: Main workflow span
        CHATS: Multiple chats, e.g. initiate_chats
        CHAT: Single chat, e.g. initiate_chat
        NESTED_CHAT: A nested chat, e.g. _summary_from_nested_chats
        GROUP_CHAT: A groupchat, e.g. run_chat
        ROUND: A round within a chat (where chats have max_round or max_turn)
        REPLY: Agent replying to another agent
        REPLY_FUNCTION: Functions executed during a reply, such as generate_oai_reply and check_termination_and_human_reply
        SUMMARY: Summarization of a chat
        REASONING: Agent reasoning step, for advanced agents like ReasoningAgent and CaptainAgent
        GROUPCHAT_SELECT_SPEAKER: GroupChat speaker selection (covers all selection methods)
        SWARM_ON_CONDITION: Swarm-specific, ON_CONDITION hand off
    """

    WORKFLOW = "workflow"
    CHATS = "chats"
    CHAT = "chat"
    NESTED_CHAT = "nested_chat"
    GROUP_CHAT = "group_chat"
    ROUND = "round"
    REPLY = "reply"
    REPLY_FUNCTION = "reply_function"
    SUMMARY = "summary"
    REASONING = "reasoning"
    GROUPCHAT_SELECT_SPEAKER = "groupchat_select_speaker"
    SWARM_ON_CONDITION = "swarm_on_condition"


class EventKind(Enum):
    """Enumeration of span event kinds

    Events represent a singular point in time within a span, capturing specific moments or actions.

    Descriptions:
        AGENT_TRANSITION: Transition moved from one agent to another
        AGENT_CREATION: Creation of an Agent
        GROUPCHAT_CREATION: Creation of a GroupChat
        LLM_CREATE: LLM execution
        AGENT_SEND_MSG: Agent sending a message to another agent
        TOOL_EXECUTION: Tool or Function execution
        COST: Cost event
        SWARM_TRANSITION: Swarm-specific, transition reason (e.g. ON_CONDITION)
        CONSOLE_PRINT: Console output (TBD)
    """

    AGENT_TRANSITION = "agent_transition"
    AGENT_CREATION = "agent_creation"
    GROUPCHAT_CREATION = "groupchat_creation"
    LLM_CREATE = "llm_create"
    AGENT_SEND_MSG = "agent_send_msg"
    TOOL_EXECUTION = "tool_execution"
    COST = "cost"
    SWARM_TRANSITION = "swarm_transition"
    CONSOLE_PRINT = "console_print"  # TBD


@dataclass
class SpanContext:
    """Data class to represent a span (a unit of work within a trace)."""

    kind: SpanKind
    trace_id: str
    timestamp: datetime = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = None
    core_span_id: Optional[str] = None

    def __post_init__(self):
        # Timestamps for ordering spans
        self.timestamp = datetime.now()

        if self.attributes is None:
            self.attributes = {}

    def set_attribute(self, key: str, value: Any) -> None:
        """Set/update an attribute."""
        self.attributes[key] = value

    def has_attribute(self, key: str) -> bool:
        """Check if an attribute exists."""
        return key in self.attributes


@dataclass
class EventContext:
    """Data class to represent a span event (point in time events occurring within a span)."""

    kind: EventKind
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class TelemetryProvider(ABC):
    """Base class for telemetry providers.

    All telemetry providers are to implement this interface and then be registered
    with the InstrumentationManager.

    Telemetry follows the OpenTelemetry signals terminology of traces, spans, and events. However, it is
    not restricted to OpenTelemetry use and can be used to gather AG2 activity or represent it in other formats,
    such as diagrams or logging to files/databases.

    Trace: Full workflow, including object creations, chats, summaries, etc.
    Span: A unit of work within a trace, such as a chat, round, or agent reply. Spans typically have child spans and events.
    Event: A singular point in time within a span, such as LLM execution, agent creation, agent transition, or cost.

    Multiple providers can be attached to the InstrumentationManager for simultaneous and real-time telemetry.
    """

    @abstractmethod
    def start_trace(self, name: str, core_span_id: str, attributes: Dict[str, Any] = None) -> SpanContext:
        """Start a new trace.

        Args:
            name: User-provided name of the trace
            core_span_id: Unique ID for the trace-level span that's created with a trace
            attributes: Optional attributes for the trace

        Returns:
            SpanContext: The trace-level span
        """
        pass

    @abstractmethod
    def start_span(
        self,
        kind: SpanKind,
        core_span_id: str,
        parent_context: Optional[SpanContext] = None,
        attributes: Dict[str, Any] = None,
    ) -> SpanContext:
        """Start a new span.

        Args:
            kind (SpanKind): The kind of span
            core_span_id (str): Unique ID for the span, determined by the InstrumentationManager and common to the span across all providers
            parent_context (SpanContext or None): Optional parent span context to help maintain hierarchy
                Note: a provider, such as OpenTelemetry, may not require this to maintain hierarchy
            attributes (dict or None): Optional attributes for the span

        Returns:
            SpanContext: The created span context
        """
        pass

    @abstractmethod
    def set_span_attribute(self, context: SpanContext, key: str, value: Any) -> None:
        """Set an attribute on a span.

        Args:
            context (SpanContext): The span context
            key (str): The attribute key
            value (Any): The attribute value
        """
        pass

    @abstractmethod
    def end_span(self, context: SpanContext) -> None:
        """Ends a span.

        Note: Some telemetry providers, like OpenTelemetry, must have an end_span for every start_span to create a valid trace.

        Args:
            context (SpanContext): The span context to end"""
        pass

    @abstractmethod
    def record_event(
        self, span_context: SpanContext, event_name: str, kind: EventKind, attributes: Dict[str, Any] = None
    ) -> None:
        """Record a span event.

        Args:
            span_context (SpanContext): The span context to record the event in
            event_name (str): The name of the event
            kind (EventKind): The kind of event
            attributes (dict or None): Optional attributes for the event
        """
        pass

    @abstractmethod
    def convert_attribute_value(self, value: Any) -> Any:
        """Convert an attribute value to a format suitable for the provider.

        Args:
            value (Any): The value to convert

        Returns:
            Any: The converted value
        """
        pass


# STATIC METHODS


@staticmethod
def generate_id(bits: int) -> str:
    """Generate a random ID of specified bit length.

    Creates a random identifier by generating a UUID4 and masking it to the desired bit length.
    The result is formatted as a lowercase hex string with the appropriate number of leading zeros.

    Based on the requirement for OpenTelemetry trace and span IDs.

    Args:
        bits: Number of bits for the ID (e.g. 128 for trace ID, 64 for span ID)

    Returns:
        str: Hex string representation of the ID with appropriate length (bits/4 characters)

    Examples:
        >>> InstrumentationManager.generate_id(128)  # For trace ID
        'a1b2c3d4e5f67890a1b2c3d4e5f67890'
        >>> InstrumentationManager.generate_id(64)   # For span ID
        'a1b2c3d4e5f67890'
    """
    hex_chars = bits // 4  # Each hex char represents 4 bits
    return format(uuid.uuid4().int & ((1 << bits) - 1), f"0{hex_chars}x")


@staticmethod
def _is_list_of_string_dicts(item: Any) -> bool:
    """Check if an object is a list of dictionaries with string values, for handling messages

    Args:
        item: The object to check

    Returns:
        bool: True if the object is a list of dictionaries with string values, False otherwise
    """
    if not isinstance(item, list):
        return False
    if not all(isinstance(d, dict) for d in item):
        return False
    return all(isinstance(key, str) and isinstance(value, str) for d in item for key, value in d.items())

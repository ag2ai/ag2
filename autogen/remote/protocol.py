# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ProtocolEvents(StrEnum):
    START_CHAT = "START_CHAT"
    STOP_CHAT = "STOP_CHAT"

    SEND_MESSAGE = "SEND_MESSAGE"
    """Send outer agent a message to update conversation state."""

    NEXT_SPEAKER = "NEXT_SPEAKER"
    """Command remote agent to speak."""

    # UPDATE_CONTEXT = "update_context"
    # ASK_HUMAN_INPUT = "ask_human_input"
    # PING = "ping"
    # MARK_DEAD = "mark_dead"
    # MARK_ALIVE = "mark_alive"


class StopEvent(BaseModel):
    event_type: Literal[ProtocolEvents.STOP_CHAT] = ProtocolEvents.STOP_CHAT


class NextSpeakerEvent(BaseModel):
    event_type: Literal[ProtocolEvents.NEXT_SPEAKER] = ProtocolEvents.NEXT_SPEAKER


class SendEvent(BaseModel):
    event_type: Literal[ProtocolEvents.SEND_MESSAGE] = ProtocolEvents.SEND_MESSAGE
    content: dict[str, Any] | str | None = None


AgentBusEvent = StopEvent | SendEvent | NextSpeakerEvent

_AGENT_BUS_EVENT_VALIDATOR = TypeAdapter(Annotated[AgentBusEvent, Field(discriminator="event_type")])


def serialize_event(content: dict[str, Any] | str | None) -> AgentBusEvent:
    if not content:
        return StopEvent()

    if isinstance(content, dict) and "event_type" in content:
        return _AGENT_BUS_EVENT_VALIDATOR.validate_python(content)

    return SendEvent(content=content)

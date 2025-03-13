# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, Callable, ClassVar, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

__all__ = [
    "AgentMessageEvent",
    "AsyncInputRequestEvent",
    "ErrorEvent",
    "Event",
    "InputRequestEvent",
    "InputResponseEvent",
    "OutputEvent",
    "SystemEvent",
]

EventType = Literal[
    "input_request",
    "async_input_request",
    "input_response",
    "agent_message",
    "output",
    "system",
    "error",
    "terminate",
    "unknown",
]
Message = dict[str, Any]


class Event(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)

    type: ClassVar[EventType]

    @classmethod
    def accept(cls, message: dict[str, Any]) -> bool:
        return message["type"] == cls.type  # type: ignore[no-any-return]

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the Event instance, including the `type` field."""
        data = super().model_dump(**kwargs)
        data["type"] = self.type  # Include the ClassVar type explicitly
        return data


_registered_events: list[type[Event]] = []


class UnkownEventType(Event):
    raw: Message

    type: ClassVar[EventType] = "unknown"


def register_event() -> Callable[[type[Event]], type[Event]]:
    def decorator(event_cls: type[Event]) -> type[Event]:
        global _registered_events
        _registered_events.append(event_cls)
        return event_cls

    return decorator


def get_event(message: dict[str, Any]) -> Event:
    global _registered_events
    for event_cls in _registered_events:
        if event_cls.accept(message):
            return event_cls.model_validate(message)
    # raise ValueError(f"Unknown event type for: {message=}")
    return UnkownEventType(raw=message)


@register_event()
class InputRequestEvent(Event):
    prompt: str
    respond: Optional[Callable[[str], None]] = None

    type: ClassVar[EventType] = "input_request"


@register_event()
class AsyncInputRequestEvent(Event):
    prompt: str

    async def a_respond(self, response: "InputResponseEvent") -> None:
        pass

    type: ClassVar[EventType] = "async_input_request"


@register_event()
class InputResponseEvent(Event):
    type: ClassVar[EventType] = "input_response"

    value: str


@register_event()
class AgentMessageEvent(Event):
    message: Message

    type: ClassVar[EventType] = "agent_message"


@register_event()
class OutputEvent(Event):
    value: str

    type: ClassVar[EventType] = "output"


@register_event()
class SystemEvent(Event):
    value: str

    type: ClassVar[EventType] = "system"


@register_event()
class ErrorEvent(Event):
    type: ClassVar[EventType] = "error"

    error: str


@register_event()
class TerminationEvent(Event):
    type: ClassVar[EventType] = "terminate"
    summary: Optional[str] = None

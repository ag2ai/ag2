# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from .base import BaseEvent, Field
from .tool_events import ToolCallsEvent


class ModelRequest(BaseEvent):
    """Event representing an input request sent to the model."""

    content: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelRequest):
            return NotImplemented
        return self.content == other.content

    def to_api(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "role": "user",
        }


class ModelEvent(BaseEvent):
    """Base class for all model-related events."""


class ModelReasoning(ModelEvent):
    """Intermediate reasoning content emitted by the model."""

    content: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelReasoning):
            return NotImplemented
        return self.content == other.content


class ModelMessage(ModelEvent):
    """Single message emitted by the model."""

    content: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelMessage):
            return NotImplemented
        return self.content == other.content


class ModelResponse(ModelEvent):
    """Final model response produced for a given request."""

    message: ModelMessage | None = None
    tool_calls: ToolCallsEvent = Field(default_factory=ToolCallsEvent)
    usage: dict[str, float] = Field(default_factory=dict)
    response_force: bool = False
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None

    @property
    def content(self) -> str | None:
        return self.message.content if self.message else None

    def __repr__(self) -> str:
        text = f"content={getattr(self.message, 'content', None)}"
        if self.tool_calls:
            text += f", tool_calls={self.tool_calls}"
        if self.usage:
            text += f", usage={self.usage}"
        return f"ModelResponse({text})"

    def to_api(self) -> dict[str, Any]:
        msg = {
            "content": self.message.content if self.message else None,
            "role": "assistant",
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls.to_api()
        return msg

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelResponse):
            return NotImplemented
        return (
            self.message == other.message
            and self.tool_calls == other.tool_calls
            and self.usage == other.usage
            and self.response_force == other.response_force
        )


class ModelMessageChunk(ModelEvent):
    """Chunk of a streamed model message."""

    content: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelMessageChunk):
            return NotImplemented
        return self.content == other.content


class HumanInputRequest(BaseEvent):
    """Event requesting input from a human user."""

    content: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HumanInputRequest):
            return NotImplemented
        return self.content == other.content


class HumanMessage(BaseEvent):
    """Event representing a human user's response."""

    content: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HumanMessage):
            return NotImplemented
        return self.content == other.content

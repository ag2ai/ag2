# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import uuid4

from .base import BaseEvent, Field


class ToolCalls(BaseEvent):
    """Container event holding a collection of tool calls."""

    calls: list["ToolCall"] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.calls)

    def to_api(self) -> list[dict[str, Any]]:
        return [c.to_api() for c in self.calls]


class ToolResults(BaseEvent):
    """Container event holding results (or errors) produced by tools."""

    results: list["ToolResult | ToolError"]


class ToolEvent(BaseEvent):
    """Base class for all tool-related events."""


class ToolCall(ToolEvent):
    """Represents a single tool invocation requested by the model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "arguments": self.arguments,
                "name": self.name,
            },
        }


class ClientToolCall(ToolCall):
    @classmethod
    def from_call(cls, call: ToolCall) -> "ClientToolCall":
        return cls(
            parent_id=call.id,
            name=call.name,
            arguments=call.arguments,
        )


class ToolResult(ToolEvent):
    """Represents a successful tool execution result."""

    parent_id: str
    name: str
    content: str

    def to_api(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.parent_id,
            "content": self.content,
        }


class ToolError(ToolResult):
    """Represents a failed tool execution with an associated error."""

    parent_id: str
    name: str
    content: str
    error: Exception


class ToolNotFoundEvent(ToolError):  # noqa: N818
    """ToolError raised when the requested tool cannot be found."""


class ModelRequest(BaseEvent):
    """Event representing an input request sent to the model."""

    content: str

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


class ModelMessage(ModelEvent):
    """Single message emitted by the model."""

    content: str


class ModelResponse(ModelEvent):
    """Final model response produced for a given request."""

    message: ModelMessage | None = None
    tool_calls: ToolCalls = Field(default_factory=ToolCalls)
    usage: dict[str, float] = Field(default_factory=dict)
    response_force: bool = False

    def to_api(self) -> dict[str, Any]:
        msg = {
            "content": self.message.content if self.message else None,
            "role": "assistant",
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls.to_api()
        return msg


class ModelMessageChunk(ModelEvent):
    """Chunk of a streamed model message."""

    content: str


class HumanInputRequest(BaseEvent):
    """Event requesting input from a human user."""

    content: str


class HumanMessage(BaseEvent):
    """Event representing a human user's response."""

    content: str

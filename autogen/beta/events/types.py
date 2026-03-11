# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from traceback import format_exc
from typing import Any
from uuid import uuid4

from fast_depends.pydantic import PydanticSerializer

from .base import BaseEvent, Field


class ToolCalls(BaseEvent):
    """Container event holding a collection of tool calls."""

    calls: list["ToolCall"] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.calls)

    def to_api(self) -> list[dict[str, Any]]:
        return [c.to_api() for c in self.calls]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCalls):
            return NotImplemented
        return self.calls == other.calls


class ToolResults(BaseEvent):
    """Container event holding results (or errors) produced by tools."""

    results: list["ToolResult | ToolError"]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolResults):
            return NotImplemented
        return self.results == other.results


class ToolEvent(BaseEvent):
    """Base class for all tool-related events."""


class ToolCall(ToolEvent):
    """Represents a single tool invocation requested by the model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str = "{}"

    _serialized_arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def serialized_arguments(self) -> dict[str, Any]:
        if not self._serialized_arguments:
            self._serialized_arguments = json.loads(self.arguments)
        return self._serialized_arguments

    @serialized_arguments.setter
    def serialized_arguments(self, value: dict[str, Any]) -> None:
        self._serialized_arguments = value

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id}, name={self.name}, arguments={self.arguments})"

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "arguments": json.dumps(self.serialized_arguments),
                "name": self.name,
            },
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.id == other.id and self.name == other.name and self.arguments == other.arguments


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

    raw_content: Any = None
    _content: str = Field(default_factory=str)

    @property
    def content(self) -> str:
        if not self._content:
            self._content = PydanticSerializer.encode(self.raw_content).decode()
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    def __repr__(self) -> str:
        return f"ToolResult(parent_id={self.parent_id}, name={self.name}, content={self.content})"

    def to_api(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.parent_id,
            "content": self.content,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolResult):
            return NotImplemented
        return self.parent_id == other.parent_id and self.name == other.name and self.content == other.content


class ToolError(ToolResult):
    """Represents a failed tool execution with an associated error."""

    error: Exception

    @property
    def content(self) -> str:
        if not self._content:
            self._content = format_exc(limit=3)
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolError):
            return NotImplemented
        # Compare error types and messages to avoid relying on identity.
        same_error = type(self.error) is type(other.error) and str(self.error) == str(other.error)
        return (
            self.parent_id == other.parent_id
            and self.name == other.name
            and self.content == other.content
            and same_error
        )


class ToolNotFoundEvent(ToolError):  # noqa: N818
    """ToolError raised when the requested tool cannot be found."""


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
    tool_calls: ToolCalls = Field(default_factory=ToolCalls)
    usage: dict[str, float] = Field(default_factory=dict)
    response_force: bool = False

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

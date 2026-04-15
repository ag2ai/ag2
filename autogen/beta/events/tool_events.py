# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Generic
from uuid import uuid4

from fast_depends.pydantic import PydanticSerializer
from typing_extensions import TypeVar as TypeVar313

from .base import BaseEvent, Field

ResultT = TypeVar313("ResultT", default=Any)


@dataclass(slots=True)
class ToolResult(Generic[ResultT]):
    content: ResultT = None
    final: bool = field(default=False, kw_only=True)

    @classmethod
    def ensure_result(cls, data: Any) -> "ToolResult":
        if isinstance(data, ToolResult):
            return data
        return cls(content=data)


class ToolCallsEvent(BaseEvent):
    """Container event holding a collection of tool calls."""

    calls: list["ToolCallEvent"] = Field(default_factory=list, kw_only=False)

    def __len__(self) -> int:
        return len(self.calls)

    def to_api(self) -> list[dict[str, Any]]:
        return [c.to_api() for c in self.calls]


class ToolResultsEvent(BaseEvent):
    """Container event holding results (or errors) produced by tools."""

    results: list["ToolResultEvent[Any] | ToolErrorEvent"] = Field(kw_only=False)


class ToolEvent(BaseEvent):
    """Base class for all tool-related events."""


class ToolCallEvent(ToolEvent):
    """Represents a single tool invocation requested by the model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str = "{}"
    provider_data: dict[str, Any] = Field(default_factory=dict, compare=False)

    _serialized_arguments: dict[str, Any] | None = Field(default=None, init=False, compare=False)

    @property
    def serialized_arguments(self) -> dict[str, Any]:
        if self._serialized_arguments is None:
            self._serialized_arguments = json.loads(self.arguments or "{}")
        return self._serialized_arguments

    @serialized_arguments.setter
    def serialized_arguments(self, value: dict[str, Any]) -> None:
        self._serialized_arguments = value

    def __repr__(self) -> str:
        text = f"id={self.id}, name='{self.name}'"
        if c := self.arguments:
            text += f", arguments='{c}'"
        return f"{self.__class__.__name__}({text})"

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "arguments": json.dumps(self.serialized_arguments),
                "name": self.name,
            },
        }


class ClientToolCallEvent(ToolCallEvent):
    @classmethod
    def from_call(cls, call: ToolCallEvent) -> "ClientToolCallEvent":
        return cls(
            parent_id=call.id,
            name=call.name,
            arguments=call.arguments,
        )


class BuiltinToolCallEvent(ToolCallEvent):
    """Represents a builtin tool invocation requested by the model."""


class ToolResultEvent(ToolEvent, Generic[ResultT]):
    """Represents a successful tool execution result."""

    parent_id: str
    name: str | None = None

    result: "ToolResult[ResultT]"
    _content: str = Field(default_factory=str, init=False)

    @property
    def content(self) -> str:
        if not self._content:
            self._content = PydanticSerializer.encode(self.result.content).decode()
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parent_id={self.parent_id}, name='{self.name}', content={self.content})"

    @classmethod
    def from_call(cls, call: ToolCallEvent, result: ResultT) -> "ToolResultEvent[ResultT]":
        return cls(
            parent_id=call.id,
            name=call.name,
            result=ToolResult.ensure_result(result),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.parent_id,
            "content": self.content,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolResultEvent):
            return NotImplemented
        return self.parent_id == other.parent_id and self.name == other.name and self.content == other.content


class BuiltinToolResultEvent(ToolResultEvent):
    """Represents a successful builtin tool execution result."""


class ToolErrorEvent(ToolResultEvent[None]):
    """Represents a failed tool execution with an associated error."""

    error: Exception

    @classmethod
    def from_call(cls, call: ToolCallEvent, error: Exception) -> "ToolErrorEvent":
        return cls(
            parent_id=call.id,
            name=call.name,
            error=error,
            result=ToolResult(),
        )

    @property
    def content(self) -> str:
        if not self._content:
            self._content = "".join(
                traceback.format_exception(
                    type(self.error),
                    self.error,
                    self.error.__traceback__,
                )
            )
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolErrorEvent):
            return NotImplemented
        # Compare error types and messages to avoid relying on identity.
        same_error = type(self.error) is type(other.error) and str(self.error) == str(other.error)
        return (
            self.parent_id == other.parent_id
            and self.name == other.name
            and self.content == other.content
            and same_error
        )


class ToolNotFoundEvent(ToolErrorEvent):  # noqa: N818
    """ToolErrorEvent raised when the requested tool cannot be found."""

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import traceback
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from autogen.beta.types import SendableMessage

from .base import BaseEvent, Field
from .input_events import Input


@dataclass(slots=True)
class ToolResult:
    parts: list[Input]
    final: bool = field(default=False, kw_only=True, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *parts: SendableMessage | Input,
        final: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.parts = [Input.ensure_input(p) for p in parts]
        self.final = final
        self.metadata = metadata or {}

    @classmethod
    def ensure_result(cls, data: "ToolResult | SendableMessage | Input") -> "ToolResult":
        if isinstance(data, ToolResult):
            return data
        return cls(data)


class ToolCallsEvent(BaseEvent):
    """Container event holding a collection of tool calls."""

    calls: list["ToolCallEvent"] = Field(default_factory=list, kw_only=False)

    def __len__(self) -> int:
        return len(self.calls)

    def to_api(self) -> list[dict[str, Any]]:
        return [c.to_api() for c in self.calls]


class ToolResultsEvent(BaseEvent):
    """Container event holding results (or errors) produced by tools."""

    results: list["ToolResultEvent"] = Field(kw_only=False)


class ToolEvent(BaseEvent):
    """Base class for all tool-related events."""


class ToolCallEvent(ToolEvent):
    """Represents a single tool invocation requested by the model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(kw_only=False)
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


class ClientToolCallEvent(ToolEvent):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(kw_only=False)
    arguments: str = "{}"

    @classmethod
    def from_call(cls, call: ToolCallEvent) -> "ClientToolCallEvent":
        return cls(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
        )


class BuiltinToolCallEvent(ToolCallEvent):
    """Represents a builtin tool invocation requested by the model."""


class ToolResultEvent(ToolEvent):
    """Represents a successful tool execution result."""

    parent_id: str
    name: str | None = None

    result: "ToolResult"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parent_id={self.parent_id}, name='{self.name}', result={self.result})"

    @classmethod
    def from_call(cls, call: ToolCallEvent, result: Any) -> "ToolResultEvent":
        return cls(
            parent_id=call.id,
            name=call.name,
            result=ToolResult.ensure_result(result),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolResultEvent):
            return NotImplemented
        return self.parent_id == other.parent_id and self.name == other.name and self.result == other.result


class BuiltinToolResultEvent(ToolResultEvent):
    """Represents a successful builtin tool execution result."""


class ToolErrorEvent(ToolResultEvent):
    """Represents a failed tool execution with an associated error."""

    error: Exception

    @classmethod
    def from_call(cls, call: ToolCallEvent, error: Exception) -> "ToolErrorEvent":
        return cls(
            parent_id=call.id,
            name=call.name,
            error=error,
            result=ToolResult(
                "".join(
                    traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                )
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolErrorEvent):
            return NotImplemented
        # Compare error types and messages to avoid relying on identity.
        same_error = type(self.error) is type(other.error) and str(self.error) == str(other.error)
        return self.parent_id == other.parent_id and self.name == other.name and same_error


class ToolNotFoundEvent(ToolErrorEvent):  # noqa: N818
    """ToolErrorEvent raised when the requested tool cannot be found."""

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any
from uuid import uuid4

from .base import BaseEvent, Field
from .tool_events import ToolCallsEvent


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage normalized across beta LLM providers."""

    prompt_tokens: float | None = None
    completion_tokens: float | None = None
    total_tokens: float | None = None
    cache_read_input_tokens: float | None = None
    cache_creation_input_tokens: float | None = None

    def __bool__(self) -> bool:
        return any((
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        ))


class ModelEvent(BaseEvent):
    """Base class for all model-related events."""


class ModelReasoning(ModelEvent):
    """Intermediate reasoning content emitted by the model."""

    content: str = Field(kw_only=False)


class ModelMessage(ModelEvent):
    """Single message emitted by the model."""

    content: str = Field(kw_only=False)

    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BinaryResult:
    """Binary result emitted by the model."""

    data: bytes
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


class ModelResponse(ModelEvent):
    """Final model response produced for a given request."""

    message: ModelMessage | None = Field(default=None, kw_only=False)
    tool_calls: ToolCallsEvent = Field(default_factory=ToolCallsEvent)
    usage: Usage = Field(default_factory=Usage)
    response_force: bool = False

    files: list[BinaryResult] = Field(default_factory=list)

    # Tracing information
    model: str | None = Field(default=None, compare=False)
    provider: str | None = Field(default=None, compare=False)
    finish_reason: str | None = Field(default=None, compare=False)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.message.metadata if self.message else {}

    @property
    def content(self) -> str | None:
        return self.message.content if self.message else None

    def __repr__(self) -> str:
        if self.message:
            text = f"content={self.message.content}"
            if self.message.metadata:
                text += f", metadata={self.message.metadata}"
        else:
            text = "content=None"
        if self.tool_calls:
            text += f", tool_calls={self.tool_calls}"
        if self.usage:
            text += f", usage={self.usage}"
        if self.files:
            text += f", files={len(self.files)}"
        return f"ModelResponse({text})"

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

    content: str = Field(kw_only=False)


class HumanInputRequest(BaseEvent):
    """Event requesting input from a human user."""

    id: str = Field(default_factory=lambda: str(uuid4()), compare=False)
    content: str = Field(kw_only=False)


class HumanMessage(BaseEvent):
    """Event representing a human user's response."""

    parent_id: str = Field(default="", compare=False)
    content: str = Field(kw_only=False)

    @classmethod
    def ensure_message(cls, content: "str | HumanMessage", parent_id: str) -> "HumanMessage":
        msg = content if isinstance(content, HumanMessage) else cls(content)
        if not msg.parent_id:
            # Set parent_id after creation to hide this option from public API
            msg.parent_id = parent_id
        return msg

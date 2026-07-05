# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any
from uuid import uuid4

from .base import BaseEvent, Field
from .tool_events import ToolCallsEvent


def _add_tokens(a: float | None, b: float | None) -> float | None:
    """Add two optional token counts. ``None + None`` stays ``None`` so a
    field absent on both sides isn't fabricated as ``0``; if either side has a
    value, the missing side counts as ``0``."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage normalized across beta LLM providers."""

    prompt_tokens: float | None = None
    completion_tokens: float | None = None
    total_tokens: float | None = None
    cache_read_input_tokens: float | None = None
    cache_creation_input_tokens: float | None = None
    thinking_tokens: float | None = None

    def __bool__(self) -> bool:
        return any((
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
            self.thinking_tokens,
        ))

    def __add__(self, other: "Usage") -> "Usage":
        """Field-wise sum so ``sum(usages, Usage())`` aggregates an iterable."""
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=_add_tokens(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=_add_tokens(self.completion_tokens, other.completion_tokens),
            total_tokens=_add_tokens(self.total_tokens, other.total_tokens),
            cache_read_input_tokens=_add_tokens(self.cache_read_input_tokens, other.cache_read_input_tokens),
            cache_creation_input_tokens=_add_tokens(
                self.cache_creation_input_tokens, other.cache_creation_input_tokens
            ),
            thinking_tokens=_add_tokens(self.thinking_tokens, other.thinking_tokens),
        )


class ModelEvent(BaseEvent):
    """Base class for all model-related events."""


class ModelReasoning(ModelEvent):
    """Intermediate reasoning content emitted by the model.

    Transient: intermediate thinking content, not part of the final response.
    """

    __transient__ = True

    content: str = Field(kw_only=False)


class ModelMessage(ModelEvent):
    """Single message emitted by the model.

    Transient: already embedded in ``ModelResponse.message``.
    Not persisted to durable storage by default.
    """

    __transient__ = True

    content: str = Field(kw_only=False)

    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BinaryResult:
    """Binary result emitted by the model."""

    data: bytes
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.get("filename", "generated_file")

    async def content(self) -> bytes:
        return self.data


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


class UsageEvent(BaseEvent):
    """Token usage for a single unit of work, emitted at the point the tokens
    are spent.

    Decoupled from :class:`ModelResponse` so token accounting is independent of
    whether a response is produced and persisted: every LLM call (main loop,
    live session, history compaction, memory aggregation) and every sub-task
    rollup emits one of these onto the stream. :class:`~ag2.UsageReport`
    aggregates the event log over these events alone — the single source of
    truth, so there is no double counting.

    Persisted (not transient): the report reads it back from history.
    Not conversational: telemetry, so it must not drive history management
    (compaction trigger, retention window, summary input).
    """

    __conversational__ = False

    usage: Usage = Field(default_factory=Usage, kw_only=False)
    kind: str = Field(default="model_call")
    """``"model_call"`` for a direct LLM call, ``"subtask"`` for a sub-agent
    rollup, ``"compaction"`` / ``"aggregation"`` for internal maintenance calls."""

    model: str | None = Field(default=None, compare=False)
    provider: str | None = Field(default=None, compare=False)
    finish_reason: str | None = Field(default=None, compare=False)
    label: str | None = Field(default=None, compare=False)
    """Sub-agent name for ``"subtask"`` events; ``None`` otherwise."""


class ModelMessageChunk(ModelEvent):
    """Chunk of a streamed model message.

    Transient: superseded by the final ``ModelResponse`` which carries the
    complete content.  Not persisted to durable storage by default.
    """

    __transient__ = True

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

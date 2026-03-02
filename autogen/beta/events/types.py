# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from .base import BaseEvent, Field


class ToolCalls(BaseEvent):
    calls: list["ToolCall"] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.calls)


class ToolResults(BaseEvent):
    results: list["ToolResult | ToolError"]


class ToolEvent(BaseEvent):
    pass


class ToolCall(ToolEvent):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str


class ToolResult(ToolEvent):
    parent_id: str
    name: str
    content: str


class ToolError(ToolEvent):
    parent_id: str
    name: str
    content: str
    error: Exception


class ToolNotFoundErrorEvent(ToolError):  # noqa: N818
    pass


class ModelRequest(BaseEvent):
    content: str


class ModelEvent(ToolEvent):
    pass


class ModelReasoning(ModelEvent):
    content: str


class ModelMessage(ModelEvent):
    content: str


class ModelResponse(ModelEvent):
    """Final ModelMessage."""

    message: ModelMessage | None = None
    tool_calls: ToolCalls = Field(default_factory=ToolCalls)
    usage: dict[str, float] = Field(default_factory=dict)


class ModelMessageChunk(ModelEvent):
    content: str


class HumanInputRequest(BaseEvent):
    content: str


class HumanMessage(BaseEvent):
    """HumanInputRequest Response."""

    content: str

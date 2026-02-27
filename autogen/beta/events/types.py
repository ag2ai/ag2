# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from .base import BaseEvent


class ToolCalls(BaseEvent):
    calls: list["ToolCall"]

    def __len__(self) -> int:
        return len(self.calls)


class ToolResults(BaseEvent):
    results: list["ToolResult | ToolError"]


class ToolCall(BaseEvent):
    id: str
    name: str
    arguments: str


class ToolResult(BaseEvent):
    id: str
    name: str
    content: str


class ToolError(BaseEvent):
    id: str
    name: str
    content: str


class ModelRequest(BaseEvent):
    content: str


class ModelReasoning(BaseEvent):
    content: str


class ModelMessage(BaseEvent):
    content: str


class ModelResponse(BaseEvent):
    """Final ModelMessage."""

    message: ModelMessage | None
    tool_calls: ToolCalls
    usage: dict[str, float]


class ModelMessageChunk(BaseEvent):
    content: str


class HITL(BaseEvent):
    content: str


class UserMessage(BaseEvent):
    """HITL Response."""

    content: str

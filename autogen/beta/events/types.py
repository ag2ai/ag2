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


class ModelMessageChunk(BaseEvent):
    content: str


class HITL(BaseEvent):
    content: str


class UserMessage(BaseEvent):
    """HITL Response."""

    content: str

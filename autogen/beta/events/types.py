from .base import BaseEvent


class ToolCall(BaseEvent):
    name: str
    arguments: str


class StreamToolCall(BaseEvent):
    name: str
    arguments: str


class ToolResult(BaseEvent):
    name: str
    result: str


class ToolError(BaseEvent):
    name: str
    error: str


class StreamToolResult(BaseEvent):
    name: str
    result: str


class MoveSpeaker(BaseEvent):
    speaker: str
    context: str


class ModelRequest(BaseEvent):
    prompt: str


class ModelReasoning(BaseEvent):
    content: str


class ModelResponse(BaseEvent):
    content: str


class StreamModelResult(BaseEvent):
    content: str


class HITL(BaseEvent):
    message: str


class UserMessage(BaseEvent):
    content: str


class ThreadStarted(BaseEvent):
    thread_id: str


class ThreadComplete(BaseEvent):
    thread_id: str


class ThreadError(BaseEvent):
    thread_id: str
    error: str


class RunStarted(BaseEvent):
    run_id: str


class RunComplete(BaseEvent):
    run_id: str


class RunError(BaseEvent):
    run_id: str
    error: str

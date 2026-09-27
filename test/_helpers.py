# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Any

from ag2 import Context
from ag2.events import BaseEvent, DataInput, Input, ModelReasoning, ModelResponse, ProviderReplay, TextInput
from ag2.middleware import BaseMiddleware, LLMCall, Middleware
from ag2.tools import tool
from ag2.tools.types import FunctionTool, FunctionToolSchema, Tool, ToolSchema


class DurableReasoning(ModelReasoning, ProviderReplay):
    """Provider reasoning item that must be replayed, like OpenAIReasoningEvent.

    Reasoning is transient by default; a provider that persists its item opts out
    (see ``ag2.config.openai.events``). Redeclared here so tests outside a
    provider package can exercise the anchor case without importing its SDK.
    """

    __transient__ = False
    __replay_role__ = "anchor"


class ProviderTurnState(BaseEvent, ProviderReplay):
    """Provider-native object standing in for a whole assistant turn.

    Like ``XAIAssistantEvent``: the only way to rebuild a turn carrying
    ``tool_calls`` for a provider whose SDK cannot construct one from primitives.
    Redeclared here so tests outside a provider package can exercise the turn
    case without importing its SDK.
    """

    __replay_role__ = "turn"


class LLMCalls:
    """What each model call was given, read on a real ``on_llm_call`` hook.

    For assertions past ``TrackingConfig``, which keeps only ``messages[-1]``.

    Pass :meth:`middleware` to ``Agent(middleware=)`` to see the history before
    assembly, or to ``ask(middleware=)`` to see exactly what the client gets.
    """

    def __init__(self) -> None:
        self.messages: list[list[BaseEvent]] = []
        self.prompts: list[list[str]] = []
        self.variables: list[dict[str, Any]] = []

    def middleware(self) -> Middleware:
        return Middleware(_LLMCallRecorder, calls=self)


class _LLMCallRecorder(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, calls: LLMCalls) -> None:
        super().__init__(event, context)
        self._calls = calls

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        self._calls.messages.append(list(events))
        self._calls.prompts.append(list(context.prompt))
        self._calls.variables.append(dict(context.variables))
        return await call_next(events, context)


@tool
def lookup() -> str:
    """Look something up."""
    return "42"


def text_of(part: Input) -> str:
    """The text of a part the test expects to be a ``TextInput``."""
    assert isinstance(part, TextInput), part
    return part.content


def data_of(part: Input) -> object:
    """The payload of a part the test expects to be a ``DataInput``."""
    assert isinstance(part, DataInput), part
    return part.data


def function_schemas(schemas: Iterable[ToolSchema]) -> list[FunctionToolSchema]:
    """``schemas``, each asserted to be a function tool's."""
    narrowed = []
    for schema in schemas:
        assert isinstance(schema, FunctionToolSchema), schema
        narrowed.append(schema)
    return narrowed


def function_tool(tool: Tool) -> FunctionTool:
    """``tool``, asserted to be a ``FunctionTool``."""
    assert isinstance(tool, FunctionTool), tool
    return tool

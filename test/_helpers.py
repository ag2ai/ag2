# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from ag2.events import BaseEvent, DataInput, Input, ModelReasoning, ProviderReplay, TextInput
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

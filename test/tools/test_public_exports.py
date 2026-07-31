# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import ag2.tools
from ag2 import Agent
from ag2.tools import (
    ClientTool,
    FunctionDefinition,
    FunctionParameters,
    FunctionTool,
    FunctionToolSchema,
    Tool,
    ToolSchema,
    Toolkit,
    tool,
)


def test_every_public_name_resolves() -> None:
    """``__all__`` must not advertise a name the module does not expose."""
    assert [name for name in ag2.tools.__all__ if not hasattr(ag2.tools, name)] == []


class TestToolAbstraction:
    """ADR 0002 — one ``Tool`` abstraction; every kind of tool implements it."""

    def test_function_tool_is_a_tool(self) -> None:
        assert issubclass(FunctionTool, Tool)

    def test_client_tool_is_a_tool(self) -> None:
        assert issubclass(ClientTool, Tool)

    def test_toolkit_is_a_tool(self) -> None:
        # The composite: a toolkit *is a* Tool, so an agent accepts it anywhere
        # it accepts a tool.
        assert issubclass(Toolkit, Tool)

    def test_function_tool_schema_is_a_tool_schema(self) -> None:
        assert issubclass(FunctionToolSchema, ToolSchema)


class TestPublicReturnTypesAreImportable:
    """Callables AG2 exports must return types a caller can name."""

    def test_tool_decorator_returns_function_tool(self) -> None:
        @tool
        def my_tool(a: str) -> str:
            """Tool description."""
            return a

        assert isinstance(my_tool, FunctionTool)

    def test_agent_as_tool_returns_function_tool(self) -> None:
        child = Agent("child", prompt="You are a child agent.")

        assert isinstance(child.as_tool(description="Delegate to the child."), FunctionTool)

    def test_toolkit_tool_decorator_returns_function_tool(self) -> None:
        toolkit = Toolkit()

        @toolkit.tool
        def my_tool(a: str) -> str:
            """Tool description."""
            return a

        assert isinstance(my_tool, FunctionTool)


def test_function_tool_schema_is_built_from_public_parts() -> None:
    """``FunctionDefinition`` / ``FunctionParameters`` describe a function tool."""
    parameters: FunctionParameters = {"type": "object", "properties": {}}
    schema = FunctionToolSchema(function=FunctionDefinition(name="my_tool", parameters=parameters))

    assert schema.type == "function"
    assert schema.function.name == "my_tool"

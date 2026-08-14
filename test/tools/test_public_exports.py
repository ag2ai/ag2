# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import ag2.tools
import ag2.tools.types
from ag2 import Agent
from ag2.tools import Toolkit, tool
from ag2.tools.types import (
    ClientTool,
    FunctionDefinition,
    FunctionParameters,
    FunctionTool,
    FunctionToolSchema,
    Tool,
    ToolSchema,
)

# Ready-to-use tools that must stay importable from `ag2.tools`, so the
# re-homing of the abstractions cannot quietly take them with it.
_CONCRETE_TOOLS = ("CodeExecutionTool", "ShellTool", "WebSearchTool", "MCPServerTool")


def test_every_public_name_resolves() -> None:
    """``__all__`` must not advertise a name the module does not expose."""
    assert [name for name in ag2.tools.__all__ if not hasattr(ag2.tools, name)] == []
    assert [name for name in ag2.tools.types.__all__ if not hasattr(ag2.tools.types, name)] == []


class TestImportPathsStaySeparate:
    """`ag2.tools` is ready-to-use tools; `ag2.tools.types` is the abstractions."""

    def test_abstractions_are_importable_from_one_place(self) -> None:
        assert set(ag2.tools.types.__all__) == {
            "ClientTool",
            "FunctionDefinition",
            "FunctionParameters",
            "FunctionTool",
            "FunctionToolSchema",
            "Tool",
            "ToolSchema",
            "Toolkit",
        }

    def test_abstractions_are_not_mixed_into_ag2_tools(self) -> None:
        # Toolkit predates this split and stays exported from both.
        assert [name for name in ag2.tools.types.__all__ if name in ag2.tools.__all__] == ["Toolkit"]

    def test_concrete_tools_are_untouched(self) -> None:
        assert [name for name in _CONCRETE_TOOLS if name not in ag2.tools.__all__] == []


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

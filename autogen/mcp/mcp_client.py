# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import sys
from typing import Any, Optional, Union

from ..doc_utils import export_module
from ..import_utils import optional_import_block, require_optional_import
from ..tools import Tool, Toolkit

with optional_import_block():
    from mcp import ClientSession
    from mcp.types import (
        CallToolResult,
        TextContent,
    )
    from mcp.types import (
        Tool as MCPTool,
    )

__all__ = ["create_toolkit"]


class MCPClient:
    @staticmethod
    def _convert_call_tool_result(  # type: ignore[no-any-unimported]
        call_tool_result: "CallToolResult",  # type: ignore[no-any-unimported]
    ) -> tuple[Union[str, list[str]], Any]:
        text_contents: list[TextContent] = []  # type: ignore[no-any-unimported]
        non_text_contents = []
        for content in call_tool_result.content:
            if isinstance(content, TextContent):
                text_contents.append(content)
            else:
                non_text_contents.append(content)

        tool_content: Union[str, list[str]] = [content.text for content in text_contents]
        if len(text_contents) == 1:
            tool_content = tool_content[0]

        if call_tool_result.isError:
            raise ValueError(f"Tool call failed: {tool_content}")

        return tool_content, non_text_contents or None

    @classmethod
    @require_optional_import("mcp", "mcp")
    def convert_tool(  # type: ignore[no-any-unimported]
        cls, tool: Any, session: "ClientSession", **kwargs: Any
    ) -> Tool:
        if not isinstance(tool, MCPTool):
            raise ValueError(f"Expected an instance of `mcp.types.Tool`, got {type(tool)}")

        # needed for type checking
        mcp_tool: MCPTool = tool  # type: ignore[no-any-unimported]

        async def call_tool(  # type: ignore[no-any-unimported]
            **arguments: dict[str, Any],
        ) -> tuple[Union[str, list[str]], Any]:
            call_tool_result = await session.call_tool(tool.name, arguments)
            return MCPClient._convert_call_tool_result(call_tool_result)

        ag2_tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            func_or_tool=call_tool,
            parameters_json_schema=mcp_tool.inputSchema,
        )
        return ag2_tool

    @classmethod
    @require_optional_import("mcp", "mcp")
    async def load_mcp_toolkit(cls, session: "ClientSession") -> Toolkit:  # type: ignore[no-any-unimported]
        """Load all available MCP tools and convert them to AG2 Toolkit."""
        tools = await session.list_tools()
        ag2_tools: list[Tool] = [cls.convert_tool(tool=tool, session=session) for tool in tools.tools]

        return Toolkit(tools=ag2_tools)

    @classmethod
    def get_unsupported_reason(cls) -> Optional[str]:
        if sys.version_info < (3, 10):
            return "This submodule is only supported for Python versions 3.10 and above"

        with optional_import_block() as result:
            import mcp  # noqa: F401

        if not result.is_successful:
            return "Please install `mcp` extra to use this module:\n\n\tpip install ag2[mcp]"

        return None


@export_module("autogen.mcp")
async def create_toolkit(session: "ClientSession") -> Toolkit:  # type: ignore[no-any-unimported]
    """Create a toolkit from the MCP client session.

    Args:
        session (ClientSession): The MCP client session.
    Returns:
        Toolkit: The toolkit containing the converted tools.
    """

    return await MCPClient.load_mcp_toolkit(session=session)

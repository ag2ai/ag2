# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import anyio
from pydantic import BaseModel

from ..doc_utils import export_module
from ..import_utils import optional_import_block, require_optional_import
from ..tools import Tool, Toolkit

with optional_import_block():
    from mcp import ClientSession
    from mcp.types import (
        CallToolResult,
        ReadResourceResult,
        ResourceTemplate,
        TextContent,
    )
    from mcp.types import (
        Tool as MCPTool,
    )

__all__ = ["ResultSaved", "create_toolkit"]


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
    def convert_resource(  # type: ignore[no-any-unimported]
        cls,
        resource_template: Any,
        session: "ClientSession",
        resource_download_folder: Optional[Path],
        **kwargs: Any,
    ) -> Tool:
        if not isinstance(resource_template, ResourceTemplate):
            raise ValueError(f"Expected an instance of `mcp.types.ResourceTemplate`, got {type(resource_template)}")

        # needed for type checking
        mcp_resource: ResourceTemplate = resource_template  # type: ignore[no-any-unimported]

        uri_description = f"""A URI template (according to RFC 6570) that can be used to construct resource URIs.
Here is the correct format for the URI template:
{mcp_resource.uriTemplate}
"""

        async def call_resource(uri: Annotated[str, uri_description]) -> Union[ReadResourceResult, ResultSaved]:  # type: ignore[no-any-unimported]
            result = await session.read_resource(uri)

            if not resource_download_folder:
                return result

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = uri.split("://")[-1] + f"_{timestamp}"
            file_path = resource_download_folder / filename

            async with await anyio.open_file(file_path, "w") as f:
                await f.write(result.model_dump_json(indent=4))

            return ResultSaved(
                explanation=f"Request for uri {uri} was saved to {file_path}",
                file_path=file_path,
            )

        # Wrap resource as AG2 tool
        ag2_tool = Tool(
            name=mcp_resource.name,
            description=mcp_resource.description,
            func_or_tool=call_resource,
        )
        return ag2_tool

    @classmethod
    @require_optional_import("mcp", "mcp")
    async def load_mcp_toolkit(
        cls,
        session: "ClientSession",
        *,
        use_mcp_tools: bool,
        use_mcp_resources: bool,
        resource_download_folder: Optional[Path],
    ) -> Toolkit:  # type: ignore[no-any-unimported]
        """Load all available MCP tools and convert them to AG2 Toolkit."""
        all_ag2_tools: list[Tool] = []

        if use_mcp_tools:
            tools = await session.list_tools()
            ag2_tools: list[Tool] = [cls.convert_tool(tool=tool, session=session) for tool in tools.tools]
            all_ag2_tools.extend(ag2_tools)

        if use_mcp_resources:
            resource_templates = await session.list_resource_templates()
            ag2_resources: list[Tool] = [
                cls.convert_resource(
                    resource_template=resource_template,
                    session=session,
                    resource_download_folder=resource_download_folder,
                )
                for resource_template in resource_templates.resourceTemplates
            ]
            all_ag2_tools.extend(ag2_resources)

        return Toolkit(tools=all_ag2_tools)

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
async def create_toolkit(
    session: "ClientSession",
    *,
    use_mcp_tools: bool = True,
    use_mcp_resources: bool = True,
    resource_download_folder: Optional[Union[Path, str]] = None,
) -> Toolkit:  # type: ignore[no-any-unimported]
    """Create a toolkit from the MCP client session.

    Args:
        session (ClientSession): The MCP client session.
        use_mcp_tools (bool): Whether to include MCP tools in the toolkit.
        use_mcp_resources (bool): Whether to include MCP resources in the toolkit.
        resource_download_folder (Optional[Union[Path, str]]): The folder to download files to.
    Returns:
        Toolkit: The toolkit containing the converted tools.
    """
    if resource_download_folder:
        if isinstance(resource_download_folder, str):
            resource_download_folder = Path(resource_download_folder)
        await anyio.to_thread.run_sync(lambda: resource_download_folder.mkdir(parents=True, exist_ok=True))

    return await MCPClient.load_mcp_toolkit(
        session=session,
        use_mcp_tools=use_mcp_tools,
        use_mcp_resources=use_mcp_resources,
        resource_download_folder=resource_download_folder,
    )


@export_module("autogen.mcp.mcp_client")
class ResultSaved(BaseModel):
    """Result saved to a file"""

    explanation: str
    file_path: Path

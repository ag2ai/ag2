# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field

from ..doc_utils import export_module
from ..import_utils import optional_import_block, require_optional_import
from ..tools import Tool, Toolkit

with optional_import_block():
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

# Type definitions
EncodingErrorHandlerType = Literal["strict", "ignore", "replace"]

# Default constants
DEFAULT_TEXT_ENCODING = "utf-8"
DEFAULT_TEXT_ENCODING_ERROR_HANDLER: EncodingErrorHandlerType = "strict"
DEFAULT_HTTP_REQUEST_TIMEOUT = 5
DEFAULT_SSE_EVENT_READ_TIMEOUT = 60 * 5
DEFAULT_STREAMABLE_HTTP_REQUEST_TIMEOUT = timedelta(seconds=30)
DEFAULT_STREAMABLE_HTTP_SSE_EVENT_READ_TIMEOUT = timedelta(seconds=60 * 5)


class StdioConfig(BaseModel):
    """Configuration for a single stdio MCP server."""

    command: str = Field(..., description="Command to execute")
    args: list[str] = Field(..., description="Arguments for the command")
    transport: Literal["stdio"] = Field(default="stdio", description="Transport type")
    server_name: str = Field(..., description="Name of the server")
    environment: dict[str, str] | None = Field(default=None, description="Environment variables")
    working_dir: str | Path | None = Field(default=None, description="Working directory")
    encoding: str = Field(default=DEFAULT_TEXT_ENCODING, description="Character encoding")
    encoding_error_handler: EncodingErrorHandlerType = Field(
        default=DEFAULT_TEXT_ENCODING_ERROR_HANDLER, description="How to handle encoding errors"
    )
    session_options: dict[str, Any] | None = Field(default=None, description="Additional session options")


class MCPConfig(BaseModel):
    """Configuration for multiple MCP sessions using stdio transport."""

    servers: list[StdioConfig] = Field(..., description="List of stdio server configurations")


class MCPClient:
    @staticmethod
    def _convert_call_tool_result(  # type: ignore[no-any-unimported]
        call_tool_result: "CallToolResult",  # type: ignore[no-any-unimported]
    ) -> tuple[str | list[str], Any]:
        text_contents: list[TextContent] = []  # type: ignore[no-any-unimported]
        non_text_contents = []
        for content in call_tool_result.content:
            if isinstance(content, TextContent):
                text_contents.append(content)
            else:
                non_text_contents.append(content)

        tool_content: str | list[str] = [content.text for content in text_contents]
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
        ) -> tuple[str | list[str], Any]:
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
        resource_download_folder: Path | None,
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

        async def call_resource(uri: Annotated[str, uri_description]) -> ReadResourceResult | ResultSaved:  # type: ignore[no-any-unimported]
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
        resource_download_folder: Path | None,
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
    def get_unsupported_reason(cls) -> str | None:
        with optional_import_block() as result:
            import mcp  # noqa: F401

        if not result.is_successful:
            return "Please install `mcp` extra to use this module:\n\n\tpip install ag2[mcp]"

        return None


class MCPClientSessionManager:
    """A class to manage MCP client sessions using stdio transport."""

    def __init__(self):
        """Initialize the MCP client session manager."""
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}

    async def _initialize_session(self, server_name: str, session: ClientSession) -> None:
        await session.initialize()
        self.sessions[server_name] = session

    @asynccontextmanager
    async def create_stdio_session(
        self,
        config: StdioConfig,
    ) -> AsyncIterator[ClientSession]:
        """Create a new session to an MCP server using stdio transport.

        Args:
            config: StdioConfig object containing stdio session parameters

        Yields:
            ClientSession: The MCP client session
        """
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.environment,
            encoding=config.encoding,
            encoding_error_handler=config.encoding_error_handler,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        reader, writer = stdio_transport

        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(ClientSession(reader, writer)),
        )
        await self._initialize_session(config.server_name, session)
        yield session

    @asynccontextmanager
    async def open_session(
        self,
        config: StdioConfig,
    ) -> AsyncIterator[ClientSession]:
        """Open a new session to an MCP server based on configuration.

        Args:
            config: StdioConfig object containing session configuration

        Yields:
            ClientSession: The MCP client session
        """
        if isinstance(config, StdioConfig):
            async with self.create_stdio_session(config) as session:
                yield session


@export_module("autogen.mcp")
async def create_toolkit(
    session: "ClientSession",
    *,
    use_mcp_tools: bool = True,
    use_mcp_resources: bool = True,
    resource_download_folder: Path | str | None = None,
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

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict

from autogen.beta.config.anthropic.mappers import extract_mcp_servers, tool_to_api
from autogen.beta.context import Context
from autogen.beta.tools.builtin.mcp_server import MCPServerTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "mcp_toolset",
        "mcp_server_name": "example-mcp",
    }


@pytest.mark.asyncio
async def test_extract_mcp_servers_defaults(context: Context) -> None:
    tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")

    [schema] = await tool.schemas(context)

    assert extract_mcp_servers([schema]) == [
        {
            "type": "url",
            "url": "https://mcp.example.com/sse",
            "name": "example-mcp",
        },
    ]


@pytest.mark.asyncio
async def test_with_auth_token(context: Context) -> None:
    tool = MCPServerTool(
        server_url="https://mcp.example.com/sse",
        server_label="example-mcp",
        authorization_token="token123",
    )

    [schema] = await tool.schemas(context)

    assert extract_mcp_servers([schema]) == [
        {
            "type": "url",
            "url": "https://mcp.example.com/sse",
            "name": "example-mcp",
            "authorization_token": "token123",
        },
    ]


@pytest.mark.asyncio
async def test_allowed_tools(context: Context) -> None:
    tool = MCPServerTool(
        server_url="https://mcp.example.com/sse",
        server_label="example-mcp",
        allowed_tools=["search_events", "create_event"],
    )

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "mcp_toolset",
        "mcp_server_name": "example-mcp",
        "default_config": {"enabled": False},
        "configs": {
            "search_events": {"enabled": True},
            "create_event": {"enabled": True},
        },
    }


@pytest.mark.asyncio
async def test_blocked_tools(context: Context) -> None:
    tool = MCPServerTool(
        server_url="https://mcp.example.com/sse",
        server_label="example-mcp",
        blocked_tools=["delete_all"],
    )

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "mcp_toolset",
        "mcp_server_name": "example-mcp",
        "configs": {
            "delete_all": {"enabled": False},
        },
    }


@pytest.mark.asyncio
async def test_extract_mcp_servers_skips_non_mcp(context: Context) -> None:
    from autogen.beta.tools.builtin.web_search import WebSearchTool

    mcp_tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")
    ws_tool = WebSearchTool()

    [mcp_schema] = await mcp_tool.schemas(context)
    [ws_schema] = await ws_tool.schemas(context)

    servers = extract_mcp_servers([mcp_schema, ws_schema])
    assert servers == [IsPartialDict({"name": "example-mcp"})]

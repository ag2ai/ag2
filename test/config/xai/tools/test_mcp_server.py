# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.xai.mappers import tool_to_api
from ag2.tools.builtin.mcp_server import MCPServerTool


@pytest.mark.asyncio
async def test_required_fields(context: Context) -> None:
    tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="ex")

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.HasField("mcp")
    assert api.mcp.server_url == "https://mcp.example.com/sse"
    assert api.mcp.server_label == "ex"


@pytest.mark.asyncio
async def test_authorization_token_becomes_bearer(context: Context) -> None:
    tool = MCPServerTool(
        server_url="https://mcp.example.com/sse",
        server_label="ex",
        authorization_token="secret-xyz",
    )

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.mcp.authorization == "Bearer secret-xyz"


@pytest.mark.asyncio
async def test_allowed_tools(context: Context) -> None:
    tool = MCPServerTool(
        server_url="https://mcp.example.com/sse",
        server_label="ex",
        allowed_tools=["search", "fetch"],
    )

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert list(api.mcp.allowed_tool_names) == ["search", "fetch"]

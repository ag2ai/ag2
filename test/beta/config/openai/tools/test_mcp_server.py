# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.openai.mappers import tool_to_api, tool_to_responses_api
from autogen.beta.context import Context
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.mcp_server import MCPServerTool


class TestCompletionsApi:
    @pytest.mark.asyncio
    async def test_unsupported(self, context: Context) -> None:
        tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")

        [schema] = await tool.schemas(context)

        with pytest.raises(UnsupportedToolError):
            tool_to_api(schema)


class TestResponsesApi:
    @pytest.mark.asyncio
    async def test_defaults(self, context: Context) -> None:
        tool = MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp")

        [schema] = await tool.schemas(context)

        assert tool_to_responses_api(schema) == {
            "type": "mcp",
            "server_label": "example-mcp",
            "server_url": "https://mcp.example.com/sse",
            "require_approval": "never",
        }

    @pytest.mark.asyncio
    async def test_with_allowed_tools(self, context: Context) -> None:
        tool = MCPServerTool(
            server_url="https://mcp.example.com/sse",
            server_label="example-mcp",
            allowed_tools=["search", "create"],
        )

        [schema] = await tool.schemas(context)

        assert tool_to_responses_api(schema) == {
            "type": "mcp",
            "server_label": "example-mcp",
            "server_url": "https://mcp.example.com/sse",
            "require_approval": "never",
            "allowed_tools": ["search", "create"],
        }

    @pytest.mark.asyncio
    async def test_with_headers(self, context: Context) -> None:
        tool = MCPServerTool(
            server_url="https://mcp.example.com/sse",
            server_label="example-mcp",
            headers={"Authorization": "Bearer token123", "X-Custom": "value"},
        )

        [schema] = await tool.schemas(context)

        assert tool_to_responses_api(schema) == {
            "type": "mcp",
            "server_label": "example-mcp",
            "server_url": "https://mcp.example.com/sse",
            "require_approval": "never",
            "headers": {"Authorization": "Bearer token123", "X-Custom": "value"},
        }

    @pytest.mark.asyncio
    async def test_auth_token_becomes_header(self, context: Context) -> None:
        tool = MCPServerTool(
            server_url="https://mcp.example.com/sse",
            server_label="example-mcp",
            authorization_token="token123",
        )

        [schema] = await tool.schemas(context)

        assert tool_to_responses_api(schema) == {
            "type": "mcp",
            "server_label": "example-mcp",
            "server_url": "https://mcp.example.com/sse",
            "require_approval": "never",
            "headers": {"Authorization": "Bearer token123"},
        }

    @pytest.mark.asyncio
    async def test_headers_take_precedence_over_auth_token(self, context: Context) -> None:
        tool = MCPServerTool(
            server_url="https://mcp.example.com/sse",
            server_label="example-mcp",
            authorization_token="token123",
            headers={"Authorization": "Custom auth"},
        )

        [schema] = await tool.schemas(context)

        result = tool_to_responses_api(schema)
        assert result["headers"] == {"Authorization": "Custom auth"}

    @pytest.mark.asyncio
    async def test_with_description(self, context: Context) -> None:
        tool = MCPServerTool(
            server_url="https://mcp.example.com/sse",
            server_label="example-mcp",
            description="An example MCP server for testing",
        )

        [schema] = await tool.schemas(context)

        assert tool_to_responses_api(schema) == {
            "type": "mcp",
            "server_label": "example-mcp",
            "server_url": "https://mcp.example.com/sse",
            "require_approval": "never",
            "server_description": "An example MCP server for testing",
        }

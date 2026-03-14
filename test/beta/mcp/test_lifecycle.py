# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from httpx import Response
from mcphero import MCPServerConfig
from mcphero.adapters.base_adapter import InitMode

from autogen.beta import Agent


@pytest.mark.asyncio()
async def test_authorization_after_failed_called_then_recall_succeeds(respx_mock):
    tool_name = "test_tool_name"
    tool_description = "test_tool_description"

    init_response = {
        "jsonrpc": "2.0",
        "id": "id",
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "serverInfo": {"name": "test", "version": "1.0"},
        },
    }
    tools_response = {
        "jsonrpc": "2.0",
        "id": "id",
        "result": {
            "tools": [
                {
                    "name": tool_name,
                    "description": tool_description,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "The message to send"},
                        },
                        "required": ["message"],
                    },
                }
            ]
        },
    }

    # Sequence: tools/list → 401, initialize → 200, notification → 200, tools/list retry → 200
    respx_mock.post("https://mcp.example.com").mock(
        side_effect=[
            Response(401),
            Response(200, json=init_response),
            Response(200),
            Response(200, json=tools_response),
        ]
    )

    agent = Agent(
        name="test",
        mcp_servers=[MCPServerConfig(url="https://mcp.example.com", init_mode=InitMode.on_fail)],
    )
    await agent._ensure_mcp_servers_initialized()

    assert len(agent.tools) == 1
    assert agent.tools[0].name == tool_name
    assert agent.tools[0].schema.function.name == tool_name
    assert agent.tools[0].schema.function.description == tool_description

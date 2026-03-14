# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from httpx import Response

from autogen.beta import Agent


@pytest.mark.asyncio()
async def test_tool_registered_from_mcp_server(respx_mock):
    tool_name = "test_tool_name"
    tool_description = "test_tool_description"

    respx_mock.post("https://mcp.example.com").mock(
        return_value=Response(
            200,
            json={
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
                                    "count": {"type": "integer", "description": "How many times"},
                                },
                                "required": ["message"],
                            },
                        }
                    ]
                },
            },
        )
    )

    agent = Agent(
        name="test",
        mcp_servers=["https://mcp.example.com"],
    )
    await agent._ensure_mcp_servers_initialized()

    assert len(agent.tools) == 1
    assert agent.tools[0].name == tool_name
    assert agent.tools[0].schema.function.name == tool_name
    assert agent.tools[0].schema.function.description == tool_description

    params = agent.tools[0].schema.function.parameters
    assert params["type"] == "object"
    assert "message" in params["properties"]
    assert params["properties"]["message"]["type"] == "string"
    assert "count" in params["properties"]
    assert params["properties"]["count"]["type"] == "integer"
    assert "message" in params["required"]
    assert "count" not in params["required"]

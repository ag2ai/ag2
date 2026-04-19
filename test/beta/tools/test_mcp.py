# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from httpx import Response

from autogen.beta import Agent, Context
from autogen.beta.testing import TestConfig
from autogen.beta.tools import MCPServer


@pytest.mark.asyncio()
async def test_tool_registered_from_mcp_server(respx_mock, context: Context):
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
        tools=[
            MCPServer("https://mcp.example.com"),
        ],
        config=TestConfig('{"data": 42}'),
    )
    await agent.ask("test")

    assert len(agent.tools) == 1
    schemas = await agent.tools[0].schemas(context)
    # assert agent.tools[0].name == tool_name
    assert schemas[0].function.name == tool_name
    assert schemas[0].function.description == tool_description

    params = schemas[0].function.parameters
    assert params["type"] == "object"
    assert "message" in params["properties"]
    assert params["properties"]["message"]["type"] == "string"
    assert "count" in params["properties"]
    assert params["properties"]["count"]["type"] == "integer"
    assert "message" in params["required"]
    assert "count" not in params["required"]

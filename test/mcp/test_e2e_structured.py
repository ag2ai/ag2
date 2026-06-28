# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from ag2 import Agent
from ag2.mcp import MCPServer
from ag2.mcp.testing import connect
from ag2.testing import TestConfig


class Weather(BaseModel):
    city: str
    temp_c: float


@pytest.mark.asyncio
class TestE2EStructured:
    async def test_structured_content_populated(self) -> None:
        agent = Agent(
            "weather",
            config=TestConfig('{"city": "SF", "temp_c": 18.5}'),
            response_schema=Weather,
        )
        server = MCPServer(agent)

        async with connect(server) as session:
            tool = next(t for t in (await session.list_tools()).tools if t.name == "ask")
            result = await session.call_tool("ask", {"message": "weather in SF?"})

        assert tool.outputSchema is not None
        assert result.isError is False
        assert result.structuredContent == {"city": "SF", "temp_c": 18.5}
        # Raw JSON body is still present as text content.
        assert any(c.type == "text" for c in result.content)

    async def test_invalid_structured_output_is_error(self) -> None:
        agent = Agent(
            "weather",
            config=TestConfig("not json at all"),
            response_schema=Weather,
        )
        server = MCPServer(agent)

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "weather?"})

        assert result.isError is True

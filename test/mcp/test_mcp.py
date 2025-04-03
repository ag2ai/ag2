# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from autogen import AssistantAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports, skip_on_missing_imports
from autogen.mcp import create_toolkit

from ..conftest import Credentials

with optional_import_block():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client


@skip_on_missing_imports(
    [
        "mcp.client.stdio",
        "mcp.server.fastmcp",
    ],
    "mcp",
)
class TestMCPClient:
    @pytest.fixture
    def server_params(self) -> "StdioServerParameters":  # type: ignore[no-any-unimported]
        server_file = Path(__file__).parent / "math_server.py"
        return StdioServerParameters(
            command="python",
            args=[str(server_file)],
        )

    @pytest.mark.asyncio
    async def test_mcp_issue_with_stdio_client_context_manager(self, server_params: "StdioServerParameters") -> None:  # type: ignore[no-any-unimported]
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as _:
                pass
            print("exit ClientSession")
        print("exit stdio_client")

    @pytest.mark.asyncio
    async def test_convert_tool(self, server_params: "StdioServerParameters", mock_credentials: Credentials) -> None:  # type: ignore[no-any-unimported]
        async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            toolkit = await create_toolkit(session=session)
            assert len(toolkit) == 2

            agent = AssistantAgent(
                name="agent",
                llm_config=mock_credentials.llm_config,
            )
            toolkit.register_for_llm(agent)
            expected_schema = [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers",
                        "parameters": {
                            "properties": {
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                            },
                            "required": ["a", "b"],
                            "title": "addArguments",
                            "type": "object",
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "description": "Multiply two numbers",
                        "parameters": {
                            "properties": {
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                            },
                            "required": ["a", "b"],
                            "title": "multiplyArguments",
                            "type": "object",
                        },
                    },
                },
            ]
            assert agent.llm_config["tools"] == expected_schema  # type: ignore[index]

    @pytest.mark.asyncio
    @run_for_optional_imports("openai", "openai")
    async def test_with_llm(self, server_params: "StdioServerParameters", credentials_gpt_4o_mini: Credentials) -> None:  # type: ignore[no-any-unimported]
        async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            toolkit = await create_toolkit(session=session)

            agent = AssistantAgent(
                name="agent",
                llm_config=credentials_gpt_4o_mini.llm_config,
            )
            toolkit.register_for_llm(agent)

            result = await agent.a_run(
                message="What is 1234 + 5678?",
                tools=toolkit.tools,
                max_turns=3,
                user_input=False,
                summary_method="reflection_with_llm",
            )
            await result.process()
            summary = await result.summary
            assert "6912" in summary

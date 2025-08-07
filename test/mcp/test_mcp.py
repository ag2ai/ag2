# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import tempfile
from asyncio.subprocess import PIPE, Process, create_subprocess_exec
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from pydantic.networks import AnyUrl

from autogen import AssistantAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.mcp.mcp_client import (
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DEFAULT_SSE_EVENT_READ_TIMEOUT,
    MCPClientSessionManager,
    ResultSaved,
    SseConfig,
    StdioConfig,
    create_toolkit,
)

from ..conftest import Credentials

with optional_import_block():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import ReadResourceResult, TextResourceContents


@asynccontextmanager
async def run_sse_server(
    *,
    mcp_server_path: str,
    storage_path: str,
    env_vars: Optional[Dict[str, str]] = None,
    startup_wait_secs: float = 3.0,
) -> AsyncGenerator[Process, None]:
    """
    Async context manager to run a Python subprocess for SSE server with custom env vars.

    Args:
        mcp_server_path: Path to the Python script to run.
        storage_path: Path for the server to store files.
        env_vars: Environment variables to export to the subprocess.
        startup_wait_secs: Time to wait for the server to start (in seconds).
    Yields:
        An asyncio.subprocess.Process object.
    """
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    process = await create_subprocess_exec(
        "python", mcp_server_path, "sse", "--storage-path", storage_path, env=env, stdout=PIPE, stderr=PIPE
    )

    # Optional startup delay to let the server initialize
    await asyncio.sleep(startup_wait_secs)

    try:
        yield process
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()


class TestMCPClient:
    @pytest.fixture
    def server_params(self) -> "StdioServerParameters":  # type: ignore[no-any-unimported]
        server_file = Path(__file__).parent / "math_server.py"
        return StdioServerParameters(
            command="python3",
            args=[str(server_file)],
        )

    @pytest.mark.asyncio
    async def test_mcp_issue_with_stdio_client_context_manager(self, server_params: "StdioServerParameters") -> None:  # type: ignore[no-any-unimported]
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as _:
                pass
            print("exit ClientSession")
        print("exit stdio_client")

    @pytest.mark.asyncio
    async def test_tools_schema(self, server_params: "StdioServerParameters", mock_credentials: Credentials) -> None:  # type: ignore[no-any-unimported]
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            # Initialize the connection
            await session.initialize()
            toolkit = await create_toolkit(session=session)
            assert len(toolkit) == 3

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
                            "type": "object",
                            "properties": {
                                "arguments": {
                                    "type": "object",
                                    "description": "arguments",
                                    "additionalProperties": True,
                                }
                            },
                            "required": ["arguments"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "description": "Multiply two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "arguments": {
                                    "type": "object",
                                    "description": "arguments",
                                    "additionalProperties": True,
                                }
                            },
                            "required": ["arguments"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "description": "Echo a message as a resource",
                        "name": "echo_resource",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "uri": {
                                    "type": "string",
                                    "description": "A URI template (according to RFC 6570) that can be used to construct resource URIs.\nHere is the correct format for the URI template:\necho://{message}\n",
                                }
                            },
                            "required": ["uri"],
                        },
                    },
                },
            ]
            assert agent.llm_config["tools"] == expected_schema  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_convert_resource(self, server_params: "StdioServerParameters") -> None:  # type: ignore[no-any-unimported]
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            # Initialize the connection
            await session.initialize()
            toolkit = await create_toolkit(session=session)
            echo_resource_tool = toolkit.get_tool("echo_resource")
            assert echo_resource_tool is not None
            assert echo_resource_tool.name == "echo_resource"

            result = await echo_resource_tool(uri="echo://AG2User")
            assert isinstance(result, ReadResourceResult)
            expected_result = [
                TextResourceContents(uri=AnyUrl("echo://AG2User"), mimeType="text/plain", text="Resource echo: AG2User")
            ]
            assert result.contents == expected_result

    @pytest.mark.asyncio
    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set, skipping integration test.")
    async def test_register_for_llm_tool(
        self, server_params: "StdioServerParameters", credentials_gpt_4o_mini: Credentials
    ) -> None:  # type: ignore[no-any-unimported]
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            # Initialize the connection
            await session.initialize()
            toolkit = await create_toolkit(session=session)
            agent = AssistantAgent(
                name="agent",
                llm_config=credentials_gpt_4o_mini.llm_config,
            )
            toolkit.register_for_llm(agent)
            assert len(agent.tools) == len(toolkit.tools)

    @pytest.mark.asyncio
    async def test_convert_resource_with_download_folder(self, server_params: "StdioServerParameters") -> None:  # type: ignore[no-any-unimported]
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            await session.initialize()
            with tempfile.TemporaryDirectory() as tmp:
                temp_path = Path(tmp)
                temp_path.mkdir(parents=True, exist_ok=True)
                toolkit = await create_toolkit(session=session, resource_download_folder=temp_path)
                echo_resource_tool = toolkit.get_tool("echo_resource")
                result = await echo_resource_tool(uri="echo://AG2User")
                assert isinstance(result, ResultSaved)

                async with await anyio.open_file(result.file_path, "r") as f:
                    content = await f.read()
                    parsed = json.loads(content)
                    loaded_result = ReadResourceResult.model_validate(parsed)

                    expected_result = [
                        TextResourceContents(
                            uri=AnyUrl("echo://AG2User"),
                            mimeType="text/plain",
                            text="Resource echo: AG2User",
                            meta=None,
                        )
                    ]
                    assert loaded_result.contents == expected_result

    @pytest.mark.asyncio
    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set, skipping integration test.")
    @run_for_optional_imports("openai", "openai")
    async def test_with_llm(self, server_params: "StdioServerParameters", credentials_gpt_4o_mini: Credentials) -> None:  # type: ignore[no-any-unimported]
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
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


class MockClientSession:
    def __init__(self, reader, writer):
        pass

    async def __aenter__(self) -> "AsyncMock":
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        return mock_session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_client() -> "MagicMock":
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=(None, None))
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=mock_context_manager)


@pytest.fixture
def session_manager() -> "MCPClientSessionManager":  # type: ignore[no-any-unimported]
    return MCPClientSessionManager()


class TestSseConfig:
    @pytest.fixture
    def sse_config(self) -> "SseConfig":  # type: ignore[no-any-unimported]
        return SseConfig(
            url="http://localhost:8080/sse",
            server_name="test_sse_server",
            headers=None,
            timeout=5.0,
            sse_read_timeout=300.0,
        )

    @pytest.mark.asyncio
    async def test_sse_config_creation(self, sse_config: "SseConfig") -> None:
        assert sse_config.url == "http://localhost:8080/sse"
        assert sse_config.server_name == "test_sse_server"
        assert sse_config.headers is None
        assert sse_config.timeout == DEFAULT_HTTP_REQUEST_TIMEOUT
        assert sse_config.sse_read_timeout == DEFAULT_SSE_EVENT_READ_TIMEOUT

    @pytest.mark.asyncio
    async def test_create_session_mocked(
        self,
        sse_config: "SseConfig",
        mock_client: "MagicMock",
        session_manager: "MCPClientSessionManager",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:  # type: ignore[no-any-unimported]
        monkeypatch.setattr("autogen.mcp.mcp_client.sse_client", mock_client)
        monkeypatch.setattr("autogen.mcp.mcp_client.ClientSession", MockClientSession)

        async with session_manager.open_session(sse_config) as session:
            mock_client.assert_called_once_with(
                sse_config.url,
                sse_config.headers,
                sse_config.timeout,
                sse_config.sse_read_timeout,
            )

            session.initialize.assert_called_once()


class TestMCPStdioConfig:
    @pytest.fixture
    def stdio_config(self) -> "StdioConfig":  # type: ignore[no-any-unimported]
        return StdioConfig(
            command="python3",
            args=["/path/to/server.py"],
            server_name="test_stdio_server",
            environment={"ENV_VAR": "test_value"},
            working_dir="/tmp",
            encoding="utf-8",
            encoding_error_handler="strict",
            session_options={"read_timeout_seconds": 30},
        )

    @pytest.mark.asyncio
    async def test_stdio_config_creation(self, stdio_config: "StdioConfig") -> None:  # type: ignore[no-any-unimported]
        assert stdio_config.command == "python3"
        assert stdio_config.args == ["/path/to/server.py"]
        assert stdio_config.server_name == "test_stdio_server"
        assert stdio_config.environment == {"ENV_VAR": "test_value"}
        assert stdio_config.working_dir == "/tmp"
        assert stdio_config.encoding == "utf-8"
        assert stdio_config.encoding_error_handler == "strict"
        assert stdio_config.transport == "stdio"
        assert stdio_config.session_options == {"read_timeout_seconds": 30}

    @pytest.mark.asyncio
    async def test_create_session(
        self,
        stdio_config: "StdioConfig",
        monkeypatch: pytest.MonkeyPatch,
        mock_client: "MagicMock",
        session_manager: "MCPClientSessionManager",
    ) -> None:  # type: ignore[no-any-unimported]
        monkeypatch.setattr("autogen.mcp.mcp_client.stdio_client", mock_client)
        monkeypatch.setattr("autogen.mcp.mcp_client.ClientSession", MockClientSession)

        async with session_manager.open_session(stdio_config) as session:
            mock_client.assert_called_once_with(
                StdioServerParameters(
                    command=stdio_config.command,
                    args=stdio_config.args,
                    env=stdio_config.environment,
                    encoding=stdio_config.encoding,
                    encoding_error_handler=stdio_config.encoding_error_handler,
                ),
            )

            session.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_session_manager_initialization(session_manager: "MCPClientSessionManager") -> None:  # type: ignore[no-any-unimported]
    assert session_manager.exit_stack is not None
    assert session_manager.sessions == {}

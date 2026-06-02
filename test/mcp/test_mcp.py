# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from pydantic.networks import AnyUrl

from autogen import AssistantAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.mcp.mcp_client import (
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DEFAULT_SSE_EVENT_READ_TIMEOUT,
    MCPClient,
    MCPClientSessionManager,
    ResultSaved,
    SseConfig,
    StdioConfig,
    create_toolkit,
)
from test.credentials import Credentials

with optional_import_block():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import GetPromptResult, Prompt, PromptArgument, ReadResourceResult, TextResourceContents


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
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                            },
                            "required": ["a", "b"],
                            "title": "addArguments",
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
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                            },
                            "required": ["a", "b"],
                            "title": "multiplyArguments",
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
        self, server_params: "StdioServerParameters", credentials_openai_mini: Credentials
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
                llm_config=credentials_openai_mini.llm_config,
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
    async def test_with_llm(self, server_params: "StdioServerParameters", credentials_openai_mini: Credentials) -> None:  # type: ignore[no-any-unimported]
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            # Initialize the connection
            await session.initialize()
            toolkit = await create_toolkit(session=session)

            agent = AssistantAgent(
                name="agent",
                llm_config=credentials_openai_mini.llm_config,
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


class TestMCPPrompts:
    """Tests for MCP prompts support."""

    @pytest.fixture
    def prompts_server_params(self) -> "StdioServerParameters":  # type: ignore[no-any-unimported]
        server_file = Path(__file__).parent / "prompts_server.py"
        return StdioServerParameters(
            command="python3",
            args=[str(server_file)],
        )

    @pytest.mark.asyncio
    async def test_prompts_list_and_get(self, prompts_server_params: "StdioServerParameters") -> None:  # type: ignore[no-any-unimported]
        """Test that the prompts server provides prompts and get_prompt works."""
        import anyio
        from anyio import BrokenResourceError

        try:
            async with (
                stdio_client(prompts_server_params) as (read, write),
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
            ):
                await session.initialize()
                result = await session.list_prompts()
                prompt_names = [p.name for p in result.prompts]
                assert "greeting" in prompt_names
                assert "echo" in prompt_names
                assert "code_review" in prompt_names

                # Verify echo prompt has arguments
                for p in result.prompts:
                    if p.name == "echo":
                        assert p.arguments is not None
                        assert p.arguments[0].name == "message"
                        assert p.arguments[0].required is True

                # Get prompts and verify content
                greeting = await session.get_prompt("greeting")
                assert len(greeting.messages) == 1

                echo = await session.get_prompt("echo", {"message": "test"})
                assert "test" in echo.messages[0].content.text

                review = await session.get_prompt(
                    "code_review", {"code": "print(1)", "language": "python"}
                )
                assert "python" in review.messages[0].content.text
        except BaseExceptionGroup as eg:
            # Filter out BrokenResourceError during cleanup (MCP SDK quirk)
            remaining = [
                e
                for e in eg.exceptions
                if not (
                    isinstance(e, BrokenResourceError)
                    or (
                        hasattr(e, "exceptions")
                        and all(isinstance(x, BrokenResourceError) for x in e.exceptions)
                    )
                )
            ]
            if remaining:
                raise BaseExceptionGroup("remaining errors", remaining)

    @pytest.mark.asyncio
    async def test_convert_prompt(self) -> None:  # type: ignore[no-any-unimported]
        """Test converting an MCP prompt to an AG2 Tool using a mock session."""
        from mcp.types import PromptMessage, TextContent

        mock_session = AsyncMock()
        mock_get_prompt = AsyncMock(
            return_value=GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text="Echoing: hello world"),
                    )
                ]
            )
        )
        mock_session.get_prompt = mock_get_prompt

        mcp_prompt = Prompt(
            name="test_echo",
            description="Echo a message",
            arguments=[
                PromptArgument(name="message", description="The message to echo", required=True),
            ],
        )

        tool = MCPClient.convert_prompt(prompt=mcp_prompt, session=mock_session)

        assert tool.name == "test_echo"
        assert "Echo a message" in tool.description

        # Verify the function schema (from parameters_json_schema)
        func_schema = tool._func_schema  # type: ignore[attr-defined]
        assert func_schema is not None
        assert func_schema["function"]["name"] == "test_echo"
        assert "message" in func_schema["function"]["parameters"]["properties"]

        # Call the tool
        result = await tool(message="hello world")
        mock_get_prompt.assert_called_once_with("test_echo", {"message": "hello world"})

    @pytest.mark.asyncio
    async def test_convert_prompt_no_args(self) -> None:  # type: ignore[no-any-unimported]
        """Test converting a prompt without arguments."""
        mock_session = AsyncMock()

        mcp_prompt = Prompt(
            name="simple_greeting",
            description="A greeting prompt",
        )

        tool = MCPClient.convert_prompt(prompt=mcp_prompt, session=mock_session)

        assert tool.name == "simple_greeting"
        assert "A greeting prompt" in tool.description

        # Should have empty properties since no args
        assert tool._func_schema["function"]["parameters"]["properties"] == {}  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_convert_prompt_invalid_type(self) -> None:  # type: ignore[no-any-unimported]
        """Test that convert_prompt raises ValueError for non-Prompt types."""
        mock_session = AsyncMock()

        with pytest.raises(ValueError, match="Expected an instance of"):
            MCPClient.convert_prompt(prompt="not_a_prompt", session=mock_session)

    @pytest.mark.asyncio
    async def test_prompts_in_toolkit(self) -> None:  # type: ignore[no-any-unimported]
        """Test that prompts are included in the toolkit via load_mcp_toolkit."""
        mock_session = AsyncMock()
        mock_session.list_prompts = AsyncMock(
            return_value=type(
                "ListPromptsResult",
                (),
                {
                    "prompts": [
                        Prompt(name="prompt_a", description="Prompt A"),
                        Prompt(name="prompt_b", description="Prompt B"),
                    ]
                },
            )()
        )

        toolkit = await MCPClient.load_mcp_toolkit(
            session=mock_session,
            use_mcp_tools=False,
            use_mcp_resources=False,
            use_mcp_prompts=True,
            resource_download_folder=None,
        )

        assert len(toolkit) == 2
        assert toolkit.get_tool("prompt_a") is not None
        assert toolkit.get_tool("prompt_b") is not None

    @pytest.mark.asyncio
    async def test_prompts_disabled(self) -> None:  # type: ignore[no-any-unimported]
        """Test that prompts can be excluded from the toolkit."""
        mock_session = AsyncMock()

        toolkit = await MCPClient.load_mcp_toolkit(
            session=mock_session,
            use_mcp_tools=False,
            use_mcp_resources=False,
            use_mcp_prompts=False,
            resource_download_folder=None,
        )

        assert len(toolkit) == 0
        # list_prompts should not have been called
        mock_session.list_prompts.assert_not_called()


    @pytest.mark.asyncio
    async def test_prompts_in_full_toolkit(self, prompts_server_params: "StdioServerParameters") -> None:  # type: ignore[no-any-unimported]
        """Test that prompts are included alongside tools and resources."""
        import anyio
        from anyio import BrokenResourceError

        try:
            async with (
                stdio_client(prompts_server_params) as (read, write),
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
            ):
                await session.initialize()
                # The prompts server has no tools/resources, so with all enabled
                # we should still get only prompts
                toolkit = await create_toolkit(session=session)
                assert len(toolkit) == 3
        except BaseExceptionGroup as eg:
            # Filter out BrokenResourceError during cleanup (MCP SDK quirk)
            remaining = [
                e
                for e in eg.exceptions
                if not (
                    isinstance(e, BrokenResourceError)
                    or (
                        hasattr(e, "exceptions")
                        and all(isinstance(x, BrokenResourceError) for x in e.exceptions)
                    )
                )
            ]
            if remaining:
                raise BaseExceptionGroup("remaining errors", remaining)


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
                    cwd=str(stdio_config.working_dir),
                    encoding=stdio_config.encoding,
                    encoding_error_handler=stdio_config.encoding_error_handler,
                ),
            )

            session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_without_working_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_client: "MagicMock",
        session_manager: "MCPClientSessionManager",
    ) -> None:  # type: ignore[no-any-unimported]
        # When working_dir is not set, cwd must be passed as None (not omitted/defaulted incorrectly).
        stdio_config = StdioConfig(
            command="python3",
            args=["/path/to/server.py"],
            server_name="test_stdio_server",
        )
        monkeypatch.setattr("autogen.mcp.mcp_client.stdio_client", mock_client)
        monkeypatch.setattr("autogen.mcp.mcp_client.ClientSession", MockClientSession)

        async with session_manager.open_session(stdio_config):
            params = mock_client.call_args.args[0]
            assert params.cwd is None


@pytest.mark.asyncio
async def test_session_manager_initialization(session_manager: "MCPClientSessionManager") -> None:  # type: ignore[no-any-unimported]
    assert session_manager.exit_stack is not None
    assert session_manager.sessions == {}

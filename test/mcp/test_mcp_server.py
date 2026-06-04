# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.mcp.mcp_server import create_mcp_server
from autogen.tools import Tool, Toolkit

with optional_import_block():
    from mcp.types import (
        CallToolResult,
        ListToolsResult,
        TextContent,
        Tool as MCPTool,
    )


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------

def _sync_tool_func(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def _async_tool_func(message: str) -> str:
    """Echo the message."""
    return message


def _make_sync_tool() -> Tool:
    return Tool(name="add", description="Add two numbers", func_or_tool=_sync_tool_func)


def _make_async_tool() -> Tool:
    return Tool(name="echo", description="Echo a message", func_or_tool=_async_tool_func)


# ------------------------------------------------------------------
# Synchronous helper to invoke async handlers
# ------------------------------------------------------------------

def _run_handler(coro: Any) -> Any:
    """Run an async handler synchronously for testing."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # If already in an event loop (e.g. pytest-asyncio), we use nest_asyncio
    # or just rely on the caller being async.
    raise RuntimeError("Cannot run async handler inside a running event loop — use pytest.mark.asyncio")


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_returns_correct_schemas() -> None:
    """tools/list should expose one MCP tool per AG2 tool with correct metadata."""
    server = create_mcp_server(
        [Tool(name="foo", description="A foo tool", func_or_tool=_sync_tool_func)],
        server_name="test-server",
    )

    result: ListToolsResult = await server.on_list_tools(None)  # type: ignore[arg-type]
    assert len(result.tools) == 1

    mcp_tool: MCPTool = result.tools[0]
    assert mcp_tool.name == "foo"
    assert mcp_tool.description == "A foo tool"
    assert "type" in mcp_tool.input_schema
    assert mcp_tool.input_schema["type"] == "object"


@pytest.mark.asyncio
async def test_list_tools_from_toolkit() -> None:
    """tools/list should accept a Toolkit as well as a plain list."""
    toolkit = Toolkit()
    toolkit.register_tool(_make_sync_tool())
    toolkit.register_tool(_make_async_tool())

    server = create_mcp_server(toolkit)
    result: ListToolsResult = await server.on_list_tools(None)  # type: ignore[arg-type]
    assert len(result.tools) == 2
    assert {t.name for t in result.tools} == {"add", "echo"}


@pytest.mark.asyncio
async def test_list_tools_empty() -> None:
    """tools/list with an empty tool list returns no MCP tools."""
    server = create_mcp_server([])
    result: ListToolsResult = await server.on_list_tools(None)  # type: ignore[arg-type]
    assert len(result.tools) == 0


@pytest.mark.asyncio
async def test_call_tool_sync() -> None:
    """tools/call should invoke a sync AG2 tool and return the JSON result."""
    tool = _make_sync_tool()
    server = create_mcp_server([tool])

    params = MagicMock()
    params.name = "add"
    params.arguments = {"x": 3, "y": 5}

    result: CallToolResult = await server.on_call_tool(None, params)  # type: ignore[arg-type]
    assert not result.isError
    assert result.content is not None
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "8"


@pytest.mark.asyncio
async def test_call_tool_async() -> None:
    """tools/call should invoke an async AG2 tool and return the result."""
    tool = _make_async_tool()
    server = create_mcp_server([tool])

    params = MagicMock()
    params.name = "echo"
    params.arguments = {"message": "hello"}

    result: CallToolResult = await server.on_call_tool(None, params)  # type: ignore[arg-type]
    assert not result.isError
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].text == "hello"


@pytest.mark.asyncio
async def test_call_tool_unknown() -> None:
    """tools/call with an unknown tool name should return an error."""
    server = create_mcp_server([_make_sync_tool()])

    params = MagicMock()
    params.name = "nonexistent"
    params.arguments = {}

    result: CallToolResult = await server.on_call_tool(None, params)  # type: ignore[arg-type]
    assert result.isError
    assert result.content is not None
    assert "Unknown tool" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_runtime_error() -> None:
    """tools/call should catch runtime exceptions and return them as errors."""

    def _failing_func() -> None:
        raise ValueError("something went wrong")

    server = create_mcp_server([Tool(name="fail", description="Fails", func_or_tool=_failing_func)])

    params = MagicMock()
    params.name = "fail"
    params.arguments = {}

    result: CallToolResult = await server.on_call_tool(None, params)  # type: ignore[arg-type]
    assert result.isError
    assert "something went wrong" in result.content[0].text

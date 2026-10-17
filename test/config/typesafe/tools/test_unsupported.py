# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Jev does not call tools, so every tool, builtin or function, is rejected before a request is sent."""

import pytest

from ag2 import Context
from ag2.config.typesafe.mappers import tool_to_api
from ag2.exceptions import UnsupportedToolError
from ag2.tools import tool
from ag2.tools.builtin.anthropic_bash import AnthropicBashTool
from ag2.tools.builtin.code_execution import CodeExecutionTool
from ag2.tools.builtin.file_search import FileSearchTool
from ag2.tools.builtin.google_maps import GoogleMapsTool
from ag2.tools.builtin.image_generation import ImageGenerationTool
from ag2.tools.builtin.mcp_server import MCPServerTool
from ag2.tools.builtin.memory import MemoryTool
from ag2.tools.builtin.retrieval import RetrievalTool
from ag2.tools.builtin.shell import ShellTool
from ag2.tools.builtin.skills import SkillsTool
from ag2.tools.builtin.web_fetch import WebFetchTool
from ag2.tools.builtin.web_search import WebSearchTool
from ag2.tools.builtin.x_search import XSearchTool
from ag2.tools.types import Tool


@tool
def lookup(order_id: str) -> str:
    return order_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejected",
    [
        lookup,
        WebSearchTool(),
        WebFetchTool(),
        CodeExecutionTool(),
        ShellTool(),
        MemoryTool(),
        ImageGenerationTool(),
        MCPServerTool(server_url="https://mcp.example.com/sse", server_label="example-mcp"),
        SkillsTool("pptx"),
        XSearchTool(),
        RetrievalTool("kb_123"),
        FileSearchTool(vector_store_ids=["vs_1"]),
        GoogleMapsTool(),
        AnthropicBashTool(),
    ],
    ids=lambda t: type(t).__name__,
)
async def test_every_tool_is_rejected(rejected: Tool, context: Context) -> None:
    [schema] = await rejected.schemas(context)

    with pytest.raises(UnsupportedToolError, match="typesafe"):
        tool_to_api(schema)

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from google.genai import types

from autogen import Context
from autogen.config.gemini.mappers import build_tools
from autogen.tools.builtin.code_execution import CodeExecutionTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = CodeExecutionTool()

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [
        types.Tool(code_execution=types.ToolCodeExecution()),
    ]

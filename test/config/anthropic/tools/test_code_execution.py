# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.anthropic.mappers import tool_to_api
from ag2.tools.builtin.code_execution import CodeExecutionTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = CodeExecutionTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "code_execution_20250825", "name": "code_execution"}


@pytest.mark.asyncio
async def test_version_20260521(context: Context) -> None:
    tool = CodeExecutionTool(version="code_execution_20260521")

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "code_execution_20260521", "name": "code_execution"}

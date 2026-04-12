# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.code_execution import CodeExecutionTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = CodeExecutionTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "code_execution_20250825", "name": "code_execution"}

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import Context
from autogen.config.xai.mappers import tool_to_api
from autogen.tools.builtin.code_execution import CodeExecutionTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = CodeExecutionTool()

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.HasField("code_execution")

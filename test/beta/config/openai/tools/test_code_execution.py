# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.openai.mappers import tool_to_responses_api
from autogen.beta.tools.builtin.code_execution import CodeExecutionTool


@pytest.mark.asyncio
async def test_responses_api_defaults(context: Context) -> None:
    tool = CodeExecutionTool()

    [schema] = await tool.schemas(context)

    assert tool_to_responses_api(schema) == {"type": "code_interpreter", "container": {"type": "auto"}}

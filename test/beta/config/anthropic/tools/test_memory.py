# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.memory import MemoryTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = MemoryTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "memory_20250818", "name": "memory"}

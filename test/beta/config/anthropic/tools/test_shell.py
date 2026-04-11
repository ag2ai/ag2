# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.context import Context
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, ShellTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = ShellTool()

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "bash_20250124", "name": "bash"}


@pytest.mark.asyncio
async def test_ignores_environment(context: Context) -> None:
    # Anthropic maps to bash regardless of the environment field
    tool = ShellTool(environment=ContainerAutoEnvironment())

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "bash_20250124", "name": "bash"}

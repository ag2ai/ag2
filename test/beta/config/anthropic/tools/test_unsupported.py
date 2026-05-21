# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Context
from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.image_generation import ImageGenerationTool
from autogen.beta.tools.builtin.shell import ShellTool


@pytest.mark.asyncio
async def test_image_generation(context: Context) -> None:
    tool = ImageGenerationTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


@pytest.mark.asyncio
async def test_shell(context: Context) -> None:
    """ShellTool is unsupported on Anthropic (client-side bash; use LocalShellTool)."""
    tool = ShellTool()

    [schema] = await tool.schemas(context)

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)

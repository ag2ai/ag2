# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.memory import MemoryToolSchema


def test_tool_to_api_memory() -> None:
    schema = MemoryToolSchema()

    result = tool_to_api(schema)

    assert result == {"type": "memory_20250818", "name": "memory"}

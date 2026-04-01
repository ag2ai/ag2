# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.openai.mappers import tool_to_responses_api
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.memory import MemoryToolSchema


def test_tool_to_responses_api_memory_raises() -> None:
    schema = MemoryToolSchema()

    with pytest.raises(UnsupportedToolError):
        tool_to_responses_api(schema)

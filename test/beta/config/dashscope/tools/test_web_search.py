# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.dashscope.mappers import tool_to_api
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema


def test_tool_to_api_web_search_raises() -> None:
    schema = WebSearchToolSchema()

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)

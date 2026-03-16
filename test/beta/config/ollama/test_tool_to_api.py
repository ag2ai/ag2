# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.ollama.mappers import tool_to_api
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema

from .._helpers import make_parameterless_tool, make_tool


def test_tool_to_api() -> None:
    api_tool = tool_to_api(make_tool().schema)

    assert api_tool == {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search documentation by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
            },
        },
    }


def test_tool_to_api_parameterless() -> None:
    api_tool = tool_to_api(make_parameterless_tool().schema)

    assert api_tool["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_tool_to_api_web_search_raises() -> None:
    schema = WebSearchToolSchema()

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)

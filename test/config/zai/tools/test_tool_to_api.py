# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.config.zai.mappers import tool_to_api
from test.config._helpers import make_parameterless_tool, make_tool


def test_tool_to_api() -> None:
    assert tool_to_api(make_tool().schema) == {
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
                "additionalProperties": False,
            },
        },
    }


def test_tool_to_api_parameterless() -> None:
    api_tool = tool_to_api(make_parameterless_tool().schema)

    assert api_tool == {
        "type": "function",
        "function": {
            "name": "ask_human",
            "description": "Ask the human for input.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.openai.mappers import tool_to_api, tool_to_responses_api
from test.beta.config._helpers import make_parameterless_tool, make_tool


def test_tool_to_api() -> None:
    api_tool = tool_to_api(make_tool().schema)

    assert api_tool == {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search documentation by query.",
            "parameters": {
                "additionalProperties": False,
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
        "additionalProperties": False,
    }


def test_tool_to_responses_api_parameterless() -> None:
    api_tool = tool_to_responses_api(make_parameterless_tool().schema)

    assert api_tool["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

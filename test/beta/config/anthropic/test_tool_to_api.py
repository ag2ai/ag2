# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.web_search import UserLocation, WebSearchToolSchema

from .._helpers import make_parameterless_tool, make_tool


def test_tool_to_api() -> None:
    api_tool = tool_to_api(make_tool().schema)

    assert api_tool == {
        "name": "search_docs",
        "description": "Search documentation by query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
        },
    }


def test_tool_to_api_parameterless() -> None:
    api_tool = tool_to_api(make_parameterless_tool().schema)

    assert api_tool["input_schema"] == {
        "type": "object",
        "properties": {},
    }


def test_tool_to_api_web_search_defaults() -> None:
    schema = WebSearchToolSchema()

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_search_20250305",
        "name": "web_search",
    }


def test_tool_to_api_web_search_with_max_uses() -> None:
    schema = WebSearchToolSchema(max_uses=10)

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 10,
    }


def test_tool_to_api_web_search_with_user_location() -> None:
    schema = WebSearchToolSchema(
        user_location=UserLocation(city="London", country="GB", timezone="Europe/London"),
    )

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_search_20250305",
        "name": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "London",
            "country": "GB",
            "timezone": "Europe/London",
        },
    }

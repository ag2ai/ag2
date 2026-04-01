# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.web_search import UserLocation, WebSearchToolSchema


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


def test_tool_to_api_web_search_with_allowed_domains() -> None:
    schema = WebSearchToolSchema(allowed_domains=["example.com", "trusteddomain.org"])

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_search_20250305",
        "name": "web_search",
        "allowed_domains": ["example.com", "trusteddomain.org"],
    }


def test_tool_to_api_web_search_with_blocked_domains() -> None:
    schema = WebSearchToolSchema(blocked_domains=["untrusted.com"])

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_search_20250305",
        "name": "web_search",
        "blocked_domains": ["untrusted.com"],
    }


def test_tool_to_api_web_search_dynamic_filtering() -> None:
    schema = WebSearchToolSchema(web_search_version="web_search_20260209")

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_search_20260209",
        "name": "web_search",
    }


def test_tool_to_api_web_search_dynamic_filtering_with_domains() -> None:
    schema = WebSearchToolSchema(
        max_uses=5,
        allowed_domains=["docs.example.com"],
        blocked_domains=["spam.example.com"],
        web_search_version="web_search_20260209",
    )

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 5,
        "allowed_domains": ["docs.example.com"],
        "blocked_domains": ["spam.example.com"],
    }

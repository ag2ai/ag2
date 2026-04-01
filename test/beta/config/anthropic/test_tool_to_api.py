# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.memory import MemoryToolSchema
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, ShellToolSchema
from autogen.beta.tools.builtin.web_fetch import WebFetchCitations, WebFetchToolSchema
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


def test_tool_to_api_memory() -> None:
    schema = MemoryToolSchema()

    result = tool_to_api(schema)

    assert result == {"type": "memory_20250818", "name": "memory"}


def test_tool_to_api_shell() -> None:
    schema = ShellToolSchema()

    result = tool_to_api(schema)

    assert result == {"type": "bash_20250124", "name": "bash"}


def test_tool_to_api_shell_ignores_environment() -> None:
    # Anthropic maps to bash regardless of the environment field
    schema = ShellToolSchema(environment=ContainerAutoEnvironment())

    result = tool_to_api(schema)

    assert result == {"type": "bash_20250124", "name": "bash"}


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
    schema = WebSearchToolSchema()

    api_tool = tool_to_api(schema, web_search_version="web_search_20260209")

    assert api_tool == {
        "type": "web_search_20260209",
        "name": "web_search",
    }


def test_tool_to_api_web_search_dynamic_filtering_with_domains() -> None:
    schema = WebSearchToolSchema(
        max_uses=5,
        allowed_domains=["docs.example.com"],
        blocked_domains=["spam.example.com"],
    )

    api_tool = tool_to_api(schema, web_search_version="web_search_20260209")

    assert api_tool == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 5,
        "allowed_domains": ["docs.example.com"],
        "blocked_domains": ["spam.example.com"],
    }


def test_tool_to_api_web_fetch_defaults() -> None:
    schema = WebFetchToolSchema()

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
    }


def test_tool_to_api_web_fetch_full() -> None:
    schema = WebFetchToolSchema(
        max_uses=5,
        allowed_domains=["docs.example.com"],
        blocked_domains=["private.example.com"],
        citations=WebFetchCitations(enabled=True),
        max_content_tokens=50000,
    )

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 5,
        "allowed_domains": ["docs.example.com"],
        "blocked_domains": ["private.example.com"],
        "citations": {"enabled": True},
        "max_content_tokens": 50000,
    }


def test_tool_to_api_web_fetch_dynamic_filtering() -> None:
    schema = WebFetchToolSchema()

    api_tool = tool_to_api(schema, web_fetch_version="web_fetch_20260209")

    assert api_tool == {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
    }

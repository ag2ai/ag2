# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.openai.mappers import tool_to_api, tool_to_responses_api
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.memory import MemoryToolSchema
from autogen.beta.tools.builtin.shell import (
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    LocalEnvironment,
    NetworkPolicy,
    ShellToolSchema,
)
from autogen.beta.tools.builtin.web_search import UserLocation, WebSearchToolSchema

from .._helpers import make_parameterless_tool, make_tool


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


def test_tool_to_api_web_search_raises() -> None:
    schema = WebSearchToolSchema()

    with pytest.raises(UnsupportedToolError):
        tool_to_api(schema)


def test_tool_to_responses_api_web_search_defaults() -> None:
    schema = WebSearchToolSchema()
    api_tool = tool_to_responses_api(schema)

    assert api_tool == {"type": "web_search"}


def test_tool_to_responses_api_web_search_with_context_size() -> None:
    schema = WebSearchToolSchema(search_context_size="high")
    api_tool = tool_to_responses_api(schema)

    assert api_tool == {"type": "web_search", "search_context_size": "high"}


def test_tool_to_responses_api_web_search_with_max_uses() -> None:
    schema = WebSearchToolSchema(max_uses=5)
    api_tool = tool_to_responses_api(schema)

    assert api_tool == {"type": "web_search", "max_uses": 5}


def test_tool_to_responses_api_web_search_all_options() -> None:
    schema = WebSearchToolSchema(search_context_size="low", max_uses=3)
    api_tool = tool_to_responses_api(schema)

    assert api_tool == {
        "type": "web_search",
        "search_context_size": "low",
        "max_uses": 3,
    }


def test_tool_to_responses_api_web_search_with_user_location() -> None:
    schema = WebSearchToolSchema(
        user_location=UserLocation(city="San Francisco", region="California", country="US"),
    )
    api_tool = tool_to_responses_api(schema)

    assert api_tool == {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "city": "San Francisco",
            "region": "California",
            "country": "US",
        },
    }


def test_tool_to_responses_api_web_search_with_user_location_partial() -> None:
    schema = WebSearchToolSchema(
        user_location=UserLocation(country="DE", timezone="Europe/Berlin"),
    )
    api_tool = tool_to_responses_api(schema)

    assert api_tool == {
        "type": "web_search",
        "user_location": {
            "type": "approximate",
            "country": "DE",
            "timezone": "Europe/Berlin",
        },
    }


def test_tool_to_responses_api_memory_raises() -> None:
    schema = MemoryToolSchema()

    with pytest.raises(UnsupportedToolError):
        tool_to_responses_api(schema)


def test_tool_to_responses_api_shell_no_environment() -> None:
    schema = ShellToolSchema()

    result = tool_to_responses_api(schema)

    assert result == {"type": "shell"}


def test_tool_to_responses_api_shell_container_auto() -> None:
    schema = ShellToolSchema(environment=ContainerAutoEnvironment())

    result = tool_to_responses_api(schema)

    assert result == {"type": "shell", "environment": {"type": "container_auto"}}


def test_tool_to_responses_api_shell_container_auto_with_network_policy() -> None:
    schema = ShellToolSchema(
        environment=ContainerAutoEnvironment(network_policy=NetworkPolicy(allowed_domains=["example.com"]))
    )

    result = tool_to_responses_api(schema)

    assert result == {
        "type": "shell",
        "environment": {
            "type": "container_auto",
            "network_policy": {"type": "allowlist", "allowed_domains": ["example.com"]},
        },
    }


def test_tool_to_responses_api_shell_container_reference() -> None:
    schema = ShellToolSchema(environment=ContainerReferenceEnvironment(container_id="cntr_xyz"))

    result = tool_to_responses_api(schema)

    assert result == {
        "type": "shell",
        "environment": {"type": "container_reference", "container_id": "cntr_xyz"},
    }


def test_tool_to_responses_api_shell_local() -> None:
    schema = ShellToolSchema(environment=LocalEnvironment())

    result = tool_to_responses_api(schema)

    assert result == {"type": "shell", "environment": {"type": "local"}}


def test_tool_to_responses_api_web_search_with_allowed_domains() -> None:
    schema = WebSearchToolSchema(allowed_domains=["example.com", "docs.example.com"])
    api_tool = tool_to_responses_api(schema)

    assert api_tool == {
        "type": "web_search",
        "filters": {"allowed_domains": ["example.com", "docs.example.com"]},
    }

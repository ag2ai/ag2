# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.anthropic.mappers import tool_to_api
from autogen.beta.tools.builtin.web_fetch import WebFetchToolSchema


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
        citations=True,
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
    schema = WebFetchToolSchema(web_fetch_version="web_fetch_20260209")

    api_tool = tool_to_api(schema)

    assert api_tool == {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
    }

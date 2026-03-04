# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.llms.anthropic import AnthropicClient

from .._helpers import make_tool


def test_tool_to_api() -> None:
    api_tool = AnthropicClient._tool_to_api(make_tool())

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

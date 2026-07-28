# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.config.mistral.mappers import tool_to_api
from test.config._helpers import make_parameterless_tool, make_tool


def test_function_tool() -> None:
    tool = tool_to_api(make_tool().schema)

    assert tool.type == "function"
    assert tool.function.name == "search_docs"
    assert tool.function.description == "Search documentation by query."
    assert tool.function.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
    }


def test_parameterless_tool_is_coerced_to_object_schema() -> None:
    """Mistral requires an object schema; a bare ``{"type": "null"}`` is rejected."""
    tool = tool_to_api(make_parameterless_tool().schema)

    assert tool.function.parameters == {"type": "object", "properties": {}}

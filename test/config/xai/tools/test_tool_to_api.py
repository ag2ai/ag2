# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

from xai_sdk.chat import chat_pb2

from ag2.config.xai.mappers import tool_to_api
from test.config._helpers import make_tool


def test_function_tool_to_api() -> None:
    api_tool = tool_to_api(make_tool().schema)

    assert isinstance(api_tool, chat_pb2.Tool)
    assert api_tool.function.name == "search_docs"
    assert api_tool.function.description == "Search documentation by query."

    params = json.loads(api_tool.function.parameters)
    assert params == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

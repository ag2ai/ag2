# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.tools.builtin.tool_search import ToolSearchToolSchema, assert_tool_search_config
from ag2.tools.final.function_tool import FunctionDefinition, FunctionToolSchema


def _fn(name: str, *, defer: bool = False) -> FunctionToolSchema:
    return FunctionToolSchema(
        function=FunctionDefinition(name=name, description="d", parameters={"type": "object"}),
        defer_loading=defer,
    )


def test_deferred_without_search_tool_raises():
    with pytest.raises(ValueError, match="ToolSearchTool"):
        assert_tool_search_config([_fn("a", defer=True), _fn("b")])


def test_deferred_with_search_tool_ok():
    assert_tool_search_config([ToolSearchToolSchema(), _fn("a", defer=True), _fn("b")])


def test_no_deferred_tools_ok():
    assert_tool_search_config([_fn("a"), _fn("b")])

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

import pytest
from dirty_equals import IsPartialDict
from pydantic import Field

from autogen.beta import Depends, Inject
from autogen.beta.tools import FunctionTool, tool

DEFAULT_SCHEMA = {
    "function": {
        "description": "Tool description.",
        "name": "my_tool",
        "parameters": {
            "properties": {
                "a": {
                    "title": "A",
                    "type": "string",
                },
                "b": {
                    "title": "B",
                    "type": "integer",
                },
            },
            "required": [
                "a",
                "b",
            ],
            "type": "object",
            "additionalProperties": False,
        },
    },
    "type": "function",
}


def test_simple_tool() -> None:
    @tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert my_tool.schema.to_api() == DEFAULT_SCHEMA


def test_tool_schema_ignores_di() -> None:
    def get_dep(b: int) -> int:
        return b

    @tool
    def my_tool(
        a: str,
        depends: Annotated[int, Depends(get_dep)],
        dep: Annotated[str, Inject()],
    ) -> str:
        """Tool description."""
        return ""

    assert my_tool.schema.to_api() == DEFAULT_SCHEMA


def test_override_options() -> None:
    @tool(name="another_name", description="another_description")
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert my_tool.schema.to_api() == {
        "function": IsPartialDict({
            "description": "another_description",
            "name": "another_name",
        }),
        "type": "function",
    }


def test_ensure_tools() -> None:
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert FunctionTool.ensure_tool(my_tool).schema.to_api() == DEFAULT_SCHEMA


def test_ensure_tool_from_tool() -> None:
    @tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert FunctionTool.ensure_tool(my_tool).schema.to_api() == DEFAULT_SCHEMA


def test_option_description() -> None:
    @tool
    def my_tool(
        a: Annotated[str, Field(..., description="Just A")],
        b: int = Field(..., description="Just B"),
    ) -> str:
        """Tool description."""
        return ""

    assert FunctionTool.ensure_tool(my_tool).schema.to_api() == {
        "function": IsPartialDict({
            "parameters": IsPartialDict({
                "properties": {
                    "a": {
                        "title": "A",
                        "description": "Just A",
                        "type": "string",
                    },
                    "b": {
                        "title": "B",
                        "description": "Just B",
                        "type": "integer",
                    },
                }
            }),
        }),
        "type": "function",
    }


def test_empty_args() -> None:
    @tool
    def my_tool() -> str:
        """Tool description."""
        return ""

    assert FunctionTool.ensure_tool(my_tool).schema.to_api() == {
        "function": {
            "description": "Tool description.",
            "name": "my_tool",
            "parameters": {
                "additionalProperties": False,
                "type": "null",
            },
        },
        "type": "function",
    }


@pytest.mark.xfail()
def test_create_dynamic_options() -> None:
    @tool
    def my_tool(a: str | None = None, **kwargs: Any) -> str:
        """Tool description."""
        return ""

    assert FunctionTool.ensure_tool(my_tool).schema.to_api() == {
        "function": {
            "description": "Tool description.",
            "name": "my_tool",
            "parameters": {
                "properties": {
                    "a": {
                        "title": "A",
                        "default": None,
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                    },
                },
                "type": "object",
                "additionalProperties": True,
            },
        },
        "type": "function",
    }

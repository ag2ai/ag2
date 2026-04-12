# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from google.genai import types

from autogen.beta.config.gemini.mappers import build_tools
from autogen.beta.tools.final import FunctionDefinition, FunctionToolSchema
from test.beta.config._helpers import make_tool


def test_tool_to_api() -> None:
    schema = make_tool().schema
    api_tool = build_tools([schema])

    assert api_tool == [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=schema.function.name,
                    description=schema.function.description,
                    parameters=schema.function.parameters,
                )
            ]
        )
    ]


def test_parameterless_tool_empty_dict_gets_object_schema() -> None:
    """Gemini rejects parameters={} — must be normalised to type=object."""
    schema = FunctionToolSchema(
        function=FunctionDefinition(
            name="list_skills",
            description="List installed skills.",
            parameters={},
        )
    )
    api_tool = build_tools([schema])

    assert api_tool == [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="list_skills",
                    description="List installed skills.",
                    parameters={"type": "object", "properties": {}},
                )
            ]
        )
    ]


def test_parameterless_tool_null_type_gets_object_schema() -> None:
    """pydantic/fast_depends generates {'type': 'null'} for no-arg functions — Gemini rejects it."""
    schema = FunctionToolSchema(
        function=FunctionDefinition(
            name="list_skills",
            description="List installed skills.",
            parameters={"type": "null"},
        )
    )
    api_tool = build_tools([schema])

    assert api_tool == [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="list_skills",
                    description="List installed skills.",
                    parameters={"type": "object", "properties": {}},
                )
            ]
        )
    ]

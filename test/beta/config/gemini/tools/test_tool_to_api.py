# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from google.genai import types

from autogen.beta.config.gemini.mappers import build_tools
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

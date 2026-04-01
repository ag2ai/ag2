# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from google.genai import types

from autogen.beta.config.gemini.mappers import build_tools
from autogen.beta.context import Context
from autogen.beta.tools.builtin.web_search import WebSearchTool
from test.beta.config._helpers import make_tool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = WebSearchTool()

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [
        types.Tool(google_search=types.GoogleSearch()),
    ]


@pytest.mark.asyncio
async def test_with_blocked_domains(context: Context) -> None:
    tool = WebSearchTool(blocked_domains=["spam.com", "ads.com"])

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [
        types.Tool(google_search=types.GoogleSearch(exclude_domains=["spam.com", "ads.com"])),
    ]


@pytest.mark.asyncio
async def test_mixed_with_function_tool(context: Context) -> None:
    web_tool = WebSearchTool()
    func_schema = make_tool().schema

    [web_schema] = await web_tool.schemas(context)

    assert build_tools([func_schema, web_schema]) == [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=func_schema.function.name,
                    description=func_schema.function.description,
                    parameters=func_schema.function.parameters,
                )
            ]
        ),
        types.Tool(google_search=types.GoogleSearch()),
    ]

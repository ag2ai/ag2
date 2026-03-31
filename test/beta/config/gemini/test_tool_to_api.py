# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from google.genai import types

from autogen.beta.config.gemini.mappers import build_tools
from autogen.beta.tools.builtin.web_fetch import WebFetchToolSchema
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema

from .._helpers import make_tool


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


def test_build_tools_web_search() -> None:
    schema = WebSearchToolSchema()
    tools = build_tools([schema])

    assert tools == [
        types.Tool(google_search=types.GoogleSearch()),
    ]


def test_build_tools_web_search_with_blocked_domains() -> None:
    schema = WebSearchToolSchema(blocked_domains=["spam.com", "ads.com"])
    tools = build_tools([schema])

    assert tools == [
        types.Tool(google_search=types.GoogleSearch(exclude_domains=["spam.com", "ads.com"])),
    ]


def test_build_tools_web_fetch() -> None:
    schema = WebFetchToolSchema()
    tools = build_tools([schema])

    assert tools == [
        types.Tool(url_context=types.UrlContext()),
    ]


def test_build_tools_mixed() -> None:
    func_schema = make_tool().schema
    web_schema = WebSearchToolSchema()
    tools = build_tools([func_schema, web_schema])

    assert tools == [
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

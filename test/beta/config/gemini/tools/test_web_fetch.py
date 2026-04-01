# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from google.genai import types

from autogen.beta.config.gemini.mappers import build_tools
from autogen.beta.tools.builtin.web_fetch import WebFetchToolSchema


def test_build_tools_web_fetch() -> None:
    schema = WebFetchToolSchema()
    tools = build_tools([schema])

    assert tools == [
        types.Tool(url_context=types.UrlContext()),
    ]

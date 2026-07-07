# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from google.genai import types

from ag2 import Context
from ag2.config.gemini.mappers import build_tool_config, build_tools
from ag2.tools.builtin.google_maps import GoogleMapsTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = GoogleMapsTool()

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [types.Tool(google_maps=types.GoogleMaps())]
    assert build_tool_config([schema]) is None


@pytest.mark.asyncio
async def test_enable_widget(context: Context) -> None:
    tool = GoogleMapsTool(enable_widget=True)

    [schema] = await tool.schemas(context)

    assert build_tools([schema]) == [types.Tool(google_maps=types.GoogleMaps(enable_widget=True))]


@pytest.mark.asyncio
async def test_geo_bias_builds_tool_config(context: Context) -> None:
    tool = GoogleMapsTool(latitude=37.42, longitude=-122.08, language_code="en")

    [schema] = await tool.schemas(context)

    assert build_tool_config([schema]) == types.ToolConfig(
        retrieval_config=types.RetrievalConfig(
            lat_lng=types.LatLng(latitude=37.42, longitude=-122.08),
            language_code="en",
        )
    )

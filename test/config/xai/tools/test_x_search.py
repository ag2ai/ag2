# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import pytest

from ag2 import Context
from ag2.config.xai.mappers import tool_to_api
from ag2.tools.builtin.x_search import XSearchTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = XSearchTool()

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.HasField("x_search")


@pytest.mark.asyncio
async def test_handles_and_flags(context: Context) -> None:
    tool = XSearchTool(
        allowed_x_handles=["xai"],
        excluded_x_handles=["bot"],
        enable_image_understanding=True,
        enable_video_understanding=False,
    )

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert list(api.x_search.allowed_x_handles) == ["xai"]
    assert list(api.x_search.excluded_x_handles) == ["bot"]
    assert api.x_search.enable_image_understanding is True


@pytest.mark.asyncio
async def test_date_range(context: Context) -> None:
    tool = XSearchTool(
        from_date=datetime(2025, 1, 1),
        to_date=datetime(2025, 6, 1),
    )

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    # x_search proto stores timestamps; verify they're populated
    assert api.x_search.from_date.seconds > 0
    assert api.x_search.to_date.seconds > api.x_search.from_date.seconds

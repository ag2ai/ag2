# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from ag2 import Context
from ag2.tools.builtin.tool_search import ToolSearchTool
from ag2.tools.schemas import ToolSchema


@pytest.fixture
def context() -> Context:
    return Context(stream=MagicMock())


def _lookup(order_id: str) -> str:
    return order_id


@pytest_asyncio.fixture
async def tool_search_schema(context: Context) -> ToolSchema:
    [schema, *_] = await ToolSearchTool(_lookup).schemas(context)
    return schema

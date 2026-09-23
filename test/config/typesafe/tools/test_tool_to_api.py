# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.typesafe.mappers import tool_to_api
from ag2.exceptions import UnsupportedToolError
from ag2.tools import tool

pytestmark = pytest.mark.asyncio


@tool
def lookup(order_id: str) -> str:
    return order_id


async def test_function_tool_is_rejected(context: Context) -> None:
    [schema] = await lookup.schemas(context)

    with pytest.raises(UnsupportedToolError, match="typesafe"):
        tool_to_api(schema)

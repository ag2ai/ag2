# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.events import ModelRequest, TextInput
from ag2.exceptions import UnsupportedToolChoiceError
from test.config._helpers import make_tool
from test.config.bedrock._helpers import FakeBedrock


async def _ask(bedrock: FakeBedrock, choice: str, *, with_tools: bool = True) -> None:
    await bedrock.config().create()(
        messages=[ModelRequest([TextInput("hello")])],
        context=Context(stream=MemoryStream()).with_tool_choice(choice),
        tools=[make_tool().schema] if with_tools else [],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
class TestToolChoice:
    @pytest.mark.parametrize(("mode", "expected"), [("auto", {"auto": {}}), ("required", {"any": {}})])
    async def test_mode_is_sent_in_tool_config(self, bedrock: FakeBedrock, mode: str, expected: Any) -> None:
        await _ask(bedrock, mode)

        assert bedrock.body["toolConfig"] == {
            "tools": [IsPartialDict({"toolSpec": IsPartialDict({"name": "search_docs"})})],
            "toolChoice": expected,
        }

    async def test_none_is_unsupported(self, bedrock: FakeBedrock) -> None:
        with pytest.raises(UnsupportedToolChoiceError, match="bedrock"):
            await _ask(bedrock, "none")

        assert bedrock.requests == []

    async def test_no_tools_sends_no_tool_config(self, bedrock: FakeBedrock) -> None:
        await _ask(bedrock, "none", with_tools=False)

        assert "toolConfig" not in bedrock.body

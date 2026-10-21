# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.events import ModelRequest, TextInput
from ag2.exceptions import UnsupportedToolChoiceError
from test.config._helpers import make_tool
from test.config.ollama._helpers import FakeOllama


async def _ask(ollama: FakeOllama, choice: str, *, with_tools: bool = True) -> None:
    await ollama.config().create()(
        messages=[ModelRequest([TextInput("hello")])],
        context=Context(stream=MemoryStream()).with_tool_choice(choice),
        tools=[make_tool().schema] if with_tools else [],
        response_schema=None,
        serializer=SerializerCls,
    )


@pytest.mark.asyncio
class TestToolChoice:
    async def test_auto_keeps_the_tools(self, ollama: FakeOllama) -> None:
        await _ask(ollama, "auto")

        assert ollama.body == IsPartialDict({"tools": [IsPartialDict({"type": "function"})]})
        assert "tool_choice" not in ollama.body

    async def test_none_sends_no_tools(self, ollama: FakeOllama) -> None:
        await _ask(ollama, "none")

        assert not ollama.body.get("tools")
        assert "tool_choice" not in ollama.body

    async def test_required_is_unsupported(self, ollama: FakeOllama) -> None:
        with pytest.raises(UnsupportedToolChoiceError, match="ollama"):
            await _ask(ollama, "required")

        assert ollama.requests == []

    async def test_required_without_tools_is_accepted(self, ollama: FakeOllama) -> None:
        await _ask(ollama, "required", with_tools=False)

        assert not ollama.body.get("tools")

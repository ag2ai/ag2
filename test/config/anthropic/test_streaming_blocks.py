# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The streaming path must record what the non-streaming path records.

``server_tool_use`` input arrives as ``input_json_delta`` chunks. Building the
event at ``content_block_start`` captures ``input={}``, and that empty block is
what gets replayed. Measured live on the same prompt: ``streaming=True`` replayed
``{"input": {}, "name": "web_search"}`` where ``streaming=False`` replayed
``{"input": {"query": "capital of Iceland"}}``.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import ContainerUploadBlock, RedactedThinkingBlock, ServerToolUseBlock

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicClient
from ag2.config.anthropic.events import (
    AnthropicContainerUploadEvent,
    AnthropicRedactedThinkingEvent,
    AnthropicServerToolCallEvent,
)
from ag2.events import BaseEvent
from test.config.anthropic._helpers import FakeStream


def _start(block: Any) -> Any:
    return SimpleNamespace(type="content_block_start", content_block=block)


def _json_delta(partial: str) -> Any:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="input_json_delta", partial_json=partial)
    )


def _stop() -> Any:
    return SimpleNamespace(type="content_block_stop")


async def _stream(events: list[Any]) -> list[BaseEvent]:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    stream = MemoryStream()
    await client._process_stream(FakeStream(events), Context(stream=stream))
    return list(await stream.history.get_events())


@pytest.mark.asyncio
async def test_server_tool_use_input_is_accumulated() -> None:
    block = ServerToolUseBlock(id="srvtoolu_1", name="web_search", input={}, type="server_tool_use")

    events = await _stream([
        _start(block),
        _json_delta('{"query": "capital '),
        _json_delta('of Iceland"}'),
        _stop(),
    ])

    [call] = [e for e in events if isinstance(e, AnthropicServerToolCallEvent)]
    assert call.block.input == {"query": "capital of Iceland"}
    assert call.arguments == '{"query": "capital of Iceland"}'


@pytest.mark.asyncio
async def test_redacted_thinking_is_recorded() -> None:
    block = RedactedThinkingBlock(data="EroBCkYIBBgCKkA...", type="redacted_thinking")

    events = await _stream([_start(block), _stop()])

    assert AnthropicRedactedThinkingEvent(block=block) in events


@pytest.mark.asyncio
async def test_container_upload_is_recorded() -> None:
    block = ContainerUploadBlock(file_id="file_011CQ7", type="container_upload")

    events = await _stream([_start(block), _stop()])

    [event] = [e for e in events if isinstance(e, AnthropicContainerUploadEvent)]
    assert event.file_id == "file_011CQ7"


@pytest.mark.asyncio
async def test_unrecognized_block_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        await _stream([_start(SimpleNamespace(type="something_anthropic_adds_next")), _stop()])

    assert "something_anthropic_adds_next" in caplog.text


@pytest.mark.asyncio
async def test_truncated_server_tool_input_does_not_fail_the_turn(caplog: pytest.LogCaptureFixture) -> None:
    # max_tokens can cut a block mid-JSON. Accumulating the input must not turn
    # that into a JSONDecodeError escaping the whole stream.
    block = ServerToolUseBlock(id="srvtoolu_1", name="web_search", input={}, type="server_tool_use")

    with caplog.at_level("WARNING"):
        events = await _stream([_start(block), _json_delta('{"query": "capi'), _stop()])

    [call] = [e for e in events if isinstance(e, AnthropicServerToolCallEvent)]
    assert call.block.input == {}
    assert "srvtoolu_1" in caplog.text

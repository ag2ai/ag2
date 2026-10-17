# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Block types the response loop used to drop without a trace.

The loop had no ``else``, so an unrecognized block produced no event, no warning
and no history entry. Two stable ``ContentBlock`` arms fell through it.

Replay behaviour was measured against the live API:

- ``redacted_thinking`` is permitted in an assistant turn (a bogus ``data`` is
  rejected as ``Invalid `data` in `redacted_thinking` block``, i.e. the shape is
  accepted) and Anthropic's own docs require it echoed back unchanged.
- ``container_upload`` is refused: ``'container_upload' blocks are not permitted
  within assistant turns``. It must be recorded but never replayed.
"""

from collections.abc import Iterable
from typing import Any

import pytest
from anthropic.types import ContainerUploadBlock, Message, RedactedThinkingBlock, TextBlock, Usage
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicClient
from ag2.config.anthropic.events import AnthropicContainerUploadEvent, AnthropicRedactedThinkingEvent
from ag2.config.anthropic.mappers import convert_messages
from ag2.events import BaseEvent, ModelResponse

REDACTED = RedactedThinkingBlock(data="EroBCkYIBBgCKkA...", type="redacted_thinking")
UPLOAD = ContainerUploadBlock(file_id="file_011CQ7", type="container_upload")


async def _process(content: Iterable[Any]) -> tuple[ModelResponse, list[BaseEvent]]:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    message = Message.model_construct(
        id="m1",
        type="message",
        role="assistant",
        model="claude-sonnet-5",
        content=list(content),
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )
    stream = MemoryStream()
    response = await client._process_response(message, Context(stream=stream))
    return response, list(await stream.history.get_events())


@pytest.mark.asyncio
async def test_redacted_thinking_is_recorded() -> None:
    _, events = await _process([REDACTED, TextBlock(text="ok", type="text")])

    assert AnthropicRedactedThinkingEvent(block=REDACTED) in events


@pytest.mark.asyncio
async def test_container_upload_is_recorded() -> None:
    _, events = await _process([UPLOAD, TextBlock(text="ok", type="text")])

    [event] = [e for e in events if isinstance(e, AnthropicContainerUploadEvent)]
    assert event.file_id == "file_011CQ7"


@pytest.mark.asyncio
async def test_unrecognized_block_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    class FutureBlock:
        type = "something_anthropic_adds_next"

    with caplog.at_level("WARNING"):
        await _process([FutureBlock()])

    assert "something_anthropic_adds_next" in caplog.text


def test_redacted_thinking_replays_verbatim() -> None:
    result = convert_messages([AnthropicRedactedThinkingEvent(block=REDACTED)], SerializerCls)

    assert result == [{"role": "assistant", "content": [{"data": "EroBCkYIBBgCKkA...", "type": "redacted_thinking"}]}]


def test_container_upload_is_never_replayed() -> None:
    # The API refuses it inside an assistant turn; replaying would 400 the next request.
    assert convert_messages([AnthropicContainerUploadEvent(block=UPLOAD)], SerializerCls) == []

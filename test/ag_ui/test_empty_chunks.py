# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Empty model output, and the AG-UI frames it must not turn into.

The protocol forbids empty deltas (`MinLen(1)` on `TextMessageContentEvent.delta`
and `TextMessageChunkEvent.delta`), so a provider that streams an empty chunk —
or returns an empty message — must leave no text frame behind at all, not an
empty one. Streaming a chunk the provider never sends a body for is ordinary
provider behaviour, so it is scripted as ordinary provider behaviour: the empty
chunk goes on the stream, and what the subscriber does with it is the test.
"""

from typing import Any

import pytest
from ag_ui.core import UserMessage
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ModelMessage, ModelMessageChunk, ModelResponse
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, every, run_input, types_of

pytestmark = pytest.mark.asyncio


async def frames_of(config: TestConfig) -> list[dict[str, Any]]:
    """Every AG-UI frame one run of `config` emits."""
    return await dispatch_run(
        AGUIStream(Agent("test_agent", config=config)), run_input(UserMessage(id="m1", content="hi"))
    )


class TestEmptyModelMessageChunk:
    async def test_an_empty_first_chunk_does_not_open_a_text_message(self) -> None:
        """The message is opened by the chunk that has something to say, not before."""
        frames = await frames_of(TestConfig(ModelMessageChunk(""), ModelMessageChunk("Hello"), "Hello"))

        assert types_of(frames) == [
            "RUN_STARTED",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "RUN_FINISHED",
        ]
        assert every(frames, "TEXT_MESSAGE_CONTENT") == [IsPartialDict({"delta": "Hello"})]

    async def test_an_empty_chunk_between_real_ones_is_dropped(self) -> None:
        frames = await frames_of(
            TestConfig(
                ModelMessageChunk("Hello"),
                ModelMessageChunk(""),
                ModelMessageChunk(" world"),
                "Hello world",
            )
        )

        assert [f["delta"] for f in every(frames, "TEXT_MESSAGE_CONTENT")] == ["Hello", " world"]


class TestEmptyModelMessage:
    async def test_an_empty_non_streaming_message_emits_no_text_frame(self) -> None:
        empty = ModelMessage("")

        frames = await frames_of(TestConfig(empty, ModelResponse(empty)))

        assert types_of(frames) == ["RUN_STARTED", "RUN_FINISHED"]

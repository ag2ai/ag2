# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Model reasoning on its way out, as the AG-UI reasoning session a client renders.

A provider streams reasoning as ordinary events on the agent's stream, so it is
scripted that way here and the subscriber's framing is what is asserted: one
session, opened once, closed before the answer begins, and never opened for a
thought with nothing in it.

Reasoning arriving *inbound*, in an AG-UI history, is the mapper's job and is
tested there — see `TestReasoningMessages` in `test_mapper.py`.
"""

from base64 import b64encode
from typing import Any

import pytest
from ag_ui.core import UserMessage
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.config.gemini.events import GeminiToolCallEvent
from ag2.events import ModelReasoning
from ag2.testing import TestConfig, Turn
from test.ag_ui.harness import dispatch_run, every, only, run_input, types_of, weather_tool

pytestmark = pytest.mark.asyncio

REASONING_FRAMES = (
    "REASONING_START",
    "REASONING_MESSAGE_START",
    "REASONING_MESSAGE_CONTENT",
    "REASONING_MESSAGE_END",
    "REASONING_END",
)


async def frames_of(*script: Turn) -> list[dict[str, Any]]:
    """Every AG-UI frame one run of `script` emits."""
    agent = Agent("test_agent", config=TestConfig(*script))
    return await dispatch_run(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")))


async def test_reasoning_chunks_emit_one_session_under_one_message_id() -> None:
    frames = await frames_of(ModelReasoning("Thinking"), ModelReasoning(" more"), "Done")

    message_id = only(frames, "REASONING_START")["messageId"]
    assert only(frames, "REASONING_MESSAGE_START") == IsPartialDict({
        "messageId": message_id,
        "role": "reasoning",
    })
    assert every(frames, "REASONING_MESSAGE_CONTENT") == [
        IsPartialDict({"messageId": message_id, "delta": "Thinking"}),
        IsPartialDict({"messageId": message_id, "delta": " more"}),
    ]
    assert only(frames, "REASONING_MESSAGE_END") == IsPartialDict({"messageId": message_id})
    assert only(frames, "REASONING_END") == IsPartialDict({"messageId": message_id})


async def test_the_session_closes_before_the_answer_begins() -> None:
    """A client renders reasoning and answer in separate places; they must not interleave."""
    frames = await frames_of(ModelReasoning("thinking"), "Final answer")

    reasoning_or_text = [t for t in types_of(frames) if t.startswith(("REASONING_", "TEXT_MESSAGE_"))]

    assert reasoning_or_text.index("REASONING_END") < next(
        i for i, t in enumerate(reasoning_or_text) if t.startswith("TEXT_MESSAGE_")
    )


async def test_an_empty_reasoning_chunk_is_skipped() -> None:
    frames = await frames_of(ModelReasoning(""), ModelReasoning("real thought"), "Done")

    assert every(frames, "REASONING_MESSAGE_CONTENT") == [IsPartialDict({"delta": "real thought"})]


async def test_a_run_that_did_no_reasoning_opens_no_session() -> None:
    frames = await frames_of("Hello")

    assert [t for t in types_of(frames) if t in REASONING_FRAMES] == []


async def test_gemini_tool_signature_is_sent_for_client_replay() -> None:
    signature = b"gemini-signature"
    agent = Agent(
        "test_agent",
        config=TestConfig(
            GeminiToolCallEvent(id="call-1", name="get_weather", arguments="{}", thought_signature=signature)
        ),
    )

    frames = await dispatch_run(
        AGUIStream(agent), run_input(UserMessage(id="m1", content="weather?"), tools=[weather_tool()])
    )

    assert only(frames, "REASONING_ENCRYPTED_VALUE") == IsPartialDict({
        "subtype": "tool-call",
        "entityId": "call-1",
        "encryptedValue": b64encode(signature).decode(),
    })

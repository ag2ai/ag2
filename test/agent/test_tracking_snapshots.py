# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent, DataInput, ImageInput, MemoryStream, TextInput
from ag2.events import BaseEvent, Field, ModelMessage, ModelRequest, ModelResponse, ProviderReplay
from ag2.middleware import RetryMiddleware
from ag2.testing import TestConfig, TrackingConfig


@pytest.mark.asyncio
async def test_tracking_records_complete_inputs_for_each_continuation() -> None:
    config = TrackingConfig(TestConfig("first answer", "second answer"))
    agent = Agent("tracking", config=config, prompt="Be helpful")

    reply = await agent.ask("first question")
    await reply.ask("second question")

    first = ModelRequest([TextInput("first question")])
    second = ModelRequest([TextInput("second question")])
    assert [call.messages for call in config.calls] == [
        (first,),
        (first, ModelResponse(ModelMessage("first answer")), second),
    ]
    assert [call.prompt for call in config.calls] == [("Be helpful",), ("Be helpful",)]
    config.mock.assert_any_call(first)
    config.mock.assert_called_with(second)


@pytest.mark.asyncio
async def test_tracking_detaches_mutable_input_parts_and_metadata() -> None:
    config = TrackingConfig(TestConfig("first answer", "second answer"))
    agent = Agent("tracking", config=config)
    data = {"steps": ["queued"]}
    text = TextInput("progress", metadata={"tags": ["original"]})
    image = ImageInput(data=b"image bytes", media_type="image/png")
    image.vendor_metadata["labels"] = ["original"]

    reply = await agent.ask(text, DataInput(data), image)
    data["steps"].append("done")
    text.content = "changed"
    text.metadata["tags"].append("changed")
    image.vendor_metadata["labels"].append("changed")
    await reply.ask("continue")

    expected_image = ImageInput(data=b"image bytes", media_type="image/png")
    expected_image.vendor_metadata["labels"] = ["original"]
    assert config.calls[0].messages == (
        ModelRequest([
            TextInput("progress", metadata={"tags": ["original"]}),
            DataInput({"steps": ["queued"]}),
            expected_image,
        ]),
    )
    assert config.calls[0].messages != config.calls[1].messages[:1]


@pytest.mark.asyncio
async def test_tracking_records_failed_attempts_before_retry_and_success() -> None:
    config = TrackingConfig(TestConfig(TimeoutError("retry"), "recovered"))
    agent = Agent(
        "tracking",
        config=config,
        middleware=[RetryMiddleware(max_retries=1, retry_on=(TimeoutError,))],
    )

    reply = await agent.ask("question")

    assert reply.body == "recovered"
    assert [call.messages for call in config.calls] == [
        (ModelRequest([TextInput("question")]),),
        (ModelRequest([TextInput("question")]),),
    ]
    assert config.mock.call_count == 2


@pytest.mark.asyncio
async def test_tracking_records_final_failed_attempt() -> None:
    config = TrackingConfig(TestConfig(ValueError("provider rejected request")))
    agent = Agent("tracking", config=config)

    with pytest.raises(ValueError, match="provider rejected request"):
        await agent.ask("question")

    assert [call.messages for call in config.calls] == [(ModelRequest([TextInput("question")]),)]


class ReplayEvent(BaseEvent, ProviderReplay):
    __replay_role__ = "turn"

    payload: dict = Field(kw_only=False)


@pytest.mark.asyncio
async def test_tracking_preserves_provider_replay_event_type_and_payload() -> None:
    config = TrackingConfig(TestConfig("answer"))
    agent = Agent("tracking", config=config)
    stream = MemoryStream()
    replay = ReplayEvent({"native": ["original"]})
    await stream.history.replace([replay])

    await agent.ask("question", stream=stream)
    replay.payload["native"].append("changed")

    assert config.calls[0].messages == (
        ReplayEvent({"native": ["original"]}),
        ModelRequest([TextInput("question")]),
    )

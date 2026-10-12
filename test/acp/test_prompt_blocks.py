# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import acp
import pytest
from acp import schema

from ag2 import Agent
from ag2.acp.testing import duplex_acp_config
from ag2.events import AudioInput, DocumentInput, ImageInput, TextInput
from ag2.stream import MemoryStream


class RecordingAgent:
    def __init__(self, prompts: list[list[Any]], capabilities: schema.PromptCapabilities | None = None) -> None:
        self.prompts = prompts
        self._capabilities = capabilities

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        agent_capabilities = (
            schema.AgentCapabilities(prompt_capabilities=self._capabilities) if self._capabilities else None
        )
        return schema.InitializeResponse(protocol_version=acp.PROTOCOL_VERSION, agent_capabilities=agent_capabilities)

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        return schema.NewSessionResponse(session_id="recording-session")

    async def prompt(self, *, prompt: list[Any], **kwargs: Any) -> schema.PromptResponse:
        self.prompts.append(prompt)
        return schema.PromptResponse(stop_reason="end_turn")


@pytest.mark.asyncio
async def test_the_parts_of_one_message_arrive_as_separate_blocks() -> None:
    prompts: list[list[Any]] = []
    cfg = duplex_acp_config(lambda _conn: RecordingAgent(prompts))

    try:
        await Agent("acp", config=cfg).ask(TextInput("first"), TextInput("second"))
    finally:
        await cfg.aclose()

    assert prompts == [[acp.text_block("first"), acp.text_block("second")]]


async def _prompt_sent(capabilities: schema.PromptCapabilities | None, *inputs: Any) -> list[Any]:
    prompts: list[list[Any]] = []
    cfg = duplex_acp_config(lambda _conn: RecordingAgent(prompts, capabilities))

    try:
        await Agent("acp", config=cfg).ask(*inputs)
    finally:
        await cfg.aclose()

    [prompt] = prompts
    return prompt


@pytest.fixture
def all_capabilities() -> schema.PromptCapabilities:
    return schema.PromptCapabilities(image=True, audio=True, embedded_context=True)


@pytest.mark.asyncio
async def test_media_the_agent_accepts_is_sent_as_its_own_block(all_capabilities: schema.PromptCapabilities) -> None:
    prompt = await _prompt_sent(
        all_capabilities,
        TextInput("look"),
        ImageInput(data=b"img", media_type="image/png"),
        AudioInput(data=b"aud", media_type="audio/wav"),
        DocumentInput(data=b"doc", media_type="application/pdf"),
    )

    assert prompt == [
        acp.text_block("look"),
        acp.image_block("aW1n", "image/png"),
        acp.audio_block("YXVk", "audio/wav"),
        acp.resource_block(
            schema.BlobResourceContents(blob="ZG9j", mime_type="application/pdf", uri="attachment:document")
        ),
    ]


@pytest.mark.asyncio
async def test_media_the_agent_does_not_accept_is_named_not_dropped() -> None:
    prompt = await _prompt_sent(
        None,
        ImageInput(data=b"img", media_type="image/png"),
        AudioInput(data=b"aud", media_type="audio/wav"),
        DocumentInput(data=b"doc", media_type="application/pdf"),
    )

    assert prompt == [acp.text_block("[image]"), acp.text_block("[audio]"), acp.text_block("[document]")]


@pytest.mark.asyncio
async def test_a_url_goes_as_a_resource_link_whatever_the_agent_advertises() -> None:
    url = "https://example.com/photo.png"

    prompt = await _prompt_sent(None, ImageInput(url))

    assert prompt == [acp.resource_link_block(name=url, uri=url)]


@pytest.mark.asyncio
async def test_a_later_turn_sends_only_the_input_not_yet_sent() -> None:
    prompts: list[list[Any]] = []
    cfg = duplex_acp_config(lambda _conn: RecordingAgent(prompts))
    agent = Agent("acp", config=cfg)
    stream = MemoryStream()

    try:
        await agent.ask("first", stream=stream)
        await agent.ask("second", stream=stream)
    finally:
        await cfg.aclose()

    assert prompts == [[acp.text_block("first")], [acp.text_block("second")]]


@pytest.mark.asyncio
async def test_a_turn_with_no_input_still_sends_an_empty_text_block() -> None:
    assert await _prompt_sent(None) == [acp.text_block("")]

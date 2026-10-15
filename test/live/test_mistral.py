# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

pytest.importorskip("mistralai")

from mistralai.client import Mistral
from mistralai.client.httpclient import httpx  # whichever of httpx / httpx2 the installed SDK is built on

from ag2 import Agent, MemoryStream, events, testing
from ag2.context import ConversationContext
from ag2.live import MistralTranscriber
from ag2.live.stt import VoiceInput

pytestmark = [pytest.mark.asyncio, pytest.mark.mistral]

VOICE = VoiceInput(b"\x00\x01" * 100, frame_rate=16000, channels=1)

RESPONSE = {
    "model": "voxtral-mini-latest",
    "text": "hello there",
    "language": "en",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class Transport:
    """Records raw multipart request bodies and answers with a canned transcription."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def client(self) -> Mistral:
        return Mistral(
            api_key="test-key",
            async_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=RESPONSE)

    @property
    def body(self) -> bytes:
        [request] = self.requests
        return request.content


@pytest.fixture
def transport() -> Transport:
    return Transport()


class TestMistralTranscriber:
    async def test_uploads_wav_and_emits_the_complete_transcript(self, transport: Transport) -> None:
        stream = MemoryStream()
        seen: list[events.TranscriptionCompletedEvent] = []

        async def collect(event: events.TranscriptionCompletedEvent) -> None:
            seen.append(event)

        stream.where(events.TranscriptionCompletedEvent).subscribe(collect)
        text = await MistralTranscriber(client=transport.client()).transcribe(VOICE, ConversationContext(stream=stream))
        await asyncio.sleep(0.01)

        assert text == "hello there"
        assert seen == [events.TranscriptionCompletedEvent("hello there")]
        assert transport.requests[0].url.path.endswith("/audio/transcriptions")
        assert b"voxtral-mini-latest" in transport.body
        assert b"RIFF" in transport.body

    async def test_omits_language_and_context_bias_when_unset(self, transport: Transport) -> None:
        await MistralTranscriber(client=transport.client()).transcribe(
            VOICE, ConversationContext(stream=MemoryStream())
        )

        assert b'name="language"' not in transport.body
        assert b"context_bias" not in transport.body

    async def test_forwards_explicit_options(self, transport: Transport) -> None:
        await MistralTranscriber(
            "voxtral-mini-latest",
            language="fr",
            context_bias=["AG2", "Voxtral"],
            diarize=True,
            client=transport.client(),
        ).transcribe(VOICE, ConversationContext(stream=MemoryStream()))

        body = transport.body
        assert b'name="language"' in body and b"fr" in body
        assert b"AG2" in body and b"Voxtral" in body
        assert b'name="diarize"' in body
        assert b'name="timestamp_granularities' in body and b"segment" in body

    async def test_builds_a_client_from_api_key(self) -> None:
        config = MistralTranscriber(api_key="explicit-key")

        assert config._client.sdk_configuration.security.api_key == "explicit-key"

    async def test_pipes_the_transcript_into_an_agent(self, transport: Transport) -> None:
        tracking = testing.TrackingConfig(testing.TestConfig("Hi!"))
        reply = await MistralTranscriber(client=transport.client()).pipe(Agent("assistant", config=tracking)).ask(VOICE)

        assert reply.body == "Hi!"
        assert tracking.mock.call_args.args[0].parts[0].content == "hello there"

    async def test_follow_up_turn_transcribes_on_the_same_conversation(self, transport: Transport) -> None:
        agent = Agent("assistant", config=testing.TestConfig("one", "two"))
        pipeline = MistralTranscriber(client=transport.client()).pipe(agent)

        first = await pipeline.ask(VOICE)
        second = await first.ask(VOICE)

        assert (first.body, second.body) == ("one", "two")
        assert len(transport.requests) == 2

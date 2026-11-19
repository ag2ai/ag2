# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path

import pytest

pytest.importorskip("google.genai")
pytest.importorskip("websockets")

import websockets.asyncio.server
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from google.genai import Client
from google.genai import types as gtypes

from ag2 import MemoryStream
from ag2.events import ModelMessageChunk, ModelResponse, TranscriptionChunkEvent, TranscriptionCompletedEvent
from ag2.live import LiveAgent, gemini

HOST = "127.0.0.1"
NATIVE_AUDIO_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"


@dataclass(frozen=True, slots=True)
class Certificate:
    cert_file: Path
    key_file: Path


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> Certificate:
    """Generate a self-signed certificate for the SDK's local WSS connection."""
    directory = tmp_path_factory.mktemp("tls")
    cert_file = directory / "cert.pem"
    key_file = directory / "key.pem"
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HOST)])
    now = datetime.now(UTC)
    cert = (
        x509
        .CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address(HOST))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return Certificate(cert_file, key_file)


class LiveServer:
    """A local Gemini Live endpoint that sends scripted server content after setup."""

    def __init__(self, certificate: Certificate, *turn: gtypes.LiveServerContent) -> None:
        self._certificate = certificate
        self._turn = turn
        self._server: websockets.asyncio.server.Server | None = None
        self._connection: websockets.asyncio.server.ServerConnection | None = None

    async def _handle(self, connection: websockets.asyncio.server.ServerConnection) -> None:
        await connection.recv()  # the session setup
        self._connection = connection
        await connection.send(_wire(gtypes.LiveServerMessage(setup_complete=gtypes.LiveServerSetupComplete())))
        for content in self._turn:
            await self.send(content)
        await connection.wait_closed()

    async def send(self, content: gtypes.LiveServerContent) -> None:
        assert self._connection is not None
        await self._connection.send(_wire(gtypes.LiveServerMessage(server_content=content)))

    async def __aenter__(self) -> "LiveServer":
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self._certificate.cert_file, self._certificate.key_file)
        self._server = await websockets.asyncio.server.serve(self._handle, HOST, 0, ssl=context)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    def client(self) -> Client:
        assert self._server is not None
        port = self._server.sockets[0].getsockname()[1]
        return Client(
            api_key="test-key-not-used",
            http_options=gtypes.HttpOptions(
                base_url=f"https://{HOST}:{port}",
                async_client_args={"ssl": ssl.create_default_context(cafile=self._certificate.cert_file)},
            ),
        )


def _wire(message: gtypes.LiveServerMessage) -> str:
    return message.model_dump_json(by_alias=True, exclude_none=True)


def heard(text: str, *, finished: bool | None = None) -> gtypes.LiveServerContent:
    return gtypes.LiveServerContent(input_transcription=gtypes.Transcription(text=text, finished=finished))


def said(text: str) -> gtypes.LiveServerContent:
    return gtypes.LiveServerContent(output_transcription=gtypes.Transcription(text=text))


def _live_agent(server: LiveServer, stream: MemoryStream, model: str) -> LiveAgent:
    return LiveAgent(
        "assistant",
        config=gemini.RealTimeConfig(
            model,
            input=gemini.InputConfig(transcribe=True),
            client=server.client(),
        ),
        stream=stream,
    )


async def _completed_transcripts(
    certificate: Certificate,
    *content: gtypes.LiveServerContent,
    model: str = NATIVE_AUDIO_MODEL,
) -> list[str]:
    async with LiveServer(certificate, *content) as server:
        stream = MemoryStream()
        agent = _live_agent(server, stream, model)
        transcripts: list[str] = []
        stream.where(TranscriptionCompletedEvent).subscribe(lambda event: transcripts.append(event.content))

        async with asyncio.timeout(5), stream.get(ModelResponse) as turn_done, agent.run():
            await turn_done

    return transcripts


@pytest.mark.asyncio
class TestGeminiInputTranscription:
    async def test_fragments_complete_when_the_generation_completes(self, certificate: Certificate) -> None:
        # Before Gemini 3, the Gemini API can send fragments without marking the last one finished.
        transcripts = await _completed_transcripts(
            certificate,
            heard(" What is"),
            heard(" the weather"),
            heard(" today?"),
            said("Sunny."),
            gtypes.LiveServerContent(generation_complete=True),
            heard(" Thanks."),
            gtypes.LiveServerContent(turn_complete=True),
        )

        assert transcripts == ["What is the weather today?", "Thanks."]

    async def test_fragments_complete_when_the_user_interrupts(self, certificate: Certificate) -> None:
        transcripts = await _completed_transcripts(
            certificate,
            heard("Tell me a story"),
            said("Once upon"),
            heard(" Stop."),
            gtypes.LiveServerContent(interrupted=True),
            gtypes.LiveServerContent(turn_complete=True),
        )

        assert transcripts == ["Tell me a story Stop."]

    async def test_fragments_complete_on_finished(self, certificate: Certificate) -> None:
        # On Vertex AI, the last fragment can carry finished before the turn completes.
        transcripts = await _completed_transcripts(
            certificate,
            heard("What is"),
            heard(" the weather", finished=True),
            said("Sunny."),
            gtypes.LiveServerContent(turn_complete=True),
        )

        assert transcripts == ["What is the weather"]

    async def test_gemini_3_transcription_is_complete_as_it_arrives(self, certificate: Certificate) -> None:
        async with LiveServer(
            certificate,
            heard("What is the weather today?"),
            said("Sunny."),
            gtypes.LiveServerContent(turn_complete=True),
        ) as server:
            stream = MemoryStream()
            agent = _live_agent(server, stream, "gemini-3.8-live")
            transcription: list[TranscriptionChunkEvent | TranscriptionCompletedEvent] = []
            stream.where(TranscriptionChunkEvent | TranscriptionCompletedEvent).subscribe(
                lambda event: transcription.append(event)
            )

            async with asyncio.timeout(5), stream.get(ModelResponse) as turn_done, agent.run():
                await turn_done

        assert transcription == [
            TranscriptionChunkEvent("What is the weather today?"),
            TranscriptionCompletedEvent("What is the weather today?"),
        ]

    async def test_gemini_3_model_resource_name_is_complete_as_it_arrives(self, certificate: Certificate) -> None:
        transcripts = await _completed_transcripts(
            certificate,
            heard("What is"),
            heard(" the weather today?"),
            gtypes.LiveServerContent(turn_complete=True),
            model="models/gemini-3.8-live",
        )

        assert transcripts == ["What is", "the weather today?"]

    async def test_gemini_3_5_translate_transcription_is_complete_as_it_arrives(self, certificate: Certificate) -> None:
        first = "What is the weather like in Paris today?"
        second = "Thanks. And what about tomorrow?"
        async with LiveServer(certificate) as server:
            stream = MemoryStream()
            agent = _live_agent(server, stream, "gemini-3.5-live-translate-preview")
            transcription: list[TranscriptionChunkEvent | TranscriptionCompletedEvent] = []
            stream.where(TranscriptionChunkEvent | TranscriptionCompletedEvent).subscribe(
                lambda event: transcription.append(event)
            )

            async with asyncio.timeout(5), agent.run():
                for text in (first, second):
                    # The API sends an empty transcription after each utterance.
                    # Wait for completion before sending it.
                    async with stream.get(TranscriptionCompletedEvent) as completed:
                        await server.send(heard(text))
                        await completed
                    await server.send(gtypes.LiveServerContent(input_transcription=gtypes.Transcription()))

                # The next output chunk confirms the last empty transcription was processed.
                async with stream.get(ModelMessageChunk) as processed:
                    await server.send(said("Translated speech."))
                    await processed

        assert transcription == [
            TranscriptionChunkEvent(first),
            TranscriptionCompletedEvent(first),
            TranscriptionChunkEvent(second),
            TranscriptionCompletedEvent(second),
        ]

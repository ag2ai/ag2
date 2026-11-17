# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import ssl
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
from ag2.events import ModelResponse, TranscriptionChunkEvent, TranscriptionCompletedEvent
from ag2.live import LiveAgent, gemini

pytestmark = pytest.mark.asyncio

HOST = "127.0.0.1"
NATIVE_AUDIO_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"


class Certificate:
    """A self-signed certificate for 127.0.0.1: the SDK only dials `wss://`."""

    def __init__(self, directory: Path) -> None:
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HOST)])
        cert = (
            x509
            .CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC) - timedelta(minutes=5))
            .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address(HOST))]), critical=False)
            .sign(key, hashes.SHA256())
        )
        self.cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        self.cert_file = directory / "cert.pem"
        self.key_file = directory / "key.pem"
        self.cert_file.write_bytes(self.cert_pem)
        self.key_file.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> Certificate:
    return Certificate(tmp_path_factory.mktemp("tls"))


class LiveServer:
    """A local Gemini Live endpoint: completes the setup, then sends one turn of server content."""

    def __init__(self, certificate: Certificate, *turn: gtypes.LiveServerContent) -> None:
        self._certificate = certificate
        self._turn = turn
        self._server: websockets.asyncio.server.Server | None = None

    async def _handle(self, connection: websockets.asyncio.server.ServerConnection) -> None:
        await connection.recv()  # the session setup
        await connection.send(_wire(gtypes.LiveServerMessage(setup_complete=gtypes.LiveServerSetupComplete())))
        for content in self._turn:
            await connection.send(_wire(gtypes.LiveServerMessage(server_content=content)))
        await connection.wait_closed()

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
                async_client_args={"ssl": ssl.create_default_context(cadata=self._certificate.cert_pem.decode())},
            ),
        )


def _wire(message: gtypes.LiveServerMessage) -> str:
    return message.model_dump_json(by_alias=True, exclude_none=True)


def heard(text: str, *, finished: bool | None = None) -> gtypes.LiveServerContent:
    return gtypes.LiveServerContent(input_transcription=gtypes.Transcription(text=text, finished=finished))


def said(text: str) -> gtypes.LiveServerContent:
    return gtypes.LiveServerContent(output_transcription=gtypes.Transcription(text=text))


async def test_fragments_complete_when_the_generation_completes(certificate: Certificate) -> None:
    # Before Gemini 3, the user's words arrive as fragments, and the Gemini API never
    # marks one `finished`: what was heard so far completes with the generation.
    async with LiveServer(
        certificate,
        heard(" What is"),
        heard(" the weather"),
        heard(" today?"),
        said("Sunny."),
        gtypes.LiveServerContent(generation_complete=True),
        heard(" Thanks."),
        gtypes.LiveServerContent(turn_complete=True),
    ) as server:
        stream = MemoryStream()
        agent = LiveAgent(
            "assistant",
            config=gemini.RealTimeConfig(
                NATIVE_AUDIO_MODEL,
                input=gemini.InputConfig(transcribe=True),
                client=server.client(),
            ),
            stream=stream,
        )
        transcripts: list[str] = []
        stream.where(TranscriptionCompletedEvent).subscribe(lambda event: transcripts.append(event.content))

        async with stream.get(ModelResponse) as turn_done, agent.run():
            await turn_done

    assert transcripts == ["What is the weather today?", "Thanks."]


async def test_fragments_complete_when_the_user_interrupts(certificate: Certificate) -> None:
    async with LiveServer(
        certificate,
        heard("Tell me a story"),
        said("Once upon"),
        heard(" Stop."),
        gtypes.LiveServerContent(interrupted=True),
        gtypes.LiveServerContent(turn_complete=True),
    ) as server:
        stream = MemoryStream()
        agent = LiveAgent(
            "assistant",
            config=gemini.RealTimeConfig(
                NATIVE_AUDIO_MODEL,
                input=gemini.InputConfig(transcribe=True),
                client=server.client(),
            ),
            stream=stream,
        )
        transcripts: list[str] = []
        stream.where(TranscriptionCompletedEvent).subscribe(lambda event: transcripts.append(event.content))

        async with stream.get(ModelResponse) as turn_done, agent.run():
            await turn_done

    assert transcripts == ["Tell me a story Stop."]


async def test_fragments_complete_on_finished(certificate: Certificate) -> None:
    # On Vertex AI the last fragment can carry `finished`; the turn completing
    # afterwards does not send the transcript a second time.
    async with LiveServer(
        certificate,
        heard("What is"),
        heard(" the weather", finished=True),
        said("Sunny."),
        gtypes.LiveServerContent(turn_complete=True),
    ) as server:
        stream = MemoryStream()
        agent = LiveAgent(
            "assistant",
            config=gemini.RealTimeConfig(
                NATIVE_AUDIO_MODEL,
                input=gemini.InputConfig(transcribe=True),
                client=server.client(),
            ),
            stream=stream,
        )
        transcripts: list[str] = []
        stream.where(TranscriptionCompletedEvent).subscribe(lambda event: transcripts.append(event.content))

        async with stream.get(ModelResponse) as turn_done, agent.run():
            await turn_done

    assert transcripts == ["What is the weather"]


async def test_gemini_3_transcription_is_complete_as_it_arrives(certificate: Certificate) -> None:
    # Gemini 3.x Live sends one final transcription, without `finished`.
    async with LiveServer(
        certificate,
        heard("What is the weather today?"),
        said("Sunny."),
        gtypes.LiveServerContent(turn_complete=True),
    ) as server:
        stream = MemoryStream()
        agent = LiveAgent(
            "assistant",
            config=gemini.RealTimeConfig(
                "gemini-3.8-live",
                input=gemini.InputConfig(transcribe=True),
                client=server.client(),
            ),
            stream=stream,
        )
        transcription: list[TranscriptionChunkEvent | TranscriptionCompletedEvent] = []
        stream.where(TranscriptionChunkEvent | TranscriptionCompletedEvent).subscribe(
            lambda event: transcription.append(event)
        )

        async with stream.get(ModelResponse) as turn_done, agent.run():
            await turn_done

    assert transcription == [
        TranscriptionChunkEvent("What is the weather today?"),
        TranscriptionCompletedEvent("What is the weather today?"),
    ]


async def test_gemini_3_model_resource_name_is_complete_as_it_arrives(certificate: Certificate) -> None:
    async with LiveServer(
        certificate,
        heard("What is"),
        heard(" the weather today?"),
        gtypes.LiveServerContent(turn_complete=True),
    ) as server:
        stream = MemoryStream()
        agent = LiveAgent(
            "assistant",
            config=gemini.RealTimeConfig(
                "models/gemini-3.8-live",
                input=gemini.InputConfig(transcribe=True),
                client=server.client(),
            ),
            stream=stream,
        )
        transcripts: list[str] = []
        stream.where(TranscriptionCompletedEvent).subscribe(lambda event: transcripts.append(event.content))

        async with stream.get(ModelResponse) as turn_done, agent.run():
            await turn_done

    assert transcripts == ["What is", "the weather today?"]


async def test_gemini_3_5_translate_joins_fragments(certificate: Certificate) -> None:
    # The 3.5 translate model still sends fragments, as models before Gemini 3 do.
    async with LiveServer(
        certificate,
        heard("What is"),
        heard(" the weather today?"),
        gtypes.LiveServerContent(turn_complete=True),
    ) as server:
        stream = MemoryStream()
        agent = LiveAgent(
            "assistant",
            config=gemini.RealTimeConfig(
                "gemini-3.5-live-translate-preview",
                input=gemini.InputConfig(transcribe=True),
                client=server.client(),
            ),
            stream=stream,
        )
        transcripts: list[str] = []
        stream.where(TranscriptionCompletedEvent).subscribe(lambda event: transcripts.append(event.content))

        async with stream.get(ModelResponse) as turn_done, agent.run():
            await turn_done

    assert transcripts == ["What is the weather today?"]

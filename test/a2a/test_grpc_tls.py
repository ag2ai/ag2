# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from ipaddress import ip_address

import grpc
import grpc.aio
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, build_card, secure_grpc_channel_factory
from ag2.a2a.testing import pick_free_port
from ag2.a2a.transports import default_grpc_channel_factory
from ag2.testing import TestConfig

HOST = "127.0.0.1"


@pytest.fixture(scope="module")
def self_signed() -> tuple[bytes, bytes]:
    """Return a private key and certificate PEM pair for 127.0.0.1."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HOST)])
    cert = (
        x509
        .CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=5))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ip_address(HOST))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return key_pem, cert.public_bytes(serialization.Encoding.PEM)


@pytest.mark.asyncio
async def test_default_factory_accepts_tls_prefixes() -> None:
    for url in (f"grpcs://{HOST}:50051", f"grpc+tls://{HOST}:50051"):
        channel = default_grpc_channel_factory(url)
        assert isinstance(channel, grpc.aio.Channel)
        await channel.close()


@pytest.mark.asyncio
async def test_default_factory_keeps_insecure_prefixes() -> None:
    for url in (f"grpc://{HOST}:50051", f"grpc+insecure://{HOST}:50051", f"{HOST}:50051"):
        channel = default_grpc_channel_factory(url)
        assert isinstance(channel, grpc.aio.Channel)
        await channel.close()


@pytest.mark.asyncio
async def test_secure_factory_builds_channels(self_signed: tuple[bytes, bytes]) -> None:
    _, cert_pem = self_signed
    factory = secure_grpc_channel_factory(
        grpc.ssl_channel_credentials(root_certificates=cert_pem),
    )
    channel = factory(f"{HOST}:50051")
    assert isinstance(channel, grpc.aio.Channel)
    await channel.close()


@pytest.mark.asyncio
async def test_grpc_tls_round_trip(self_signed: tuple[bytes, bytes]) -> None:
    key_pem, cert_pem = self_signed
    server_agent = Agent("tls-server", config=TestConfig("secure pong"))
    server = A2AServer(server_agent)

    grpc_url = f"{HOST}:{pick_free_port(HOST)}"
    card = build_card(server_agent, url=grpc_url, transports=("grpc",), grpc_url=grpc_url)
    grpc_server = server.build_grpc(
        bind=grpc_url,
        grpc_url=grpc_url,
        card=card,
        server_credentials=grpc.ssl_server_credentials([(key_pem, cert_pem)]),
    )
    await grpc_server.start()
    try:
        client = Agent(
            "tls-client",
            config=A2AConfig(
                card_url=grpc_url,
                preset_card=card,
                prefer="grpc",
                streaming=False,
                grpc_channel_factory=secure_grpc_channel_factory(
                    grpc.ssl_channel_credentials(root_certificates=cert_pem),
                ),
            ),
        )
        reply = await client.ask("ping")
        assert reply.body == "secure pong"
    finally:
        await grpc_server.stop(grace=None)

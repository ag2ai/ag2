# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import httpx
import pytest
from a2a.client.client_factory import TransportProtocol
from a2a.server.context import ServerCallContext
from a2a.types import AgentCard, AgentInterface
from a2a.utils.signing import create_agent_card_signer, create_signature_verifier
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, CardSigner, CardVerifier, build_card
from ag2.a2a.errors import A2ACardSignatureError
from ag2.a2a.testing import make_test_client_factory
from ag2.testing import TestConfig


def _keypair() -> tuple[bytes, bytes]:
    private = ec.generate_private_key(ec.SECP256R1())
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


@pytest.fixture(scope="module")
def ec_keys() -> tuple[bytes, bytes]:
    return _keypair()


@pytest.fixture(scope="module")
def other_ec_keys() -> tuple[bytes, bytes]:
    return _keypair()


@pytest.fixture
def signer(ec_keys: tuple[bytes, bytes]) -> CardSigner:
    return create_agent_card_signer(
        ec_keys[0],
        {"kid": "test-key", "alg": "ES256", "jku": None, "typo": None},
    )


@pytest.fixture
def verifier(ec_keys: tuple[bytes, bytes]) -> CardVerifier:
    return create_signature_verifier(lambda kid, jku: ec_keys[1], ["ES256"])


@pytest.fixture
def wrong_key_verifier(other_ec_keys: tuple[bytes, bytes]) -> CardVerifier:
    return create_signature_verifier(lambda kid, jku: other_ec_keys[1], ["ES256"])


def _card() -> AgentCard:
    card = AgentCard()

    card.name = "preset-server"
    card.description = "d"
    card.version = "1.0.0"
    card.supported_interfaces.append(
        AgentInterface(
            url="http://test",
            protocol_binding=TransportProtocol.JSONRPC.value,
            protocol_version="1.0",
        ),
    )
    return card


async def _fetch_card_json(app: object) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    return resp.json()


def _signed_pair_config(signer: CardSigner | None, verifier_callable: CardVerifier) -> A2AConfig:
    agent = Agent("signed-server", config=TestConfig("pong"))
    server = A2AServer(agent, card_signer=signer)
    factory = make_test_client_factory(server, url="http://test")
    return A2AConfig(
        card_url="http://test",
        httpx_client_factory=factory,
        card_signature_verifier=verifier_callable,
    )


@pytest.mark.asyncio
async def test_served_card_is_signed(signer: CardSigner) -> None:
    agent = Agent("signed-server", config=TestConfig("hi"))
    server = A2AServer(agent, card_signer=signer)

    payload = await _fetch_card_json(server.build_jsonrpc(url="http://test"))

    assert payload.get("signatures"), "served card must carry JWS signatures"


@pytest.mark.asyncio
async def test_served_card_without_signer_is_unsigned() -> None:
    agent = Agent("plain-server", config=TestConfig("hi"))
    server = A2AServer(agent)  # no card_signer — default behavior unchanged

    payload = await _fetch_card_json(server.build_jsonrpc(url="http://test"))

    assert not payload.get("signatures")


@pytest.mark.asyncio
async def test_client_verifies_signed_card_and_talks(signer: CardSigner, verifier: CardVerifier) -> None:
    client = Agent("client", config=_signed_pair_config(signer, verifier))
    reply = await client.ask("ping")
    assert reply.body == "pong"


@pytest.mark.asyncio
async def test_card_modifier_output_is_resigned(signer: CardSigner, verifier: CardVerifier) -> None:
    # The per-request card_modifier mutates the card AFTER the static
    # signature; the client's verifier only passes if the served output
    # was re-signed post-modification.
    async def modifier(card: AgentCard) -> AgentCard:
        out = AgentCard()
        out.CopyFrom(card)
        out.description = "modified per request"
        return out

    agent = Agent("signed-server", config=TestConfig("pong"))
    server = A2AServer(agent, card_signer=signer, card_modifier=modifier)
    factory = make_test_client_factory(server, url="http://test")
    config = A2AConfig(
        card_url="http://test",
        httpx_client_factory=factory,
        card_signature_verifier=verifier,
    )

    client = Agent("client", config=config)
    reply = await client.ask("ping")
    assert reply.body == "pong"


@pytest.mark.asyncio
async def test_extended_card_modifier_output_is_resigned(signer: CardSigner, verifier: CardVerifier) -> None:
    # Same re-signing guarantee for the extended-card path: the client
    # verifies the (modified) extended card before adopting it.
    async def extended_modifier(card: AgentCard, context: ServerCallContext) -> AgentCard:
        out = AgentCard()
        out.CopyFrom(card)
        out.description = "extended, modified per request"
        return out

    extended = AgentCard()
    extended.name = "ext-server"
    extended.description = "extended"
    extended.version = "1.0.0"

    agent = Agent("ext-server", config=TestConfig("pong"))
    server = A2AServer(
        agent,
        extended_card=extended,
        extended_card_modifier=extended_modifier,
        card_signer=signer,
    )
    factory = make_test_client_factory(server, url="http://test")
    config = A2AConfig(
        card_url="http://test",
        httpx_client_factory=factory,
        card_signature_verifier=verifier,
    )

    client = Agent("client", config=config)
    reply = await client.ask("ping")
    assert reply.body == "pong"


@pytest.mark.asyncio
async def test_unsigned_card_with_verifier_raises(verifier: CardVerifier) -> None:
    client = Agent("client", config=_signed_pair_config(None, verifier))
    with pytest.raises(A2ACardSignatureError):
        await client.ask("ping")


@pytest.mark.asyncio
async def test_wrong_key_raises(signer: CardSigner, wrong_key_verifier: CardVerifier) -> None:
    client = Agent("client", config=_signed_pair_config(signer, wrong_key_verifier))
    with pytest.raises(A2ACardSignatureError):
        await client.ask("ping")


@pytest.mark.asyncio
async def test_tampered_preset_card_raises(signer: CardSigner, verifier: CardVerifier) -> None:
    signed = signer(_card())
    tampered = AgentCard()
    tampered.CopyFrom(signed)
    tampered.description = "evil"

    config = A2AConfig.from_card(
        tampered,
        card_url="http://test",
        card_signature_verifier=verifier,
    )
    client = Agent("client", config=config)
    with pytest.raises(A2ACardSignatureError):
        await client.ask("ping")


@pytest.mark.asyncio
async def test_unsigned_extended_card_raises(signer: CardSigner, verifier: CardVerifier) -> None:
    # Signed base card, UNSIGNED extended card: the extended-card fetch path
    # must be verified too, otherwise it silently replaces the verified card.
    agent = Agent("ext-server", config=TestConfig("pong"))
    unsigned_extended = AgentCard()
    unsigned_extended.name = "ext-server"
    unsigned_extended.description = "extended, unsigned"
    unsigned_extended.version = "1.0.0"

    server = A2AServer(agent, extended_card=unsigned_extended)
    # Bypass the server-level signer so ONLY the base card is signed. The
    # builder's clone flips ``capabilities.extended_agent_card`` after our
    # manual signature, so set it before signing to keep the JWS valid.
    base = build_card(agent, url="http://test")
    base.capabilities.extended_agent_card = True
    base = signer(base)
    app = server.build_jsonrpc(url="http://test", card=base)
    transport = httpx.ASGITransport(app=app)

    config = A2AConfig(
        card_url="http://test",
        httpx_client_factory=lambda: httpx.AsyncClient(transport=transport, base_url="http://test"),
        card_signature_verifier=verifier,
    )
    client = Agent("client", config=config)
    with pytest.raises(A2ACardSignatureError, match="extended agent card"):
        await client.ask("ping")

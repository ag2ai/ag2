# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import httpx
import pytest
from a2a.client.client_factory import TransportProtocol
from a2a.server.context import ServerCallContext
from a2a.types import AgentCard, AgentInterface, AgentSkill
from a2a.utils.signing import create_agent_card_signer, create_signature_verifier
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from google.protobuf.json_format import ParseDict

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.client import CardVerifier
from ag2.a2a.errors import A2ACardSignatureError, A2AStaleCardSignatureError
from ag2.a2a.server import CardSigner
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
        {"kid": "test-key", "alg": "ES256", "jku": None, "typ": "JOSE"},
    )


@pytest.fixture
def other_signer(other_ec_keys: tuple[bytes, bytes]) -> CardSigner:
    return create_agent_card_signer(
        other_ec_keys[0],
        {"kid": "rotated-key", "alg": "ES256", "jku": None, "typ": "JOSE"},
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


def _extended_card() -> AgentCard:
    card = AgentCard()

    card.name = "ext-server"
    card.description = "extended"
    card.version = "1.0.0"
    return card


async def _fetch_card_json(app: object) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    return resp.json()


def _signature_validity(payload: dict, verifier_callable: CardVerifier) -> list[bool]:
    """Verify each signature on a served card independently, in wire order."""
    # The SDK verifier accepts a card as long as *one* signature validates,
    # so a whole-card assert can't catch a stale signature riding along.
    validity = []
    for signature in payload.get("signatures", []):
        single = {**payload, "signatures": [signature]}
        try:
            verifier_callable(ParseDict(single, AgentCard(), ignore_unknown_fields=True))
            validity.append(True)
        except Exception:
            validity.append(False)
    return validity


def _without_signatures(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "signatures"}


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
class TestServedCard:
    async def test_card_is_signed(self, signer: CardSigner, verifier: CardVerifier) -> None:
        agent = Agent("signed-server", config=TestConfig("hi"))
        server = A2AServer(agent, card_signer=signer)

        payload = await _fetch_card_json(server.build_jsonrpc(url="http://test"))

        assert _signature_validity(payload, verifier) == [True]

    async def test_card_without_signer_is_unsigned(self) -> None:
        agent = Agent("plain-server", config=TestConfig("hi"))
        server = A2AServer(agent)  # no card_signer — default behavior unchanged

        payload = await _fetch_card_json(server.build_jsonrpc(url="http://test"))

        assert not payload.get("signatures")

    async def test_extended_card_is_not_mutated_by_builders(self, signer: CardSigner) -> None:
        # The same extended card object is handed to every transport builder;
        # signing it must not accumulate on the card the caller constructed.
        extended = _extended_card()
        agent = Agent("ext-server", config=TestConfig("pong"))
        server = A2AServer(agent, extended_card=extended, card_signer=signer)

        server.build_jsonrpc(url="http://test")
        server.build_rest(url="http://test")

        assert not extended.signatures

    async def test_key_rotation_serves_both_signatures(
        self, signer: CardSigner, other_signer: CardSigner, verifier: CardVerifier
    ) -> None:
        # Signing a copy with prior signatures dropped must not break
        # rotation: a composed signer still lands both, and a client that
        # trusts only one of the two keys connects.
        def rotating(card: AgentCard) -> AgentCard:
            return other_signer(signer(card))

        agent = Agent("signed-server", config=TestConfig("pong"))
        server = A2AServer(agent, card_signer=rotating)

        payload = await _fetch_card_json(server.build_jsonrpc(url="http://test"))
        assert _signature_validity(payload, verifier) == [True, False]  # old key, then rotated key

        config = A2AConfig(
            card_url="http://test",
            httpx_client_factory=make_test_client_factory(server, url="http://test"),
            card_signature_verifier=verifier,  # trusts only the OLD key
        )
        reply = await Agent("client", config=config).ask("ping")

        assert reply.body == "pong"


@pytest.mark.asyncio
class TestCardModifiers:
    async def test_output_is_resigned(self, signer: CardSigner, verifier: CardVerifier) -> None:
        # The per-request card_modifier mutates the card AFTER the static
        # signature; the served card must carry the post-modification
        # signature and nothing else.
        async def modifier(card: AgentCard) -> AgentCard:
            out = AgentCard()
            out.CopyFrom(card)
            out.description = "modified per request"
            return out

        agent = Agent("signed-server", config=TestConfig("pong"))
        server = A2AServer(agent, card_signer=signer, card_modifier=modifier)
        config = A2AConfig(
            card_url="http://test",
            httpx_client_factory=make_test_client_factory(server, url="http://test"),
            card_signature_verifier=verifier,
        )

        reply = await Agent("client", config=config).ask("ping")
        payload = await _fetch_card_json(server.build_jsonrpc(url="http://test"))

        assert reply.body == "pong"
        assert payload["description"] == "modified per request"
        assert _signature_validity(payload, verifier) == [True]

    async def test_in_place_modifier_does_not_drift(self, signer: CardSigner, verifier: CardVerifier) -> None:
        # The SDK hands the modifier the one long-lived card object shared by
        # every request. A modifier that mutates it and returns it — the
        # natural thing to write — must not make the served card grow request
        # over request, in signatures or in payload.
        async def mutating_modifier(card: AgentCard) -> AgentCard:
            card.skills.append(AgentSkill(id="per-request", name="per-request", description="d", tags=["t"]))
            return card

        agent = Agent("signed-server", config=TestConfig("pong"))
        server = A2AServer(agent, card_signer=signer, card_modifier=mutating_modifier)
        app = server.build_jsonrpc(url="http://test")

        payloads = [await _fetch_card_json(app) for _ in range(3)]

        served = [_without_signatures(payload) for payload in payloads]
        assert served == [served[0], served[0], served[0]]
        assert [_signature_validity(payload, verifier) for payload in payloads] == [[True], [True], [True]]

    async def test_extended_output_is_resigned(self, signer: CardSigner, verifier: CardVerifier) -> None:
        # Same re-signing guarantee for the extended-card path: the client
        # verifies the (modified) extended card before adopting it.
        async def extended_modifier(card: AgentCard, context: ServerCallContext) -> AgentCard:
            out = AgentCard()
            out.CopyFrom(card)
            out.description = "extended, modified per request"
            return out

        agent = Agent("ext-server", config=TestConfig("pong"))
        server = A2AServer(
            agent,
            extended_card=_extended_card(),
            extended_card_modifier=extended_modifier,
            card_signer=signer,
        )
        config = A2AConfig(
            card_url="http://test",
            httpx_client_factory=make_test_client_factory(server, url="http://test"),
            card_signature_verifier=verifier,
        )

        reply = await Agent("client", config=config).ask("ping")

        assert reply.body == "pong"


@pytest.mark.asyncio
class TestClientVerification:
    async def test_signed_card_is_accepted(self, signer: CardSigner, verifier: CardVerifier) -> None:
        client = Agent("client", config=_signed_pair_config(signer, verifier))

        reply = await client.ask("ping")

        assert reply.body == "pong"

    async def test_unsigned_card_raises(self, verifier: CardVerifier) -> None:
        client = Agent("client", config=_signed_pair_config(None, verifier))

        with pytest.raises(A2ACardSignatureError, match="fetched agent card"):
            await client.ask("ping")

    async def test_wrong_key_raises(self, signer: CardSigner, wrong_key_verifier: CardVerifier) -> None:
        client = Agent("client", config=_signed_pair_config(signer, wrong_key_verifier))

        with pytest.raises(A2ACardSignatureError, match="fetched agent card"):
            await client.ask("ping")

    async def test_tampered_preset_card_raises(self, signer: CardSigner, verifier: CardVerifier) -> None:
        tampered = AgentCard()
        tampered.CopyFrom(signer(_card()))
        tampered.description = "evil"

        config = A2AConfig.from_card(
            tampered,
            card_url="http://test",
            card_signature_verifier=verifier,
        )
        client = Agent("client", config=config)

        with pytest.raises(A2ACardSignatureError, match="preset agent card"):
            await client.ask("ping")

    async def test_unsigned_extended_card_raises(self, signer: CardSigner, verifier: CardVerifier) -> None:
        # Signed base card, UNSIGNED extended card: the extended-card fetch
        # path must be verified too, otherwise it silently replaces the
        # verified card.
        agent = Agent("ext-server", config=TestConfig("pong"))
        unsigned_extended = _extended_card()
        server = A2AServer(agent, extended_card=unsigned_extended)

        # Bypass the server-level signer so ONLY the base card is signed.
        # The flag has to be set before signing — see TestPresignedCard.
        base = build_card(agent, url="http://test")
        base.capabilities.extended_agent_card = True
        app = server.build_jsonrpc(url="http://test", card=signer(base))
        transport = httpx.ASGITransport(app=app)

        config = A2AConfig(
            card_url="http://test",
            httpx_client_factory=lambda: httpx.AsyncClient(transport=transport, base_url="http://test"),
            card_signature_verifier=verifier,
        )
        client = Agent("client", config=config)

        with pytest.raises(A2ACardSignatureError, match="extended agent card"):
            await client.ask("ping")

    async def test_key_provider_error_is_wrapped(self, signer: CardSigner) -> None:
        # The SDK verifier only guards its key lookup against PyJWTError, so
        # a registry miss escapes as its own type. Callers should still only
        # need to catch ag2's error.
        def raising_key_provider(kid: str | None, jku: str | None) -> bytes:
            raise KeyError(f"unknown kid {kid!r}")

        verifier = create_signature_verifier(raising_key_provider, ["ES256"])
        client = Agent("client", config=_signed_pair_config(signer, verifier))

        with pytest.raises(A2ACardSignatureError, match="KeyError") as excinfo:
            await client.ask("ping")

        assert isinstance(excinfo.value.__cause__, KeyError)


@pytest.mark.asyncio
class TestPresignedCard:
    # Cards the caller signed themselves and handed to a ``build_*`` method.

    async def test_flipped_capability_is_rejected(self, signer: CardSigner) -> None:
        # Serving a caller-signed card whose capability flags the server has
        # to flip would put a signature on the wire that no longer matches
        # its payload, and without a card_signer there is no key to redo it.
        agent = Agent("ext-server", config=TestConfig("pong"))
        signed = signer(build_card(agent, url="http://test"))
        server = A2AServer(agent, extended_card=_extended_card())

        with pytest.raises(A2AStaleCardSignatureError, match="extended_agent_card"):
            server.build_jsonrpc(url="http://test", card=signed)

    async def test_preset_capability_is_served(self, signer: CardSigner, verifier: CardVerifier) -> None:
        # The supported way to serve your own signature: set the flags the
        # server derives before signing, so nothing has to change afterwards.
        agent = Agent("ext-server", config=TestConfig("pong"))
        base = build_card(agent, url="http://test")
        base.capabilities.extended_agent_card = True
        server = A2AServer(agent, extended_card=_extended_card())

        payload = await _fetch_card_json(server.build_jsonrpc(url="http://test", card=signer(base)))

        assert _signature_validity(payload, verifier) == [True]

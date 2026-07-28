# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.server.context import ServerCallContext
from a2a.types import AgentCard
from a2a.utils.signing import create_agent_card_signer, create_signature_verifier
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from ag2.a2a.transports._common import sign_card, wrap_card_modifier, wrap_extended_card_modifier


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
def signer(ec_keys):
    return create_agent_card_signer(
        ec_keys[0],
        {"kid": "test-key", "alg": "ES256", "jku": None, "typo": None},
    )


@pytest.fixture
def verifier(ec_keys):
    return create_signature_verifier(lambda kid, jku: ec_keys[1], ["ES256"])


@pytest.fixture
def wrong_key_verifier(other_ec_keys):
    return create_signature_verifier(lambda kid, jku: other_ec_keys[1], ["ES256"])


def _card() -> AgentCard:
    card = AgentCard()
    card.name = "helper-test"
    card.description = "d"
    card.version = "1.0.0"
    return card


def test_sign_card_none_signer_is_passthrough() -> None:
    card = _card()
    assert sign_card(card, None) is card


def test_sign_card_adds_signature(signer, verifier) -> None:
    signed = sign_card(_card(), signer)
    assert len(signed.signatures) >= 1
    verifier(signed)  # must not raise


@pytest.mark.asyncio
async def test_wrap_card_modifier_resigns_output(signer, verifier) -> None:
    async def modifier(card: AgentCard) -> AgentCard:
        out = AgentCard()
        out.CopyFrom(card)
        out.description = "modified"
        return out

    wrapped = wrap_card_modifier(modifier, signer)
    result = await wrapped(_card())
    assert result.description == "modified"
    verifier(result)  # signature valid AFTER modification


def test_wrap_card_modifier_none_cases(signer) -> None:
    assert wrap_card_modifier(None, signer) is None

    async def modifier(card: AgentCard) -> AgentCard:
        return card

    assert wrap_card_modifier(modifier, None) is modifier


@pytest.mark.asyncio
async def test_wrap_extended_card_modifier_resigns_output(signer, verifier) -> None:
    async def modifier(card: AgentCard, context: ServerCallContext) -> AgentCard:
        out = AgentCard()
        out.CopyFrom(card)
        out.description = "extended"
        return out

    wrapped = wrap_extended_card_modifier(modifier, signer)
    result = await wrapped(_card(), ServerCallContext())
    assert result.description == "extended"
    verifier(result)

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ActorIdentity serialization and immutability semantics."""

from __future__ import annotations

import pytest

from autogen.beta.network.identity import ActorIdentity, AuthBlock


def _sample() -> ActorIdentity:
    return ActorIdentity(
        name="ag2:researcher:1",
        owner="ag2",
        version="1",
        display="Research Agent",
        runtime_kind="python",
        model_hint="anthropic/claude-opus-4-6",
        capabilities=["research", "summarization"],
        summary="Produces cited reviews.",
        domains=["biomed"],
        strengths="Long-context synthesis.",
        skill_md="## Researcher\n\nPrefers consulting sessions.",
        auth=AuthBlock(scheme="api_key", key_fingerprint="sha256:abc"),
    )


def test_identity_roundtrip_via_dict() -> None:
    ident = _sample()
    restored = ActorIdentity.from_dict(ident.to_dict())
    assert restored.name == ident.name
    assert restored.capabilities == ident.capabilities
    assert restored.auth.scheme == "api_key"
    assert restored.auth.key_fingerprint == "sha256:abc"
    assert restored.skill_md == ident.skill_md
    assert restored.actor_id is None


def test_identity_roundtrip_via_json() -> None:
    ident = _sample()
    restored = ActorIdentity.from_json(ident.to_json())
    assert restored.to_dict() == ident.to_dict()
    assert restored.skill_md == ident.skill_md


def test_identity_skill_md_omitted_when_none() -> None:
    ident = ActorIdentity(name="ag2:x:1")
    assert "skill_md" not in ident.to_dict()
    restored = ActorIdentity.from_dict(ident.to_dict())
    assert restored.skill_md is None


def test_identity_has_no_actor_id_before_registration() -> None:
    assert _sample().actor_id is None


def test_with_actor_id_returns_copy() -> None:
    ident = _sample()
    stamped = ident.with_actor_id("01932f8a-stamped")
    assert stamped is not ident
    assert stamped.actor_id == "01932f8a-stamped"
    assert ident.actor_id is None
    # Other fields must be preserved.
    assert stamped.name == ident.name
    assert stamped.capabilities == ident.capabilities
    assert stamped.auth.scheme == ident.auth.scheme


def test_with_actor_id_preserves_auth_block_type() -> None:
    stamped = _sample().with_actor_id("01932-abc")
    assert isinstance(stamped.auth, AuthBlock)


def test_auth_block_optional_fields_omitted_from_dict_when_none() -> None:
    block = AuthBlock(scheme="none")
    assert block.to_dict() == {"scheme": "none", "claim": {}}


def test_auth_block_roundtrip_preserves_claim() -> None:
    block = AuthBlock(
        scheme="jwt",
        issuer="https://auth.ag2.cloud",
        audience="hub-prod",
        key_fingerprint="sha256:xyz",
        claim={"sub": "researcher-1", "role": "worker"},
    )
    restored = AuthBlock.from_dict(block.to_dict())
    assert restored == block


def test_identity_defaults_scheme_to_none_when_missing() -> None:
    ident = ActorIdentity.from_dict({"name": "ag2:x:1"})
    assert ident.auth.scheme == "none"
    assert ident.runtime_kind == "python"
    assert ident.framework == "ag2-beta"


@pytest.mark.parametrize(
    "field",
    [
        "display",
        "model_hint",
        "locale",
        "timezone",
    ],
)
def test_identity_optional_fields_omitted_when_none(field: str) -> None:
    ident = ActorIdentity(name="ag2:x:1")
    assert field not in ident.to_dict()

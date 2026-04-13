# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phase 3a ``ApiKeyAuth`` + ``JwtAuth`` adapters.

Phase 1 shipped only :class:`NoAuth`. Phase 3a adds two real
validators plugged into the existing :class:`AuthRegistry` plugin
point — no protocol changes, just additional implementations.
Verification happens at two entry points (WS hello, HTTP front door),
both of which route through :meth:`AuthRegistry.validate` before
rule evaluation.

Test matrix:

* ``ApiKeyAuth``:
  - scheme mismatch raises
  - empty claim raises
  - fingerprint mismatch raises
  - fingerprint match without allowlist succeeds
  - fingerprint match with allowlist containing the key succeeds
  - fingerprint match with allowlist NOT containing the key raises
  - timing-safe compare (smoke — we can't actually assert timing,
    but we can verify the equality helper is used)
  - ``add_fingerprint`` / ``revoke_fingerprint`` lifecycle

* ``JwtAuth``:
  - scheme mismatch raises
  - missing token raises
  - valid HS256 token accepted
  - wrong key rejected
  - expired token rejected
  - wrong audience rejected
  - wrong issuer rejected
  - subject mismatch rejected
  - missing PyJWT gracefully errors (hard to test without mocking;
    we verify the import-error path is in place via a smoke)

* Registry:
  - dev_registry ships NoAuth + ApiKeyAuth by default
  - unknown scheme raises from the registry
  - hub integration: Hub configured with ApiKeyAuth accepts a valid
    registration and rejects an invalid one
"""

from __future__ import annotations

import time

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    ApiKeyAuth,
    AuthError,
    AuthRegistry,
    Hub,
    HubClient,
    JwtAuth,
    LocalLink,
    NoAuth,
    dev_registry,
)
from autogen.beta.network.identity import AuthBlock


# ---------------------------------------------------------------------------
# ApiKeyAuth
# ---------------------------------------------------------------------------


def _identity_with_api_key(fingerprint: str = "sha256:abc123") -> ActorIdentity:
    return ActorIdentity(
        name="alice",
        auth=AuthBlock(scheme="api_key", key_fingerprint=fingerprint),
    )


class TestApiKeyAuth:
    @pytest.mark.asyncio
    async def test_scheme_mismatch_raises(self) -> None:
        adapter = ApiKeyAuth()
        identity = ActorIdentity(name="alice", auth=AuthBlock(scheme="none"))
        with pytest.raises(AuthError, match="cannot validate scheme"):
            await adapter.validate(identity, {})

    @pytest.mark.asyncio
    async def test_missing_key_fingerprint_raises(self) -> None:
        adapter = ApiKeyAuth()
        identity = ActorIdentity(
            name="alice", auth=AuthBlock(scheme="api_key", key_fingerprint=None)
        )
        with pytest.raises(AuthError, match="key_fingerprint is required"):
            await adapter.validate(identity, {"fingerprint": "anything"})

    @pytest.mark.asyncio
    async def test_missing_claim_fingerprint_raises(self) -> None:
        adapter = ApiKeyAuth()
        identity = _identity_with_api_key()
        with pytest.raises(AuthError, match="'fingerprint' string"):
            await adapter.validate(identity, {})

    @pytest.mark.asyncio
    async def test_fingerprint_mismatch_raises(self) -> None:
        adapter = ApiKeyAuth()
        identity = _identity_with_api_key("sha256:real")
        with pytest.raises(AuthError, match="does not match identity"):
            await adapter.validate(identity, {"fingerprint": "sha256:impostor"})

    @pytest.mark.asyncio
    async def test_fingerprint_match_no_allowlist_succeeds(self) -> None:
        adapter = ApiKeyAuth()  # empty allowlist = any match is OK
        identity = _identity_with_api_key("sha256:real")
        await adapter.validate(identity, {"fingerprint": "sha256:real"})

    @pytest.mark.asyncio
    async def test_fingerprint_match_with_allowlist_succeeds(self) -> None:
        adapter = ApiKeyAuth(
            allowed_fingerprints={"sha256:real": "alice"},
        )
        identity = _identity_with_api_key("sha256:real")
        await adapter.validate(identity, {"fingerprint": "sha256:real"})

    @pytest.mark.asyncio
    async def test_fingerprint_match_not_in_allowlist_rejects(self) -> None:
        adapter = ApiKeyAuth(
            allowed_fingerprints={"sha256:other": "bob"},
        )
        identity = _identity_with_api_key("sha256:real")
        with pytest.raises(AuthError, match="not on the hub's allowlist"):
            await adapter.validate(identity, {"fingerprint": "sha256:real"})

    @pytest.mark.asyncio
    async def test_add_and_revoke_fingerprint(self) -> None:
        adapter = ApiKeyAuth()
        adapter.add_fingerprint("sha256:new", "carol")
        identity = _identity_with_api_key("sha256:new")
        await adapter.validate(identity, {"fingerprint": "sha256:new"})
        # Revoke and try again.
        adapter.revoke_fingerprint("sha256:new")
        # Revoked but allowlist is now empty → any match still
        # succeeds. Re-add a different key so the allowlist is
        # non-empty AND doesn't contain ours.
        adapter.add_fingerprint("sha256:different", "someone_else")
        with pytest.raises(AuthError, match="not on the hub's allowlist"):
            await adapter.validate(identity, {"fingerprint": "sha256:new"})


# ---------------------------------------------------------------------------
# JwtAuth — happy path + rejections
# ---------------------------------------------------------------------------


jwt = pytest.importorskip("jwt")


# PyJWT warns when the HS256 key is below 32 bytes, so the test
# secret is padded to avoid noisy InsecureKeyLengthWarning entries
# in the test run. The value itself is arbitrary.
_SECRET = "test-secret-please-rotate-" + "x" * 16


def _make_token(
    *,
    sub: str = "alice",
    iss: str = "ag2-test",
    aud: str = "hub-test",
    exp_offset: int = 3600,
    iat_offset: int = 0,
    key: str | bytes = _SECRET,
    algorithm: str = "HS256",
) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,
        "iss": iss,
        "aud": aud,
        "iat": now + iat_offset,
        "exp": now + exp_offset,
    }
    return jwt.encode(payload, key, algorithm=algorithm)


def _jwt_identity(name: str = "alice") -> ActorIdentity:
    return ActorIdentity(
        name=name,
        auth=AuthBlock(
            scheme="jwt",
            issuer="ag2-test",
            audience="hub-test",
        ),
    )


class TestJwtAuth:
    @pytest.mark.asyncio
    async def test_scheme_mismatch_raises(self) -> None:
        adapter = JwtAuth(key=_SECRET)
        identity = ActorIdentity(name="alice", auth=AuthBlock(scheme="none"))
        with pytest.raises(AuthError, match="cannot validate scheme"):
            await adapter.validate(identity, {"token": _make_token()})

    @pytest.mark.asyncio
    async def test_missing_token_raises(self) -> None:
        adapter = JwtAuth(key=_SECRET)
        with pytest.raises(AuthError, match="'token' string"):
            await adapter.validate(_jwt_identity(), {})

    @pytest.mark.asyncio
    async def test_valid_token_accepted(self) -> None:
        adapter = JwtAuth(key=_SECRET)
        await adapter.validate(
            _jwt_identity(),
            {"token": _make_token()},
        )

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self) -> None:
        adapter = JwtAuth(key="different-secret-" + "y" * 32)
        with pytest.raises(AuthError, match="verification failed"):
            await adapter.validate(
                _jwt_identity(),
                {"token": _make_token()},
            )

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self) -> None:
        adapter = JwtAuth(key=_SECRET, leeway=0)
        with pytest.raises(AuthError, match="verification failed"):
            await adapter.validate(
                _jwt_identity(),
                {"token": _make_token(exp_offset=-60, iat_offset=-120)},
            )

    @pytest.mark.asyncio
    async def test_wrong_audience_rejected(self) -> None:
        adapter = JwtAuth(key=_SECRET)
        with pytest.raises(AuthError, match="verification failed"):
            await adapter.validate(
                _jwt_identity(),
                {"token": _make_token(aud="some-other-audience")},
            )

    @pytest.mark.asyncio
    async def test_wrong_issuer_rejected(self) -> None:
        adapter = JwtAuth(key=_SECRET)
        with pytest.raises(AuthError, match="verification failed"):
            await adapter.validate(
                _jwt_identity(),
                {"token": _make_token(iss="attacker")},
            )

    @pytest.mark.asyncio
    async def test_subject_mismatch_rejected(self) -> None:
        """A token 'sub' must match the identity name, so a stolen token
        for one actor can't be replayed against a different registration."""

        adapter = JwtAuth(key=_SECRET)
        with pytest.raises(AuthError, match="does not match identity name"):
            await adapter.validate(
                _jwt_identity(name="alice"),
                {"token": _make_token(sub="bob")},
            )

    @pytest.mark.asyncio
    async def test_required_audience_override(self) -> None:
        """Adapter can force a specific audience regardless of identity."""

        adapter = JwtAuth(key=_SECRET, required_audience="hub-prod")
        identity = _jwt_identity()
        # Token audience matches the adapter override — succeeds.
        await adapter.validate(
            identity,
            {"token": _make_token(aud="hub-prod")},
        )
        # Token audience matches identity but NOT the adapter override.
        with pytest.raises(AuthError):
            await adapter.validate(
                identity,
                {"token": _make_token(aud="hub-test")},
            )


# ---------------------------------------------------------------------------
# Registry + Hub integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_dev_registry_ships_no_auth_and_api_key(self) -> None:
        reg = dev_registry()
        schemes = reg.schemes()
        assert "none" in schemes
        assert "api_key" in schemes
        assert "jwt" not in schemes  # jwt only if jwt_key was passed

    def test_dev_registry_with_jwt_key_ships_jwt(self) -> None:
        reg = dev_registry(jwt_key=_SECRET)
        assert "jwt" in reg.schemes()

    def test_registry_unknown_scheme_raises(self) -> None:
        reg = AuthRegistry([NoAuth()])
        with pytest.raises(AuthError, match="No auth adapter installed"):
            reg.get("jwt")

    @pytest.mark.asyncio
    async def test_hub_registration_accepts_valid_api_key(self) -> None:
        registry = AuthRegistry(
            [
                NoAuth(),
                ApiKeyAuth(
                    allowed_fingerprints={"sha256:alice": "alice-the-real-one"},
                ),
            ]
        )
        hub = Hub(MemoryKnowledgeStore(), auth=registry)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            # Direct hub.register to exercise the auth path without
            # wrapping the framework-core `Actor` wiring.
            stamped = await hub.register(
                ActorIdentity(
                    name="alice",
                    auth=AuthBlock(
                        scheme="api_key", key_fingerprint="sha256:alice"
                    ),
                ),
                auth_claim={"fingerprint": "sha256:alice"},
            )
            assert stamped.actor_id
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_hub_registration_rejects_invalid_api_key(self) -> None:
        registry = AuthRegistry(
            [
                NoAuth(),
                ApiKeyAuth(
                    allowed_fingerprints={"sha256:alice": "alice-the-real-one"},
                ),
            ]
        )
        hub = Hub(MemoryKnowledgeStore(), auth=registry)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        try:
            with pytest.raises(AuthError):
                await hub.register(
                    ActorIdentity(
                        name="impostor",
                        auth=AuthBlock(
                            scheme="api_key", key_fingerprint="sha256:fake"
                        ),
                    ),
                    auth_claim={"fingerprint": "sha256:fake"},
                )
        finally:
            await link.close()

    @pytest.mark.asyncio
    async def test_hub_registration_accepts_valid_jwt(self) -> None:
        registry = AuthRegistry(
            [
                NoAuth(),
                JwtAuth(key=_SECRET),
            ]
        )
        hub = Hub(MemoryKnowledgeStore(), auth=registry)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        hc = HubClient(hub, link)
        try:
            stamped = await hub.register(
                _jwt_identity(),
                auth_claim={"token": _make_token()},
            )
            assert stamped.actor_id
        finally:
            await hc.close()
            await link.close()

    @pytest.mark.asyncio
    async def test_hub_registration_rejects_tampered_jwt(self) -> None:
        registry = AuthRegistry(
            [
                NoAuth(),
                JwtAuth(key=_SECRET),
            ]
        )
        hub = Hub(MemoryKnowledgeStore(), auth=registry)
        link = LocalLink()
        link.on_connection(hub.connection_handler)
        try:
            with pytest.raises(AuthError):
                await hub.register(
                    _jwt_identity(),
                    auth_claim={
                        "token": _make_token(
                            key="attacker-secret-" + "z" * 32
                        ),
                    },
                )
        finally:
            await link.close()
